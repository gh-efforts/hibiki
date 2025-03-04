use crate::sampler::Sampler;
use crate::{CompletionsTask, EmbeddingTask, KVCacheTypes};
use anyhow::{anyhow, ensure, Result};
use flume::{RecvTimeoutError, TryRecvError};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{LlamaModel};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_sys_2::{ggml_type, hibiki_common_speculative_are_compatible, LLAMA_POOLING_TYPE_NONE};
use std::cell::RefCell;
use std::cmp::{max, min};
use std::collections::{BTreeMap};
use std::num::NonZeroU32;
use std::ptr::slice_from_raw_parts;
use std::rc::Rc;
use std::slice;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;
use crate::metadata::ModelMetadata;
use crate::ngran_cache::{NgranCache, NGRAM_MAX, NGRAM_MIN};
use crate::radixtrie_kv_cache::RadixTrieKVCache;

#[derive(Copy, Clone, Eq, PartialEq)]
enum SeqState {
    Prefill,
    Decode,
}

struct Sequence {
    input_tokens: Vec<LlamaToken>,
    sampler: Sampler,
    callback: flume::Sender<LlamaToken>,
    token_pos: u32,
    maximum_tokens: u32,
    logits_pos: Option<i32>,
    state: SeqState
}

struct SequenceSlots<'a> {
    sequence_list: Vec<Option<Sequence>>,
    batch: &'a mut LlamaBatch,
    model: &'a LlamaModel,
}

impl <'a> SequenceSlots<'a> {
    fn new(n_task: u32, batch: &'a mut LlamaBatch, model: &'a LlamaModel) -> Self {
        let mut sequence_list = Vec::with_capacity(n_task as usize);

        for _ in 0..n_task {
            sequence_list.push(None);
        }

        Self {
            sequence_list,
            batch,
            model,
        }
    }

    fn len(&self) -> usize {
        self.sequence_list.iter()
            .filter(|v| v.is_some())
            .count()
    }

    fn put(&mut self, mut seq: Sequence, ctx: &mut LlamaContext, cache: &mut RadixTrieKVCache) -> Result<()> {
        for (i, slot) in self.sequence_list.iter_mut().enumerate() {
            if slot.is_some() {
                continue;
            }

            let raw_tokens = seq.input_tokens.iter().map(|t| t.0).collect::<Vec<_>>();

            match cache.get(&raw_tokens) {
                None => {
                    self.batch.add_sequence(&seq.input_tokens, i as i32, false)?;
                }
                Some((sub_seq_data, sub_seq_tokens_len)) => {
                    debug!("cache hit");

                    let sub_seq_len = min(seq.input_tokens.len() - 1, sub_seq_tokens_len);

                    unsafe {
                        let res = llama_cpp_sys_2::llama_state_seq_set_data(ctx.context.as_ptr(), sub_seq_data.as_ptr(), sub_seq_data.len(), i as i32);
                        ensure!(res != 0);
                        ctx.clear_kv_cache_seq(Some(i as u32), Some(sub_seq_len as u32), None)?;
                    }

                    for pos in sub_seq_len..seq.input_tokens.len() {
                        self.batch.add(seq.input_tokens[pos], pos as i32, &[i as i32], pos == seq.input_tokens.len() - 1)?
                    }
                }
            }

            seq.logits_pos = Some(self.batch.n_tokens() - 1);
            *slot = Some(seq);
            return Ok(());
        }
        Err(anyhow!("No available slot"))
    }

    fn batch_decode(&mut self, ctx: &mut LlamaContext, cache: &mut RadixTrieKVCache) -> Result<usize> {
        let slot_size = self.len();

        if slot_size == 0 {
            return Err(anyhow!("No sequence to decode"));
        }

        ctx.decode(self.batch)?;
        self.batch.clear();

        for (i, x) in &mut self.sequence_list.iter_mut().enumerate() {
            if let Some(seq) = x {
                if seq.state == SeqState::Prefill {
                    seq.state = SeqState::Decode;

                    unsafe {
                        let data_size = llama_cpp_sys_2::llama_state_seq_get_size(ctx.context.as_ptr(), i as i32);
                        let mut data = vec![0u8; data_size];
                        llama_cpp_sys_2::llama_state_seq_get_data(ctx.context.as_ptr(), data.as_mut_ptr(), data_size, i as i32);

                        let raw_input_tokens = seq.input_tokens.iter().map(|t| t.0).collect::<Vec<_>>();
                        cache.insert(raw_input_tokens, data)?;
                    }
                }
            }
        }
        Ok(slot_size)
    }

    fn batch_sample(&mut self, ctx: &mut LlamaContext) -> Result<usize> {
        let slot_size = self.len();

        if slot_size == 0 {
            return Err(anyhow!("No sequence to sample"));
        }

        let sequence_list = &mut self.sequence_list;

        for (i, slot) in sequence_list.iter_mut().enumerate() {
            if let Some(seq) = slot {
                let out_token = seq.sampler.sample(ctx, seq.logits_pos.take().unwrap());

                macro_rules! remove_slot {
                    () => {
                        *slot = None;
                        ctx.clear_kv_cache_seq(Some(i as u32), None, None)?;
                    };
                }

                if self.model.is_eog_token(out_token) {
                    remove_slot!();
                    continue;
                }

                if seq.callback.send(out_token).is_err() {
                    remove_slot!();
                    continue;
                }

                if seq.token_pos + 1 >= seq.maximum_tokens {
                    remove_slot!();
                    continue;
                }

                self.batch.add(out_token, seq.token_pos as i32, &[i as i32], true)?;
                seq.logits_pos = Some(self.batch.n_tokens() - 1);
                seq.sampler.accept(out_token);

                seq.token_pos += 1;
            }
        }

        Ok(slot_size)
    }
}

const RAIDX_TRIE_KV_CACHE_MAX_SEQ: usize = 64;

fn completions_handler(
    model: &LlamaModel,
    backend: &LlamaBackend,
    task_rx: &flume::Receiver<CompletionsTask>,
    n_tasks: u32,
    kv_cache_size_pre_task: u32,
    offload_kqv: bool,
    type_k: Option<KVCacheTypes>,
    type_v: Option<KVCacheTypes>,
    is_cancel: &AtomicBool,
) -> Result<()> {
    let model_metadata = ModelMetadata::from(model);

    let mut ctx_params = LlamaContextParams::default()
        .with_flash_attention(false)
        .with_offload_kqv(offload_kqv)
        .with_n_ctx(NonZeroU32::new(n_tasks * kv_cache_size_pre_task))
        .with_n_batch(n_tasks * kv_cache_size_pre_task);

    ctx_params.context_params.n_seq_max = n_tasks;

    if let Some(type_k) = type_k {
        ctx_params.context_params.type_k = type_k as ggml_type;
    };

    if let Some(type_v) = type_v {
        ctx_params.context_params.type_v = type_v as ggml_type;
    };

    model_metadata.estimate_session_size(&ctx_params);

    let mut ctx = model.new_context(backend, ctx_params)?;
    let mut batch = LlamaBatch::new(kv_cache_size_pre_task as usize * n_tasks as usize, 1);

    let mut sequence_slots = SequenceSlots::new(n_tasks, &mut batch, model);
    let mut trie_cache = RadixTrieKVCache::new(RAIDX_TRIE_KV_CACHE_MAX_SEQ);

    loop {
        if sequence_slots.len() == 0 {
            let task = match task_rx.recv_timeout(Duration::from_secs(1)) {
                Ok(task) => task,
                Err(RecvTimeoutError::Timeout) => {
                    if is_cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        return Ok(());
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(anyhow!("Task channel disconnected"));
                }
            };

            let sequence = Sequence {
                sampler: Sampler::new(
                    model,
                    task.frequency_penalty,
                    task.presence_penalty,
                    task.seed,
                    task.temperature,
                    task.top_p
                ),
                callback: task.to_api,
                token_pos: task.input_token_list.len() as u32,
                maximum_tokens: min(
                    task.maximum_tokens.map(|n_tokens| n_tokens + task.input_token_list.len() as u32).unwrap_or(kv_cache_size_pre_task),
                    kv_cache_size_pre_task
                ),
                input_tokens: task.input_token_list,
                logits_pos: None,
                state: SeqState::Prefill
            };

            sequence_slots.put(sequence, &mut ctx, &mut trie_cache)?;
        }

        while sequence_slots.len() < n_tasks as usize {
            match task_rx.try_recv() {
                Ok(task) => {
                    let sequence = Sequence {
                        sampler: Sampler::new(
                            model,
                            task.frequency_penalty,
                            task.presence_penalty,
                            task.seed,
                            task.temperature,
                            task.top_p
                        ),
                        callback: task.to_api,
                        token_pos: task.input_token_list.len() as u32,
                        maximum_tokens: min(
                            task.maximum_tokens.map(|n_tokens| n_tokens + task.input_token_list.len() as u32).unwrap_or(kv_cache_size_pre_task),
                            kv_cache_size_pre_task
                        ),
                        input_tokens: task.input_token_list,
                        logits_pos: None,
                        state: SeqState::Prefill
                    };

                    sequence_slots.put(sequence, &mut ctx, &mut trie_cache)?;
                }
                Err(flume::TryRecvError::Empty) => break,
                Err(flume::TryRecvError::Disconnected) => {
                    return Err(anyhow!("Task channel disconnected"));
                }
            }
        }

        sequence_slots.batch_decode(&mut ctx, &mut trie_cache)?;
        sequence_slots.batch_sample(&mut ctx)?;
    }
}

enum SpeculativeCompletionsTargetInput {
    PromptInput {
        token_list: Vec<LlamaToken>
    },
    DraftInput {
        draft_token_list: Vec<LlamaToken>
    },
}

struct SpeculativeCompletionsTargetOutput {
    accept_token_n: u32,
    next_token: Option<LlamaToken>
}

struct SpeculativeCompletionsTargetTask {
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: i64,
    temperature: Option<f32>,
    top_p: Option<f32>,
    input_channel: flume::Receiver<SpeculativeCompletionsTargetInput>,
    output_channel: flume::Sender<SpeculativeCompletionsTargetOutput>
}

struct SpeculativeCompletionsTargetSequence {
    prompt_token_list: Vec<LlamaToken>,
    accepted_token_list: Vec<LlamaToken>,
    sampler: Sampler,
    input_channel: flume::Receiver<SpeculativeCompletionsTargetInput>,
    output_channel: flume::Sender<SpeculativeCompletionsTargetOutput>,
}

struct SpeculativeCompletionsTargetSequenceSlots<'a> {
    sequence_list: Vec<Option<SpeculativeCompletionsTargetSequence>>,
    batch: &'a mut LlamaBatch,
    model: &'a LlamaModel,
    n_candidates: usize
}

impl <'a> SpeculativeCompletionsTargetSequenceSlots<'a> {
    fn new(n_task: u32, batch: &'a mut LlamaBatch, model: &'a LlamaModel, n_candidates: usize) -> Self {
        let mut sequence_list = Vec::with_capacity(n_task as usize);

        for _ in 0..n_task {
            sequence_list.push(None);
        }

        Self {
            sequence_list,
            batch,
            model,
            n_candidates
        }
    }

    fn len(&self) -> usize {
        self.sequence_list.iter()
            .filter(|v| v.is_some())
            .count()
    }

    fn put(&mut self, seq: SpeculativeCompletionsTargetSequence) -> Result<()> {
        for slot in self.sequence_list.iter_mut() {
            if slot.is_some() {
                continue;
            }

            *slot = Some(seq);
            return Ok(());
        }
        Err(anyhow!("No available slot"))
    }

    fn clear_closed(&mut self, ctx: &mut LlamaContext, select_tmp: &mut Option<(u32, SpeculativeCompletionsTargetInput)>) -> Result<usize> {
        let mut out = 0;

        for (seq_id, seq_opt) in self.sequence_list.iter_mut().enumerate() {
            if let Some(seq) = seq_opt {
                if seq.input_channel.is_disconnected() {
                    if let Some((id, _)) = select_tmp.as_ref() {
                        if *id == seq_id as u32 {
                            *select_tmp = None;
                        }
                    }
                    *seq_opt = None;
                    ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None)?;
                    out += 1;
                }
            }
        }
        Ok(out)
    }

    // (seq_id, task_input)
    fn poll(&mut self, ctx: &mut LlamaContext, mut select_task: Option<(u32, SpeculativeCompletionsTargetInput)>, cache: &mut RadixTrieKVCache) -> Result<u32> {
        // (logits_idx, pos, seq_id)
        let mut sample_list: Vec<(i32, u32, u32)> = Vec::new();
        // seq_id -> draft_token_list
        let mut draft_mapping: BTreeMap<u32, Vec<LlamaToken>> = BTreeMap::new();
        let mut prefill_seq_ids = Vec::new();
        let mut decode_n: u32 = 0;

        for (id, seq) in self.sequence_list.iter_mut().enumerate() {
            if let Some(seq) = seq {
                let select_task_id_eq = select_task.as_ref().map(|(seq_id, _)| id as u32 == *seq_id).unwrap_or(false);

                let input_task = if select_task_id_eq {
                    select_task.take().map(|(_, task)| task)
                } else {
                    seq.input_channel.try_recv().ok()
                };

                match input_task {
                    Some(task) => {
                        match task {
                            SpeculativeCompletionsTargetInput::PromptInput { token_list } => {
                                seq.prompt_token_list = token_list;
                                let token_list = &seq.prompt_token_list;
                                prefill_seq_ids.push(id);
                                let raw_tokens = token_list.iter().map(|t| t.0).collect::<Vec<_>>();

                                match cache.get(&raw_tokens) {
                                    None => {
                                        for i in 0..token_list.len() - 1 {
                                            self.batch.add(token_list[i], i as i32, &[id as i32], false)?
                                        }
                                    }
                                    Some((sub_seq_data, sub_seq_tokens_len)) => {
                                        debug!("cache hit");
                                        let sub_seq_len = min(token_list.len() - 1, sub_seq_tokens_len);

                                        unsafe {
                                            let res = llama_cpp_sys_2::llama_state_seq_set_data(ctx.context.as_ptr(), sub_seq_data.as_ptr(), sub_seq_data.len(), id as i32);
                                            ensure!(res != 0);
                                            ctx.clear_kv_cache_seq(Some(id as u32), Some(sub_seq_len as u32), None)?;
                                        }

                                        if sub_seq_len == token_list.len() - 1 {
                                            continue;
                                        }

                                        for i in sub_seq_len..token_list.len() - 1 {
                                            self.batch.add(token_list[i], i as i32, &[id as i32], false)?
                                        }
                                    }
                                }
                            }
                            SpeculativeCompletionsTargetInput::DraftInput { draft_token_list } => {
                                if seq.accepted_token_list.len() == 0 {
                                    self.batch.add(seq.prompt_token_list[seq.prompt_token_list.len() - 1], seq.prompt_token_list.len() as i32 - 1, &[id as i32], true)?;
                                    sample_list.push((self.batch.n_tokens() - 1, seq.prompt_token_list.len() as u32 - 1, id as u32));
                                    seq.accepted_token_list.extend_from_slice(&seq.prompt_token_list);

                                    let mut idx = 0;
                                    for pos in seq.accepted_token_list.len()..seq.accepted_token_list.len() + draft_token_list.len() - 1 {
                                        self.batch.add(draft_token_list[idx], pos as i32, &[id as i32], true)?;
                                        sample_list.push((self.batch.n_tokens() - 1, pos as u32, id as u32));
                                        idx += 1;
                                    }
                                } else {
                                    let mut tokens = Vec::with_capacity(draft_token_list.len());
                                    tokens.push(seq.accepted_token_list.last().unwrap().clone());
                                    tokens.extend_from_slice(&draft_token_list[..draft_token_list.len() - 1]);

                                    ctx.clear_kv_cache_seq(Some(id as u32), Some(seq.accepted_token_list.len() as u32 - 1), None)?;

                                    let mut idx = 0;
                                    for pos in seq.accepted_token_list.len() - 1..seq.accepted_token_list.len() - 1 + draft_token_list.len() {
                                        self.batch.add(tokens[idx], pos as i32, &[id as i32], true)?;
                                        sample_list.push((self.batch.n_tokens() - 1, pos as u32, id as u32));
                                        idx += 1;
                                    }
                                }

                                draft_mapping.insert(id as u32, draft_token_list);
                            }
                        }
                        decode_n += 1;
                    }
                    None => continue,
                }
            }
        }

        if decode_n == 0 {
            return Ok(0);
        }

        ctx.decode(self.batch)?;
        self.batch.clear();

        for seq_id in prefill_seq_ids {
            unsafe {
                let seq = self.sequence_list[seq_id].as_ref().unwrap();
                let data_size = llama_cpp_sys_2::llama_state_seq_get_size(ctx.context.as_ptr(), seq_id as i32);
                let mut data = vec![0u8; data_size];
                llama_cpp_sys_2::llama_state_seq_get_data(ctx.context.as_ptr(), data.as_mut_ptr(), data_size, seq_id as i32);

                let raw_input_tokens = seq.prompt_token_list[0..seq.prompt_token_list.len() - 1].iter().map(|t| t.0).collect::<Vec<_>>();
                cache.insert(raw_input_tokens, data)?;
            }
        }

        // seq_id -> tokens
        let mut out_mapping: BTreeMap<u32, Vec<LlamaToken>> = BTreeMap::new();
        // seq_id -> next_token
        let mut next_mapping: BTreeMap<u32, LlamaToken> = BTreeMap::new();
        for (i, pos, seq_id) in sample_list {
            if out_mapping.get(&seq_id).is_none() {
                out_mapping.insert(seq_id, Vec::new());
            }

            if next_mapping.get(&seq_id).is_some() {
                debug!("next mapping continue");
                continue;
            }

            let seq = self.sequence_list[seq_id as usize].as_mut().unwrap();
            debug!("target sample");
            let mut token = seq.sampler.sample(ctx, i);

            let draft_tokens = draft_mapping.get(&seq_id).unwrap();
            let draft_idx = pos as usize + 1 - seq.accepted_token_list.len();

            let mut set_next = true;

            if token != draft_tokens[draft_idx] {
                let candidates = seq.sampler.get_candidates();
                ensure!(candidates.sorted);

                let token_data_list = unsafe { &*slice_from_raw_parts(candidates.data, min(self.n_candidates, candidates.size)) };

                for td in token_data_list {
                    if td.id == draft_tokens[draft_idx].0 {
                        info!("candidate found");
                        token = draft_tokens[draft_idx];
                        set_next = false;
                        break;
                    }
                }
            } else {
                set_next = false;
            }

            let is_eog_token = self.model.is_eog_token(token);
            if !is_eog_token {
                seq.sampler.accept(token);
            }

            if !set_next {
                out_mapping.get_mut(&seq_id).unwrap().push(token);
            } else {
                next_mapping.insert(seq_id, token);
            }
        }

        for (seq_id, out_tokens) in out_mapping {
            let seq = self.sequence_list[seq_id as usize].as_mut().unwrap();
            seq.accepted_token_list.extend_from_slice(&out_tokens);
            let next = next_mapping.get(&seq_id).cloned();

            if let Some(next) = next {
                seq.accepted_token_list.push(next);
            }

            let out = SpeculativeCompletionsTargetOutput {
                accept_token_n: out_tokens.len() as u32,
                next_token: next
            };

            let _ = seq.output_channel.send(out);
        }
        Ok(decode_n)
    }
}

fn speculative_completions_target_handler(
    model: &LlamaModel,
    backend: &LlamaBackend,
    task_rx: &flume::Receiver<SpeculativeCompletionsTargetTask>,
    n_tasks: u32,
    kv_cache_size_pre_task: u32,
    n_candidates: usize,
    offload_kqv: bool,
    type_k: Option<KVCacheTypes>,
    type_v: Option<KVCacheTypes>,
    _is_cancel: &AtomicBool
) -> Result<()> {
    let mut ctx_params = LlamaContextParams::default()
        .with_flash_attention(false)
        .with_offload_kqv(offload_kqv)
        .with_n_ctx(NonZeroU32::new(n_tasks * kv_cache_size_pre_task))
        .with_n_batch(n_tasks * kv_cache_size_pre_task);

    ctx_params.context_params.n_seq_max = n_tasks;

    if let Some(type_k) = type_k {
        ctx_params.context_params.type_k = type_k as ggml_type;
    };

    if let Some(type_v) = type_v {
        ctx_params.context_params.type_v = type_v as ggml_type;
    };

    let mut ctx = model.new_context(backend, ctx_params)?;
    let mut batch = LlamaBatch::new(kv_cache_size_pre_task as usize * n_tasks as usize, 1);

    let mut slots = SpeculativeCompletionsTargetSequenceSlots::new(n_tasks, &mut batch, model, n_candidates);
    let mut trie_cache = RadixTrieKVCache::new(RAIDX_TRIE_KV_CACHE_MAX_SEQ);

    let select_tmp = RefCell::new(None);
    loop {
        slots.clear_closed(&mut ctx, &mut *select_tmp.borrow_mut())?;

        while slots.len() < n_tasks as usize {
            match task_rx.try_recv() {
                Ok(task) => {
                    let input_channel = task.input_channel;
                    let output_channel = task.output_channel;

                    let sequence = SpeculativeCompletionsTargetSequence {
                        sampler: Sampler::new(
                            model,
                            task.frequency_penalty,
                            task.presence_penalty,
                            Some(task.seed),
                            task.temperature,
                            task.top_p
                        ),
                        input_channel,
                        output_channel,
                        prompt_token_list: Vec::new(),
                        accepted_token_list: Vec::new(),
                    };

                    slots.put(sequence)?;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    return Err(anyhow!("Task channel disconnected"));
                }
            }
        }

        slots.poll(&mut ctx, select_tmp.take(), &mut trie_cache)?;

        let mut selector = flume::Selector::new();

        for (seq_id, seq_op) in slots.sequence_list.iter().enumerate() {
            if let Some(seq) = seq_op {
                selector = selector.recv(&seq.input_channel, {
                    let select_tmp = &select_tmp;
                    move |res| {
                        *select_tmp.borrow_mut() = res.ok().map(|input|(seq_id as u32, input));
                    }
                })
            }
        }

        let task = Rc::new(RefCell::new(None));

        if slots.len() < n_tasks as usize {
            selector = selector.recv(&task_rx, {
                let task = task.clone();
                move |task_res| *task.borrow_mut() = Some(task_res)
            });
        }
        selector.wait();

        if let Some(task) = task.take() {
            let task = task?;
            let input_channel = task.input_channel;
            let output_channel = task.output_channel;

            let sequence = SpeculativeCompletionsTargetSequence {
                sampler: Sampler::new(
                    model,
                    task.frequency_penalty,
                    task.presence_penalty,
                    Some(task.seed),
                    task.temperature,
                    task.top_p
                ),
                input_channel,
                output_channel,
                prompt_token_list: Vec::new(),
                accepted_token_list: Vec::new(),
            };

            slots.put(sequence)?;
        };
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum DraftSequenceState {
    Decode,
    WaitConfirm,
}

struct SpeculativeCompletionsDraftSequence {
    state: DraftSequenceState,
    prompt_tokens: Vec<LlamaToken>,
    confirmed_tokens: Vec<LlamaToken>,
    unconfirmed_tokens: Vec<LlamaToken>,
    sampler: Sampler,
    api_channel: flume::Sender<LlamaToken>,
    to_target_channel: flume::Sender<SpeculativeCompletionsTargetInput>,
    from_target_channel: flume::Receiver<SpeculativeCompletionsTargetOutput>,
    maximum_tokens: u32,
    max_unconfirmed_tokens: usize,
    total_draft_tokens: u32,
    total_accept_tokens: u32,
}

impl SpeculativeCompletionsDraftSequence {
    fn new(
        task: CompletionsTask,
        model: &LlamaModel,
        send_to_target: flume::Sender<SpeculativeCompletionsTargetInput>,
        from_target:  flume::Receiver<SpeculativeCompletionsTargetOutput>,
        max_unconfirmed_tokens: usize,
    ) -> Self {
        let sequence = SpeculativeCompletionsDraftSequence {
            state: DraftSequenceState::Decode,
            prompt_tokens: task.input_token_list,
            confirmed_tokens: Vec::new(),
            unconfirmed_tokens: Vec::new(),
            sampler: Sampler::new(
                model,
                task.frequency_penalty,
                task.presence_penalty,
                task.seed,
                task.temperature,
                task.top_p
            ),
            api_channel: task.to_api,
            to_target_channel: send_to_target,
            from_target_channel: from_target,
            maximum_tokens: task.maximum_tokens.unwrap(),
            max_unconfirmed_tokens,
            total_draft_tokens: 0,
            total_accept_tokens: 0
        };
        sequence
    }
}

struct SpeculativeCompletionsDraftSequenceSlots<'a> {
    sequence_list: Vec<Option<SpeculativeCompletionsDraftSequence>>,
    batch: &'a mut LlamaBatch,
    model: &'a LlamaModel,
}

impl <'a> SpeculativeCompletionsDraftSequenceSlots<'a> {
    fn new(
        n_task: u32,
        batch: &'a mut LlamaBatch,
        model: &'a LlamaModel,
    ) -> Self {
        let mut sequence_list = Vec::with_capacity(n_task as usize);

        for _ in 0..n_task {
            sequence_list.push(None);
        }

        Self {
            sequence_list,
            batch,
            model,
        }
    }

    fn len(&self) -> usize {
        self.sequence_list.iter()
            .filter(|v| v.is_some())
            .count()
    }

    fn put(&mut self, seq: SpeculativeCompletionsDraftSequence) -> Result<()> {
        for slot in self.sequence_list.iter_mut() {
            if slot.is_some() {
                continue;
            }

            *slot = Some(seq);
            return Ok(());
        }
        Err(anyhow!("No available slot"))
    }

    // (seq_id, task_input)
    fn poll(&mut self, ctx: &mut LlamaContext, mut select_task: Option<(u32, SpeculativeCompletionsTargetOutput)>, cache: &mut RadixTrieKVCache) -> Result<Poll<()>> {
        // seq_id -> logits_pos
        let mut decode_seq_list = BTreeMap::new();
        let mut need_loop = true;

        while need_loop {
            need_loop = false;
            for (seq_id, seq_op) in self.sequence_list.iter_mut().enumerate() {
                if let Some(seq) = seq_op {
                    match seq.state {
                        DraftSequenceState::Decode => {
                            if decode_seq_list.get(&seq_id).is_some() {
                                continue;
                            }

                            if seq.unconfirmed_tokens.is_empty() && seq.confirmed_tokens.is_empty() {
                                seq.confirmed_tokens.extend_from_slice(&seq.prompt_tokens);

                                let input = SpeculativeCompletionsTargetInput::PromptInput {
                                    token_list: seq.prompt_tokens.clone()
                                };

                                seq.to_target_channel.send(input)?;

                                let raw_tokens = seq.confirmed_tokens.iter().map(|t| t.0).collect::<Vec<_>>();
                                match cache.get(&raw_tokens) {
                                    None => {
                                        for i in 0..seq.confirmed_tokens.len() - 1 {
                                            self.batch.add(seq.confirmed_tokens[i], i as i32, &[seq_id as i32], false)?
                                        }
                                    }
                                    Some((sub_seq_data, sub_seq_tokens_len)) => {
                                        debug!("cache hit");
                                        let sub_seq_len = min(seq.confirmed_tokens.len() - 1, sub_seq_tokens_len);

                                        unsafe {
                                            let res = llama_cpp_sys_2::llama_state_seq_set_data(ctx.context.as_ptr(), sub_seq_data.as_ptr(), sub_seq_data.len(), seq_id as i32);
                                            ensure!(res != 0);
                                            ctx.clear_kv_cache_seq(Some(seq_id as u32), Some(sub_seq_len as u32), None)?;
                                        }

                                        for i in sub_seq_len..seq.confirmed_tokens.len() - 1 {
                                            self.batch.add(seq.confirmed_tokens[i], i as i32, &[seq_id as i32], false)?
                                        }
                                    }
                                }
                            }

                            if seq.unconfirmed_tokens.len() >= seq.max_unconfirmed_tokens ||
                                seq.confirmed_tokens.len() + seq.unconfirmed_tokens.len() >= seq.maximum_tokens as usize {
                                seq.state = DraftSequenceState::WaitConfirm;
                                let target_input = SpeculativeCompletionsTargetInput::DraftInput {
                                    draft_token_list: seq.unconfirmed_tokens.clone()
                                };
                                seq.to_target_channel.send(target_input)?;
                                continue;
                            }

                            let (enter_token, pos) = if seq.unconfirmed_tokens.is_empty() {
                                (*seq.confirmed_tokens.last().unwrap(), seq.confirmed_tokens.len() - 1)
                            } else {
                                (*seq.unconfirmed_tokens.last().unwrap(), seq.confirmed_tokens.len() + seq.unconfirmed_tokens.len() - 1)
                            };

                            self.batch.add(enter_token, pos as i32, &[seq_id as i32], true)?;
                            decode_seq_list.insert(seq_id, self.batch.n_tokens() - 1);
                            need_loop = true;
                        }
                        DraftSequenceState::WaitConfirm => {
                            let select_task_id_eq = select_task.as_ref().map(|(id, _)| seq_id as u32 == *id).unwrap_or(false);

                            let out = if select_task_id_eq {
                                select_task.take().map(|(_, task)| task).unwrap()
                            } else {
                                let out = match seq.from_target_channel.try_recv() {
                                    Ok(out) => out,
                                    Err(_) => continue,
                                };
                                out
                            };

                            info!("accept_token_n: {}", out.accept_token_n);
                            seq.total_draft_tokens += seq.unconfirmed_tokens.len() as u32;
                            seq.total_accept_tokens += out.accept_token_n;

                            let old_pos = seq.confirmed_tokens.len() - 1;
                            let update_to_confirm = &seq.unconfirmed_tokens[..out.accept_token_n as usize];

                            seq.confirmed_tokens.extend_from_slice(update_to_confirm);
                            if let Some(next_token) = out.next_token {
                                seq.confirmed_tokens.push(next_token);
                            }

                            let accept_rate = out.accept_token_n as f32 / seq.unconfirmed_tokens.len() as f32;
                            if accept_rate >= 0.8 {
                                seq.max_unconfirmed_tokens += 2;
                            }

                            if accept_rate < 0.4 {
                                seq.max_unconfirmed_tokens = max(2, seq.max_unconfirmed_tokens - 2);
                            }

                            if out.accept_token_n as usize != seq.unconfirmed_tokens.len() {
                                seq.sampler.reset();
                                for token in seq.confirmed_tokens.iter() {
                                    seq.sampler.accept(*token);
                                }

                                ctx.clear_kv_cache_seq(Some(seq_id as u32), Some(seq.confirmed_tokens.len() as u32 - 1), None)?;
                            }

                            let mut remove_seq = false;
                            for pos in old_pos + 1..seq.confirmed_tokens.len() {
                                let out_token = seq.confirmed_tokens[pos];

                                if self.model.is_eog_token(out_token) {
                                    info!("acceptance rate: {}", seq.total_accept_tokens as f32 / seq.total_draft_tokens as f32);
                                    remove_seq = true;
                                    break;
                                }

                                if seq.api_channel.send(out_token).is_err() {
                                    info!("acceptance rate: {}", seq.total_accept_tokens as f32 / seq.total_draft_tokens as f32);
                                    remove_seq = true;
                                    break;
                                }

                                if pos + 1 >= seq.maximum_tokens as usize {
                                    info!("acceptance rate: {}", seq.total_accept_tokens as f32 / seq.total_draft_tokens as f32);
                                    remove_seq = true;
                                    break;
                                }
                            }

                            if remove_seq {
                                *seq_op = None;
                                ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None)?;
                            } else {
                                seq.state = DraftSequenceState::Decode;
                                seq.unconfirmed_tokens.clear();
                                need_loop = true;
                            }
                        }
                    }
                }
            }
        }

        if !decode_seq_list.is_empty() {
            ctx.decode(self.batch)?;
            self.batch.clear();
            
            for (seq_id, logits_pos) in decode_seq_list {
                let seq = self.sequence_list[seq_id].as_mut().unwrap();

                // first prefill
                if seq.confirmed_tokens.len() == seq.prompt_tokens.len() && seq.unconfirmed_tokens.is_empty() {
                    unsafe {
                        let data_size = llama_cpp_sys_2::llama_state_seq_get_size(ctx.context.as_ptr(), seq_id as i32);
                        let mut data = vec![0u8; data_size];
                        llama_cpp_sys_2::llama_state_seq_get_data(ctx.context.as_ptr(), data.as_mut_ptr(), data_size, seq_id as i32);

                        let raw_input_tokens = seq.confirmed_tokens.iter().map(|t| t.0).collect::<Vec<_>>();
                        cache.insert(raw_input_tokens, data)?;
                    }
                }

                debug!("draft sample");
                let draft_token = seq.sampler.sample(ctx, logits_pos);
                seq.sampler.accept(draft_token);
                seq.unconfirmed_tokens.push(draft_token);
            }
            Ok(Poll::Ready(()))
        } else {
            Ok(Poll::Pending)
        }
    }
}

fn speculative_completions_draft_handler(
    model: &LlamaModel,
    backend: &LlamaBackend,
    to_target_handler: &flume::Sender<SpeculativeCompletionsTargetTask>,
    task_rx: &flume::Receiver<CompletionsTask>,
    n_tasks: u32,
    kv_cache_size_pre_task: u32,
    max_unconfirmed_tokens: usize,
    offload_kqv: bool,
    type_k: Option<KVCacheTypes>,
    type_v: Option<KVCacheTypes>,
) -> Result<()> {
    let mut ctx_params = LlamaContextParams::default()
        .with_flash_attention(false)
        .with_offload_kqv(offload_kqv)
        .with_n_ctx(NonZeroU32::new(n_tasks * kv_cache_size_pre_task))
        .with_n_batch(n_tasks * kv_cache_size_pre_task);

    ctx_params.context_params.n_seq_max = n_tasks;

    if let Some(type_k) = type_k {
        ctx_params.context_params.type_k = type_k as ggml_type;
    };

    if let Some(type_v) = type_v {
        ctx_params.context_params.type_v = type_v as ggml_type;
    };

    let mut ctx = model.new_context(backend, ctx_params)?;
    let mut batch = LlamaBatch::new(kv_cache_size_pre_task as usize * n_tasks as usize, 1);

    let mut slots = SpeculativeCompletionsDraftSequenceSlots::new(n_tasks, &mut batch, model);
    let mut trie_cache = RadixTrieKVCache::new(RAIDX_TRIE_KV_CACHE_MAX_SEQ);

    let select_task = RefCell::new(None);

    loop {
        while slots.len() < n_tasks as usize {
            match task_rx.try_recv() {
                Ok(mut task) => {
                    let (to_target, from_draft) = flume::unbounded();
                    let (to_draft, from_target) = flume::unbounded();
                    task.seed = Some(task.seed.unwrap_or_else(|| rand::random()));
                    task.maximum_tokens = {
                        let out = min(
                            task.maximum_tokens.map(|n_tokens| n_tokens + task.input_token_list.len() as u32).unwrap_or(kv_cache_size_pre_task),
                            kv_cache_size_pre_task
                        );
                        Some(out)
                    };

                    let target_task = SpeculativeCompletionsTargetTask {
                        frequency_penalty: task.frequency_penalty,
                        presence_penalty: task.presence_penalty,
                        seed: task.seed.unwrap(),
                        temperature: task.temperature,
                        top_p: task.top_p,
                        input_channel: from_draft,
                        output_channel: to_draft
                    };

                    to_target_handler.send(target_task)?;

                    let draft_seq = SpeculativeCompletionsDraftSequence::new(
                        task,
                        model,
                        to_target,
                        from_target,
                        max_unconfirmed_tokens
                    );

                    slots.put(draft_seq)?;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    return Err(anyhow!("Task channel disconnected"));
                }
            }
        }

        if slots.poll(&mut ctx, select_task.take(), &mut trie_cache)? == Poll::Pending {
            let mut selector = flume::Selector::new();

            for (seq_id, seq_op) in slots.sequence_list.iter().enumerate() {
                if let Some(seq) = seq_op {
                    selector = selector.recv(&seq.from_target_channel, {
                        let select_task = &select_task;
                        move |res| {
                            *select_task.borrow_mut() = res.ok().map(|target_out| (seq_id as u32, target_out));
                        }
                    })
                }
            }

            let completions_task = Rc::new(RefCell::new(None));

            if slots.len() < n_tasks as usize {
                selector = selector.recv(&task_rx, {
                    let completions_task = completions_task.clone();
                    move |task_res| *completions_task.borrow_mut() = task_res.ok()
                });
            }
            selector.wait();

            if let Some(mut completions_task) = completions_task.take() {
                let (to_target, from_draft) = flume::unbounded();
                let (to_draft, from_target) = flume::unbounded();
                completions_task.seed = Some(completions_task.seed.unwrap_or_else(|| rand::random()));
                completions_task.maximum_tokens = {
                    let out = min(
                        completions_task.maximum_tokens.map(|n_tokens| n_tokens + completions_task.input_token_list.len() as u32).unwrap_or(kv_cache_size_pre_task),
                        kv_cache_size_pre_task
                    );
                    Some(out)
                };

                let target_task = SpeculativeCompletionsTargetTask {
                    frequency_penalty: completions_task.frequency_penalty,
                    presence_penalty: completions_task.presence_penalty,
                    seed: completions_task.seed.unwrap(),
                    temperature: completions_task.temperature,
                    top_p: completions_task.top_p,
                    input_channel: from_draft,
                    output_channel: to_draft
                };

                to_target_handler.send(target_task)?;

                let draft_seq = SpeculativeCompletionsDraftSequence::new(
                    completions_task,
                    model,
                    to_target,
                    from_target,
                    max_unconfirmed_tokens
                );

                slots.put(draft_seq)?;
            }
        }
    }
}

struct SpeculativeCompletionsNgramSequence {
    confirmed_tokens: Vec<LlamaToken>,
    unconfirmed_tokens: Vec<LlamaToken>,
    api_channel: flume::Sender<LlamaToken>,
    to_target_channel: flume::Sender<SpeculativeCompletionsTargetInput>,
    from_target_channel: flume::Receiver<SpeculativeCompletionsTargetOutput>,
    maximum_tokens: u32,
    max_unconfirmed_tokens: usize,
    total_draft_tokens: u32,
    total_accept_tokens: u32,
}

impl SpeculativeCompletionsNgramSequence {
    fn new(
        task: CompletionsTask,
        send_to_target: flume::Sender<SpeculativeCompletionsTargetInput>,
        from_target:  flume::Receiver<SpeculativeCompletionsTargetOutput>,
        max_unconfirmed_tokens: usize,
    ) -> Self {
        SpeculativeCompletionsNgramSequence {
            confirmed_tokens: task.input_token_list,
            unconfirmed_tokens: Vec::new(),
            api_channel: task.to_api,
            to_target_channel: send_to_target,
            from_target_channel: from_target,
            maximum_tokens: task.maximum_tokens.unwrap(),
            max_unconfirmed_tokens,
            total_draft_tokens: 0,
            total_accept_tokens: 0
        }
    }
}

struct SpeculativeCompletionsNgramSlots<'a> {
    model: &'a LlamaModel,
    task_rx: &'a flume::Receiver<CompletionsTask>,
    sequence_list: Vec<SpeculativeCompletionsNgramSequence>,
    ngram_cache: &'a mut NgranCache,
    to_target_handler: &'a flume::Sender<SpeculativeCompletionsTargetTask>,
    kv_cache_size_pre_task: u32,
    max_unconfirmed_tokens: usize,
}

enum TaskEither {
    CompletionsTask(CompletionsTask),
    // seq_id, seq
    SpeculativeCompletionsTargetOutput((usize, SpeculativeCompletionsTargetOutput))
}

impl <'a> SpeculativeCompletionsNgramSlots<'a> {
    fn new(
        model: &'a LlamaModel,
        task_rx: &'a flume::Receiver<CompletionsTask>,
        ngram_cache: &'a mut NgranCache,
        to_target_handler: &'a flume::Sender<SpeculativeCompletionsTargetTask>,
        kv_cache_size_pre_task: u32,
        max_unconfirmed_tokens: usize,
    ) -> Self {
        Self {
            model,
            task_rx,
            sequence_list: Vec::new(),
            ngram_cache,
            to_target_handler,
            kv_cache_size_pre_task,
            max_unconfirmed_tokens
        }
    }

    fn poll(&mut self) -> Result<()> {
        let mut selector = flume::Selector::new();
        selector = selector.recv(&self.task_rx, |res| res.map(|t| TaskEither::CompletionsTask(t)));

        for (seq_id, seq) in self.sequence_list.iter().enumerate() {
            selector = selector.recv(&seq.from_target_channel, move |res| res.map(|t| TaskEither::SpeculativeCompletionsTargetOutput((seq_id, t))));
        }

        let task = selector.wait()?;

        match task {
            TaskEither::CompletionsTask(mut task) => {
                let (to_target, from_draft) = flume::unbounded();
                let (to_draft, from_target) = flume::unbounded();

                task.seed = Some(task.seed.unwrap_or_else(|| rand::random()));
                task.maximum_tokens = {
                    let out = min(
                        task.maximum_tokens.map(|n_tokens| n_tokens + task.input_token_list.len() as u32).unwrap_or(self.kv_cache_size_pre_task),
                        self.kv_cache_size_pre_task
                    );
                    Some(out)
                };

                let target_task = SpeculativeCompletionsTargetTask {
                    frequency_penalty: task.frequency_penalty,
                    presence_penalty: task.presence_penalty,
                    seed: task.seed.unwrap(),
                    temperature: task.temperature,
                    top_p: task.top_p,
                    input_channel: from_draft,
                    output_channel: to_draft
                };

                self.to_target_handler.send(target_task)?;
                to_target.send(SpeculativeCompletionsTargetInput::PromptInput { token_list: task.input_token_list.clone() })?;

                let mut seq = SpeculativeCompletionsNgramSequence::new(task, to_target, from_target, self.max_unconfirmed_tokens);

                let raw_confirmed_tokens = seq.confirmed_tokens.iter().map(|v| v.0).collect::<Vec<_>>();
                self.ngram_cache.update(NGRAM_MIN, NGRAM_MAX, &raw_confirmed_tokens, raw_confirmed_tokens.len() as i32);

                let drafts_tokens = min(seq.max_unconfirmed_tokens as i32, seq.maximum_tokens as i32 - raw_confirmed_tokens.len() as i32);
                let mut drafts = vec![0; drafts_tokens as usize + 1];
                drafts[0] = raw_confirmed_tokens[seq.confirmed_tokens.len() - 1];

                let draft_len = self.ngram_cache.draft(&raw_confirmed_tokens, &mut drafts, NGRAM_MIN, NGRAM_MAX);
                seq.unconfirmed_tokens = drafts[1..draft_len].iter().map(|&token| LlamaToken::new(token)).collect();

                if seq.unconfirmed_tokens.is_empty() {
                    seq.unconfirmed_tokens.push(LlamaToken::new(0));
                }

                seq.to_target_channel.send(SpeculativeCompletionsTargetInput::DraftInput {draft_token_list: seq.unconfirmed_tokens.clone()})?;
                self.sequence_list.push(seq);
            }
            TaskEither::SpeculativeCompletionsTargetOutput((seq_id, target_out)) => {
                let seq = &mut self.sequence_list[seq_id];

                info!("accept_token_n: {}", target_out.accept_token_n);
                seq.total_draft_tokens += seq.unconfirmed_tokens.len() as u32;
                seq.total_accept_tokens += target_out.accept_token_n;

                let old_confirmed_len = seq.confirmed_tokens.len();
                let new_confirmed_tokens = target_out.accept_token_n + if target_out.next_token.is_some() { 1 } else { 0 };
                let update_to_confirm = &seq.unconfirmed_tokens[..target_out.accept_token_n as usize];

                seq.confirmed_tokens.extend_from_slice(update_to_confirm);
                if let Some(next_token) = target_out.next_token {
                    seq.confirmed_tokens.push(next_token);
                }

                let accept_rate = target_out.accept_token_n as f32 / seq.unconfirmed_tokens.len() as f32;
                if accept_rate >= 0.8 {
                    seq.max_unconfirmed_tokens += 2;
                }

                if accept_rate < 0.4 {
                    seq.max_unconfirmed_tokens = max(2, seq.max_unconfirmed_tokens - 2);
                }

                let raw_confirmed_tokens = seq.confirmed_tokens.iter().map(|v| v.0).collect::<Vec<_>>();
                self.ngram_cache.update(NGRAM_MIN, NGRAM_MAX, &raw_confirmed_tokens, new_confirmed_tokens as i32 );

                for &token in &seq.confirmed_tokens[old_confirmed_len..old_confirmed_len + new_confirmed_tokens as usize] {
                    if self.model.is_eog_token(token) {
                        self.sequence_list.remove(seq_id);
                        return Ok(());
                    }

                    if seq.api_channel.send(token).is_err() {
                        self.sequence_list.remove(seq_id);
                        return Ok(());
                    }
                }

                if seq.confirmed_tokens.len() >= seq.maximum_tokens as usize {
                    self.sequence_list.remove(seq_id);
                    return Ok(());
                }

                let drafts_tokens = min(seq.max_unconfirmed_tokens as i32, seq.maximum_tokens as i32 - raw_confirmed_tokens.len() as i32);
                let mut drafts = vec![0; drafts_tokens as usize + 1];
                drafts[0] = raw_confirmed_tokens[seq.confirmed_tokens.len() - 1];

                let draft_len = self.ngram_cache.draft(&raw_confirmed_tokens, &mut drafts, NGRAM_MIN, NGRAM_MAX);
                seq.unconfirmed_tokens = drafts[1..draft_len].iter().map(|&token| LlamaToken::new(token)).collect();

                if seq.unconfirmed_tokens.is_empty() {
                    seq.unconfirmed_tokens.push(LlamaToken::new(0));
                }

                seq.to_target_channel.send(SpeculativeCompletionsTargetInput::DraftInput {draft_token_list: seq.unconfirmed_tokens.clone()})?;
            }
        };
        Ok(())
    }
}

fn speculative_completions_ngram_handler(
    model: &LlamaModel,
    to_target_handler: &flume::Sender<SpeculativeCompletionsTargetTask>,
    task_rx: &flume::Receiver<CompletionsTask>,
    kv_cache_size_pre_task: u32,
    max_unconfirmed_tokens: usize,
) -> Result<()> {
    let mut ngram_cache = NgranCache::new();

    let mut slots = SpeculativeCompletionsNgramSlots::new(
        model,
        task_rx,
        &mut ngram_cache,
        to_target_handler,
        kv_cache_size_pre_task,
        max_unconfirmed_tokens
    );

    loop {
        slots.poll()?;
    }
}

fn embedding_handler(
    model: &LlamaModel,
    backend: &LlamaBackend,
    task_rx: &flume::Receiver<EmbeddingTask>,
    n_tasks: u32,
    kv_cache_size_pre_task: u32,
    offload_kqv: bool,
    type_k: Option<KVCacheTypes>,
    type_v: Option<KVCacheTypes>,
    _is_cancel: &AtomicBool,
) -> Result<()> {
    let n_embd = model.n_embd() as usize;

    let mut ctx_params = LlamaContextParams::default()
        .with_embeddings(true)
        .with_flash_attention(false)
        .with_offload_kqv(offload_kqv)
        .with_n_ctx(NonZeroU32::new(n_tasks * kv_cache_size_pre_task))
        .with_n_batch(n_tasks * kv_cache_size_pre_task);

    ctx_params.context_params.n_seq_max = n_tasks;

    if let Some(type_k) = type_k {
        ctx_params.context_params.type_k = type_k as ggml_type;
    };

    if let Some(type_v) = type_v {
        ctx_params.context_params.type_v = type_v as ggml_type;
    };

    let mut ctx = model.new_context(backend, ctx_params)?;
    let mut batch = LlamaBatch::new(kv_cache_size_pre_task as usize * n_tasks as usize, 1);
    let pooling_type = unsafe {llama_cpp_sys_2::llama_pooling_type(ctx.context.as_ptr())};

    let mut task_list = Vec::with_capacity(n_tasks as usize);

    loop {
        if task_list.is_empty() {
            match task_rx.recv_timeout(Duration::from_secs(1)) {
                Ok(task) => task_list.push(task),
                Err(RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(anyhow!("Task channel disconnected"));
                }
            };
        }

        while task_list.len() < n_tasks as usize {
            match task_rx.try_recv() {
                Ok(task) => {
                    task_list.push(task);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    return Err(anyhow!("Task channel disconnected"));
                }
            }
        }

        for (seq_id, task) in task_list.iter().enumerate() {
            for (pos, token) in task.input_token_list.iter().enumerate() {
                batch.add(*token, pos as i32, &[seq_id as i32], pooling_type == LLAMA_POOLING_TYPE_NONE)?;
            }
        }

        ctx.clear_kv_cache();
        ctx.decode(&mut batch)?;
        let embeddings_len = task_list.len() * n_embd;
        batch.clear();

        if pooling_type == LLAMA_POOLING_TYPE_NONE {
            let mut out = unsafe {
                let out_ptr = llama_cpp_sys_2::llama_get_embeddings(ctx.context.as_ptr());
                ensure!(!out_ptr.is_null());
                slice::from_raw_parts(out_ptr, embeddings_len)
            };

            for task in task_list.drain(..){
                let (l, r) = out.split_at(n_embd);
                out = r;
                let _ = task.to_api.send(l.to_vec());
            }
        } else {
            for (seq_id, task) in task_list.drain(..).enumerate() {
                let out = ctx.embeddings_seq_ith(seq_id as i32)?.to_vec();
                let _ = task.to_api.send(out);
            }
        }
    }
}

pub async fn run_embedding (
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
    task_rx: flume::Receiver<EmbeddingTask>,
    kv_cache_size_pre_task: u32,
    n_tasks: u32,
    offload_kqv: bool,
    type_k: Option<KVCacheTypes>,
    type_v: Option<KVCacheTypes>,
) -> Result<()> {
    let is_cancel = Arc::new(AtomicBool::new(false));
    let model = model.clone();
    let backend = backend.clone();

    tokio::task::spawn_blocking(move || {
        embedding_handler(
            &*model,
            &*backend,
            &task_rx,
            n_tasks,
            kv_cache_size_pre_task,
            offload_kqv,
            type_k,
            type_v,
            &*is_cancel
        )
    }).await?
}

pub async fn run_completions(
    model: Arc<LlamaModel>,
    draft_model: Option<Arc<LlamaModel>>,
    backend: Arc<LlamaBackend>,
    task_rx: flume::Receiver<CompletionsTask>,
    kv_cache_size_pre_task: u32,
    n_tasks: u32,
    max_unconfirmed_tokens: usize,
    n_candidates: usize,
    offload_kqv: bool,
    type_k: Option<KVCacheTypes>,
    type_v: Option<KVCacheTypes>,
    draft_type_k: Option<KVCacheTypes>,
    draft_type_v: Option<KVCacheTypes>,
    lookup: bool
) -> Result<()> {
    let is_cancel = Arc::new(AtomicBool::new(false));

    match draft_model {
        None => {
            let model = model.clone();
            let backend = backend.clone();

            if lookup {
                let (to_target_handler, from_ngram_handler) = flume::unbounded();

                let target_handle = tokio::task::spawn_blocking({
                    let model = model.clone();
                    let backend = backend.clone();

                    move || {
                        speculative_completions_target_handler(
                            &*model,
                            &*backend,
                            &from_ngram_handler,
                            n_tasks,
                            kv_cache_size_pre_task,
                            n_candidates,
                            offload_kqv,
                            type_k,
                            type_v,
                            &*is_cancel,
                        )
                    }
                });

                let ngram_handle = tokio::task::spawn_blocking(move || {
                    speculative_completions_ngram_handler(
                        &model,
                        &to_target_handler,
                        &task_rx,
                        kv_cache_size_pre_task,
                        max_unconfirmed_tokens
                    )
                });

                let res = tokio::try_join!{
                    async {target_handle.await?},
                    async {ngram_handle.await?}
                };
                res?;
                Ok(())
            } else {
                tokio::task::spawn_blocking(move || {
                    completions_handler(
                        &*model,
                        &*backend,
                        &task_rx,
                        n_tasks,
                        kv_cache_size_pre_task,
                        offload_kqv,
                        type_k,
                        type_v,
                        &*is_cancel
                    )
                }).await?
            }
        }
        Some(draft_model) => {
            {
                let ctx_params = LlamaContextParams::default();
                let target_ctx = model.new_context(&*backend, ctx_params.clone())?;
                let draft_ctx = draft_model.new_context(&*backend, ctx_params)?;
                unsafe { ensure!(hibiki_common_speculative_are_compatible(target_ctx.context.as_ref(), draft_ctx.context.as_ref())) };
            }

            let (to_target_handler, from_draft_handler) = flume::unbounded();

            let target_handle = tokio::task::spawn_blocking({
                let model = model.clone();
                let backend = backend.clone();
                let is_cancel = Arc::new(AtomicBool::new(false));

                move || {
                    speculative_completions_target_handler(
                        &*model,
                        &*backend,
                        &from_draft_handler,
                        n_tasks,
                        kv_cache_size_pre_task,
                        n_candidates,
                        offload_kqv,
                        type_k,
                        type_v,
                        &*is_cancel,
                    )
                }
            });

            let draft_handle = tokio::task::spawn_blocking(move || {
                speculative_completions_draft_handler(
                    &*draft_model,
                    &*backend,
                    &to_target_handler,
                    &task_rx,
                    n_tasks,
                    kv_cache_size_pre_task,
                    max_unconfirmed_tokens,
                    offload_kqv,
                    draft_type_k,
                    draft_type_v,
                )
            });

            let res = tokio::try_join!{
                async {target_handle.await?},
                async {draft_handle.await?}
            };
            res?;
            Ok(())
        }
    }
}