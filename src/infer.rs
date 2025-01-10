use std::cmp::min;
use crate::CompletionsTask;
use anyhow::{anyhow, Result};
use flume::RecvTimeoutError;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

struct Sequence {
    input_tokens: Vec<LlamaToken>,
    sampler: LlamaSampler,
    callback: flume::Sender<LlamaToken>,
    token_pos: u32,
    maximum_tokens: u32,
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

    fn put(&mut self, seq: Sequence) -> Result<()> {
        for (i, slot) in self.sequence_list.iter_mut().enumerate() {
            if slot.is_some() {
                continue;
            }

            self.batch.add_sequence(&seq.input_tokens, i as i32, false)?;
            *slot = Some(seq);
            return Ok(());
        }
        Err(anyhow!("No available slot"))
    }

    fn batch_decode(&mut self, ctx: &mut LlamaContext) -> Result<usize> {
        let slot_size = self.len();

        if slot_size == 0 {
            return Err(anyhow!("No sequence to decode"));
        }

        ctx.decode(self.batch)?;
        self.batch.clear();
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
                let out_token = seq.sampler.sample(ctx, -1);

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

                let _res = seq.callback.send(out_token);

                if seq.token_pos + 1 >= seq.maximum_tokens {
                    remove_slot!();
                    continue;
                }

                self.batch.add(out_token, seq.token_pos as i32, &[i as i32], true)?;
                seq.sampler.accept(out_token);

                seq.token_pos += 1;
            }
        }

        Ok(slot_size)
    }
}

fn completions_handler(
    model: &LlamaModel,
    backend: &LlamaBackend,
    task_rx: &flume::Receiver<CompletionsTask>,
    n_tasks: u32,
    kv_cache_size_pre_task: u32,
    is_cancel: &AtomicBool
) -> Result<()> {
    let ctx_params = LlamaContextParams::default()
        .with_offload_kqv(true)
        .with_n_ctx(NonZeroU32::new(n_tasks * kv_cache_size_pre_task))
        .with_n_batch(n_tasks * kv_cache_size_pre_task);

    let mut ctx = model.new_context(backend, ctx_params)?;
    let mut batch = LlamaBatch::new((n_tasks * kv_cache_size_pre_task) as usize, 1);

    let mut sequence_slots = SequenceSlots::new(n_tasks, &mut batch, model);

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
                sampler: task.sampler,
                callback: task.callback,
                token_pos: task.input_token_list.len() as u32,
                maximum_tokens: min(
                    task.maximum_tokens.map(|n_tokens| n_tokens + task.input_token_list.len() as u32).unwrap_or(kv_cache_size_pre_task),
                    kv_cache_size_pre_task
                ),
                input_tokens: task.input_token_list,
            };

            sequence_slots.put(sequence)?;
        }

        while sequence_slots.len() < n_tasks as usize {
            match task_rx.try_recv() {
                Ok(task) => {
                    let sequence = Sequence {
                        sampler: task.sampler,
                        callback: task.callback,
                        token_pos: task.input_token_list.len() as u32,
                        maximum_tokens: min(
                            task.maximum_tokens.map(|n_tokens| n_tokens + task.input_token_list.len() as u32).unwrap_or(kv_cache_size_pre_task),
                            kv_cache_size_pre_task
                        ),
                        input_tokens: task.input_token_list,
                    };

                    sequence_slots.put(sequence)?;
                }
                Err(flume::TryRecvError::Empty) => break,
                Err(flume::TryRecvError::Disconnected) => {
                    return Err(anyhow!("Task channel disconnected"));
                }
            }
        }

        sequence_slots.batch_decode(&mut ctx)?;
        sequence_slots.batch_sample(&mut ctx)?;
    }
}

pub async fn run(
    model: Arc<LlamaModel>,
    backend: Arc<LlamaBackend>,
    task_rx: flume::Receiver<CompletionsTask>,
    kv_cache_size_pre_task: u32,
    n_tasks: u32
) -> Result<()> {
    let is_cancel = Arc::new(AtomicBool::new(false));

    tokio::task::spawn_blocking(move || {
        completions_handler(
            &*model,
            &*backend,
            &task_rx,
            n_tasks,
            kv_cache_size_pre_task,
            &*is_cancel
        )
    }).await?
}