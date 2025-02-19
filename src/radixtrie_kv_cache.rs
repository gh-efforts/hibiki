use std::sync::atomic::{AtomicI64, Ordering};
use anyhow::{ ensure, Result};
use chrono::Utc;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_sys_2::{llama_kv_cache_seq_cp, llama_token};
use radix_trie::{Trie, TrieCommon};

pub struct RadixTrieKVCache<'a, 'b: 'a> {
    // seq -> seq_id
    trie: Trie<Vec<llama_token>, i32>,
    seq_ids: Vec<Option<(Vec<llama_token>, AtomicI64)>>,
    ctx: &'a mut LlamaContext<'b>
}

impl <'a, 'b: 'a> RadixTrieKVCache<'a, 'b> {
    pub fn new(ctx: &'a mut LlamaContext<'b>, seq_len: usize) -> RadixTrieKVCache<'a, 'b> {
        RadixTrieKVCache {
            trie: Trie::new(),
            seq_ids: {
                let mut seq_ids = Vec::with_capacity(seq_len - 1);
                // last sequence is used for copy
                for _ in 0..seq_len - 1 {
                    seq_ids.push(None);
                }
                seq_ids
            },
            ctx,
        }
    }

    pub fn get(&mut self, seq: &[llama_token]) -> Option<(Vec<u8>, usize)> {
        let get_descendant = |seq: &[llama_token]|  {
            let seq_id = self.trie.get_raw_descendant(&seq.to_vec())?
                .iter()
                .min_by_key(|(seq, _)| seq.len())
                .map(|(_, seq_id)| *seq_id)?;

            Some(seq_id)
        };

        let mut last = None;

        for i in 1..seq.len() {
            let res = get_descendant(&seq[0..i]);

            if res.is_none() {
                break;
            }

            last = res.map(|seq_id| (seq_id, i));
        }

        let (seq_id, sub_pos) = last?;
        let (_sub_seq, access_time) = self.seq_ids[seq_id as usize].as_ref()?;
        access_time.store(Utc::now().timestamp(), Ordering::Relaxed);

        unsafe {
            self.ctx.clear_kv_cache_seq(Some(self.seq_ids.len() as u32 - 1), None, None).unwrap();
            self.ctx.copy_kv_cache_seq(seq_id, self.seq_ids.len() as i32 - 1, None, Some(sub_pos as u32)).unwrap();
            self.ctx.kv_cache_update();
            let state_size = llama_cpp_sys_2::llama_state_seq_get_size(self.ctx.context.as_ptr(), self.seq_ids.len() as i32 - 1);
            let mut state_data = vec![0u8; state_size];

            llama_cpp_sys_2::llama_state_seq_get_data(
                self.ctx.context.as_ptr(),
                state_data.as_mut_ptr(),
                state_size,
                self.seq_ids.len() as i32 - 1
            );

            Some((state_data, sub_pos))
        }
    }

    pub fn insert(&mut self, tokens: Vec<llama_token>, seq_data: &[u8]) -> Result<i32> {
        unsafe {
            let mut seq_id = None;

            for (id, x) in &mut self.seq_ids.iter_mut().enumerate() {
                if x.is_none() {
                    seq_id = Some(id as i32);
                    break;
                }
            }

            let seq_id = match seq_id {
                None => {
                    let item = self.seq_ids.iter_mut()
                        .min_by_key(|v| {
                            v.as_ref().unwrap().1.load(Ordering::Relaxed)
                        }).unwrap();

                    let (seq, _) = item.take().unwrap();
                    let seq_id = self.trie.remove(&seq).unwrap();
                    ensure!(self.ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None)?);

                    seq_id
                },
                Some(v) => v
            };


            let res = llama_cpp_sys_2::llama_state_seq_set_data(self.ctx.context.as_ptr(), seq_data.as_ptr(), seq_data.len(), seq_id);
            ensure!(res != 0);

            // (sub_seq_id, sub_pos_end)
            let mut sub_seq: Option<(i32, i32)> = None;

            if let Some(sub_tree) = self.trie.get_ancestor(&tokens) {
                let sub_pos_end = sub_tree.key().unwrap().len() as i32;
                let sub_seq_id = *sub_tree.value().unwrap();
                sub_seq = Some((sub_seq_id, sub_pos_end));
            }

            if let Some((sub_seq_id, sub_pos_end)) = sub_seq {
                self.ctx.clear_kv_cache_seq(Some(seq_id as u32), None, Some(sub_pos_end as u32))?;
                llama_kv_cache_seq_cp(self.ctx.context.as_ptr(), sub_seq_id, seq_id, -1, sub_pos_end);
            }

            self.ctx.kv_cache_update();

            self.trie.insert(tokens.clone(), seq_id);
            self.seq_ids[seq_id as usize] = Some((tokens, AtomicI64::new(Utc::now().timestamp())));
            Ok(seq_id)
        }
    }
}