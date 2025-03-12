use anyhow::Result;
use chrono::Utc;
use llama_cpp_sys_2::llama_token;
use radix_trie::{Trie, TrieCommon};
use std::sync::atomic::{AtomicI64, Ordering};

pub struct RadixTrieKVCache {
    // seq -> seq_id
    trie: Trie<Vec<llama_token>, i32>,
    // seq_id -> (seq, access_time, seq_data)
    seq_ids: Vec<Option<(Vec<llama_token>, AtomicI64, Vec<u8>)>>,
}

impl RadixTrieKVCache {
    pub fn new(seq_len: usize) -> RadixTrieKVCache {
        RadixTrieKVCache {
            trie: Trie::new(),
            seq_ids: {
                let mut seq_ids = Vec::with_capacity(seq_len);
                for _ in 0..seq_len {
                    seq_ids.push(None);
                }
                seq_ids
            },
        }
    }

    pub fn get(&self, seq: &[llama_token]) -> Option<(&[u8], usize)> {
        let get_descendant = |seq: &[llama_token]| {
            let seq_id = self
                .trie
                .get_raw_descendant(&seq.to_vec())?
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
        let (_sub_seq, access_time, seq_data) = self.seq_ids[seq_id as usize].as_ref()?;
        access_time.store(Utc::now().timestamp(), Ordering::Relaxed);
        Some((seq_data.as_slice(), sub_pos))
    }

    pub fn insert(&mut self, tokens: Vec<llama_token>, seq_data: Vec<u8>) -> Result<i32> {
        let mut seq_id = None;

        for (id, x) in &mut self.seq_ids.iter().enumerate() {
            if x.is_none() {
                seq_id = Some(id as i32);
                break;
            }
        }

        let seq_id = match seq_id {
            None => {
                let (seq_id, item) = self
                    .seq_ids
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, v)| v.as_ref().unwrap().1.load(Ordering::Relaxed))
                    .unwrap();

                let (seq, _, _) = item.as_ref().unwrap();
                self.trie.remove(&seq);
                seq_id as i32
            }
            Some(v) => v,
        };

        self.trie.insert(tokens.clone(), seq_id);
        self.seq_ids[seq_id as usize] = Some((
            tokens,
            AtomicI64::new(Utc::now().timestamp()),
            seq_data,
        ));
        Ok(seq_id)
    }
}
