use llama_cpp_sys_2::HibikiCommonNgramCache;
use anyhow::Result;

pub struct NgranCache {
    inner: *mut HibikiCommonNgramCache
}

impl NgranCache {
    pub fn new() -> Self {
        unsafe {
            let p = llama_cpp_sys_2::hibiki_common_ngram_cache_new();
            Self { inner: p }
        }
    }

    pub fn update(&mut self) {

    }
}

impl Drop for NgranCache {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::hibiki_common_ngram_cache_free(self.inner);
        }
    }
}