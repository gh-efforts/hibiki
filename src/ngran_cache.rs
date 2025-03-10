use llama_cpp_sys_2::{llama_token, HibikiCommonNgramCache};

pub const NGRAM_MIN: i32 = 1;
pub const NGRAM_MAX: i32 = 4;

pub struct NgranCache {
    nc_context: *mut HibikiCommonNgramCache,
    nc_dynamic: *mut HibikiCommonNgramCache,
    nc_static: *mut HibikiCommonNgramCache,
}

impl NgranCache {
    pub fn new() -> Self {
        unsafe {
            Self {
                nc_context: llama_cpp_sys_2::hibiki_common_ngram_cache_new(),
                nc_dynamic: llama_cpp_sys_2::hibiki_common_ngram_cache_new(),
                nc_static: llama_cpp_sys_2::hibiki_common_ngram_cache_new(),
            }
        }
    }

    pub fn update(
        &mut self,
        ngram_min: i32,
        ngram_max: i32,
        inp_data: &[llama_token],
        nnew: i32
    ) {
        unsafe {
            llama_cpp_sys_2::hibiki_common_ngram_cache_update(
                self.nc_context,
                ngram_min,
                ngram_max,
                inp_data.as_ptr(),
                inp_data.len() as i32,
                nnew,
                false
            )
        }
    }

    pub fn draft(
        &self,
        inp_data: &[llama_token],
        draft: &mut [llama_token],
        ngram_min: i32,
        ngram_max: i32,
    ) -> usize {
        let mut draft_len = 0;

        unsafe {
            llama_cpp_sys_2::hibiki_common_ngram_cache_draft(
                inp_data.as_ptr(),
                inp_data.len() as i32,
                draft.as_mut_ptr(),
                &mut draft_len,
                draft.len() as i32 - 1,
                ngram_min,
                ngram_max,
                self.nc_context,
                self.nc_dynamic,
                self.nc_static
            )
        }
        draft_len as usize
    }
}

impl Drop for NgranCache {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::hibiki_common_ngram_cache_free(self.nc_context);
            llama_cpp_sys_2::hibiki_common_ngram_cache_free(self.nc_dynamic);
            llama_cpp_sys_2::hibiki_common_ngram_cache_free(self.nc_static);
        }
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_2::llama_backend;
    use crate::ngran_cache::NgranCache;

    #[test]
    fn test() {
        let _backend = llama_backend::LlamaBackend::init().unwrap();

        let mut cache = NgranCache::new();
        let inp_data = [1, 2, 3];
        cache.update(1, 4, &inp_data, inp_data.len() as i32);
        cache.update(1, 4, &inp_data, inp_data.len() as i32);

        let mut out = [0i32; 10];
        out[0] = 2;
        let len = cache.draft(&[1, 2], &mut out, 2, 4);

        println!("{:?}", &out[1..len]);
    }
}