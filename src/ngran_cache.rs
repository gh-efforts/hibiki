use llama_cpp_sys_2::{llama_token, HibikiCommonNgramCache};
use anyhow::Result;

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
                draft.len() as i32,
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