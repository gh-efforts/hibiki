use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_sys_2::HibikiCommonSampler;

pub struct Sampler {
    inner: *mut HibikiCommonSampler
}

impl Sampler {
    pub fn new(model: &LlamaModel) -> Sampler {
        unsafe {
            let params = llama_cpp_sys_2::hibiki_common_params_sampling_init();
            let inner = llama_cpp_sys_2::hibiki_common_sampler_init(model.as_ptr(), params);
            llama_cpp_sys_2::hibiki_common_params_sampling_free(params);

            Sampler { inner }
        }
    }

    pub fn sample(&mut self, ctx: &mut LlamaContext, idx: i32) -> LlamaToken {
        unsafe {
            let token = llama_cpp_sys_2 ::hibiki_common_sampler_sample(self.inner, ctx.context.as_ptr(), idx, false);
            LlamaToken(token)
        }
    }

    pub fn accept(&mut self, token: LlamaToken) {
        unsafe {
            llama_cpp_sys_2::hibiki_common_sampler_accept(self.inner, token.0, false);
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {llama_cpp_sys_2::hibiki_common_sampler_free(self.inner);}
    }
}
