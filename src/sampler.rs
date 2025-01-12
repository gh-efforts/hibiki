use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_sys_2::HibikiCommonSampler;

pub struct Sampler {
    inner: *mut HibikiCommonSampler
}

unsafe impl Send for Sampler {}
unsafe impl Sync for Sampler {}

impl Sampler {
    pub fn new(
        model: &LlamaModel,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        seed: Option<i64>,
        temperature: Option<f32>,
        top_p: Option<f32>,
    ) -> Sampler {
        unsafe {
            let params = llama_cpp_sys_2::hibiki_common_params_sampling_init();

            if let Some(f) = frequency_penalty {
                llama_cpp_sys_2::hibiki_common_params_sampling_set_frequency_penalty(params, f);
            }

            if let Some(p) = presence_penalty {
                llama_cpp_sys_2::hibiki_common_params_sampling_set_presence_penalty(params, p);
            }

            llama_cpp_sys_2::hibiki_common_params_sampling_set_seed(params, seed.map(|v| v as i32).unwrap_or_else(|| rand::random()));

            if let Some(t) = temperature {
                llama_cpp_sys_2::hibiki_common_params_sampling_set_temperature(params, t);
            }

            if let Some(t) = top_p {
                llama_cpp_sys_2::hibiki_common_params_sampling_set_top_p(params, t);
            }

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
