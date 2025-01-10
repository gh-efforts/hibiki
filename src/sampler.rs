use std::ptr::null;
use llama_cpp_2::model::LlamaModel;

unsafe fn test(model: &LlamaModel) {
    // llama_cpp_sys_2::common_sampler_init(model.as_ptr(), null());
}