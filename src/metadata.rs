use std::cmp::min;
use std::ffi::{c_char, CStr, CString};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_sys_2::ggml_row_size;

// Retrieves a value in string form from a model's metadata.
///
/// # Parameters
///
/// * `model` - a pointer to the model to retrieve values from.
/// * `key` - the key of the metadata value.
///
/// #  Limitations
///
/// At the moment, the implementation will retrieves values of limited length, so this shouldn't be used to retrieve
/// something like the model's grammar.
pub fn get_metadata_raw(model: &LlamaModel, key: &str) -> String {
    let c_key = if let Some(stripped) = key.strip_prefix("%s") {
        let arch_key = CStr::from_bytes_with_nul(b"general.architecture\0").unwrap(); // Should never fail
        let mut arch_val = vec![0u8; 128];

        let res = unsafe {
            llama_cpp_sys_2::llama_model_meta_val_str(
                model.as_ptr(),
                arch_key.as_ptr(),
                arch_val.as_mut_ptr() as *mut c_char,
                arch_val.len(),
            )
        };

        if let Ok(len) = usize::try_from(res) {
            if let Ok(c_str) = CStr::from_bytes_with_nul(&arch_val[..=len]) {
                let formatted = format!("{}{stripped}", c_str.to_string_lossy());
                CString::new(formatted.as_bytes()).unwrap()
            } else {
                // This should be unreachable
                error!("Could not parse architecture metadata");
                return String::new();
            }
        } else {
            // This should be unreachable
            error!("Could not find architecture metadata");
            return String::new();
        }
    } else {
        CString::new(key).unwrap()
    };

    // This implementation assumes large values such as the model's vocabulary will never be queried
    let mut val = vec![0u8; 128];
    let res = unsafe {
        llama_cpp_sys_2::llama_model_meta_val_str(
            model.as_ptr(),
            c_key.as_ptr(),
            val.as_mut_ptr() as *mut c_char,
            val.len(),
        )
    };

    if let Ok(len) = usize::try_from(res) {
        if let Ok(val_str) = CStr::from_bytes_with_nul(&val[..=len])
            .map(move |val| val.to_string_lossy().to_string())
        {
            val_str
        } else {
            error!("Failed to parse retrieved metadata");
            String::new()
        }
    } else {
        warn!("{} Could not find metadata", key);
        String::new()
    }
}

pub struct ModelMetadata {
    /// The size of this model's vocabulary, in tokens.
    pub vocabulary_size: usize,

    /// The beginning of sentence (BOS) token for this model.
    pub bos_token: LlamaToken,

    /// The end of sentence (EOS) token for this model.
    pub eos_token: LlamaToken,

    /// The newline (NL) token for this model.
    pub nl_token: LlamaToken,

    // /// For infilling, the prefix token for this model.
    // infill_prefix_token: LlamaToken,
    //
    // /// For infilling, the middle token for this model.
    // infill_middle_token: LlamaToken,
    //
    // /// For infilling, the suffix token for this model.
    // infill_suffix_token: LlamaToken,
    //
    // /// For infilling, the token for the end of the infill.
    // eot_token: LlamaToken,

    /// For embeddings, the length of a single embeddings vector.
    pub embedding_length: usize,

    /// The number of tokens in the context the model was trained with.
    pub training_size: usize,

    /// The number of layers in the model's network.
    pub layers: usize,

    /// ???
    pub kv_heads: usize,
    /// Dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    pub k_attention: usize,
    /// Dimension of values (d_v) aka n_embd_head
    pub v_attention: usize,

    /// State Space Models conv kernel
    pub ssm_d_conv: usize,
    /// State Space Models inner size
    pub ssm_d_inner: usize,
    /// State Space Models state size
    pub ssm_d_state: usize,
}

impl From<&LlamaModel> for ModelMetadata {
    fn from(model: &LlamaModel) -> Self {
        let vocabulary_size = model.n_vocab();
        let n_embd = model.n_embd() as usize;

        let heads = get_metadata_raw(model, "%s.attention.head_count")
            .parse::<usize>()
            .unwrap_or(0);

        let layers = get_metadata_raw(model, "%s.block_count")
            .parse::<usize>()
            .unwrap_or(0);
        let kv_heads = get_metadata_raw(model, "%s.attention.head_count_kv")
            .parse::<usize>()
            .unwrap_or(heads);
        let k_attention = get_metadata_raw(model, "%s.attention.key_length")
            .parse::<usize>()
            .unwrap_or(n_embd / heads);
        let v_attention = get_metadata_raw(model, "%s.attention.value_length")
            .parse::<usize>()
            .unwrap_or(n_embd / heads);
        let ssm_d_conv = get_metadata_raw(model, "%s.ssm.conv_kernel")
            .parse::<usize>()
            .unwrap_or(0);
        let ssm_d_inner = get_metadata_raw(model, "%s.ssm.inner_size")
            .parse::<usize>()
            .unwrap_or(0);
        let ssm_d_state = get_metadata_raw(model, "%s.ssm.state_size")
            .parse::<usize>()
            .unwrap_or(0);

        let out = ModelMetadata {
            vocabulary_size: vocabulary_size as usize,
            bos_token: model.token_bos(),
            eos_token: model.token_eos(),
            nl_token: model.token_nl(),
            embedding_length: n_embd,
            training_size: model.n_ctx_train() as usize,
            layers,
            kv_heads,
            k_attention,
            v_attention,
            ssm_d_conv,
            ssm_d_inner,
            ssm_d_state,
        };

        out
    }
}

/// Memory requirements for something.
///
/// This is typically returned by [`LlamaModel::estimate_session_size`] and
/// [`LlamaModel::estimate_embeddings_session_size`] as an estimation of memory usage.
#[derive(Debug)]
pub struct ResourceUsage {
    kv_cache: usize
}

impl ModelMetadata {
    // TODO while llama doesn't offer memory estimation utilities, this is the best that can be done realistically
    // https://github.com/ggerganov/llama.cpp/issues/4315
    pub fn estimate_session_size(&self, session_params: &LlamaContextParams) -> ResourceUsage {
        let kv_size = session_params.n_ctx().unwrap().get() as i64; // TODO exception for mamba arch

        // dimension of key embeddings across all k-v heads
        let n_embd_k_gqa = self.k_attention * self.kv_heads;
        // dimension of value embeddings across all k-v heads
        let n_embd_v_gqa = self.v_attention * self.kv_heads;

        // dimension of the rolling state embeddings
        let n_embd_k_s = if self.ssm_d_conv > 0 {
            (self.ssm_d_conv - 1) * self.ssm_d_inner
        } else {
            0
        };
        // dimension of the recurrent state embeddings
        let n_embd_v_s = self.ssm_d_state * self.ssm_d_inner;

        let k_row_size = unsafe {
            ggml_row_size(
                session_params.context_params.type_k.into(),
                (n_embd_k_gqa + n_embd_k_s) as i64 * kv_size,
            )
        };
        let v_row_size = unsafe {
            ggml_row_size(
                session_params.context_params.type_v.into(),
                (n_embd_v_gqa + n_embd_v_s) as i64 * kv_size,
            )
        };

        let cache_size = self.layers * (k_row_size + v_row_size);
        info!("KV cache size: {}MB", cache_size / 1024 / 1024);

        // let batch = min(session_params.n_ctx().unwrap().get(), session_params.n_batch()) as usize;
        // let logits_size = self.vocabulary_size * batch;
        // let embed_size = if session_params.embeddings() {
        //     self.embedding_length * batch
        // } else {
        //     0
        // };
        // let output_size = (logits_size + embed_size) * size_of::<f32>();
        // info!("Output buffer size: {}MB", output_size / 1024 / 1024);

        // const LLAMA_MAX_NODES: usize = 8192;
        //
        // let compute_size = unsafe {
        //     ggml_tensor_overhead() * LLAMA_MAX_NODES
        //         + ggml_graph_overhead_custom(LLAMA_MAX_NODES, false)
        // };

        ResourceUsage {
            kv_cache: cache_size
        }
    }
}

