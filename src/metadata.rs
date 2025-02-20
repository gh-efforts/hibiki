use std::ffi::{c_char, CStr, CString};
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;

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

impl TryFrom<&LlamaModel> for ModelMetadata {
    type Error = ();

    fn try_from(model: &LlamaModel) -> Result<Self, Self::Error> {
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

        Ok(out)
    }
}

