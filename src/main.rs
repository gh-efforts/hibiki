#[macro_use]
extern crate log;

use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use llama_cpp_2::llama_backend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_sys_2::{ggml_backend_dev_t, ggml_backend_device_register, ggml_backend_reg_by_name, ggml_backend_reg_get_proc_address, llama_split_mode, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0};
use log::LevelFilter;
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Root};
use log4rs::encode::pattern::PatternEncoder;
use std::ffi::CString;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;

mod api;
mod infer;
mod sampler;
mod radixtrie_kv_cache;
#[allow(unused)]
mod metadata;

struct CompletionsTask {
    to_api: flume::Sender<LlamaToken>,
    input_token_list: Vec<LlamaToken>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<i64>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    maximum_tokens: Option<u32>
}

struct EmbeddingTask {
    to_api: flume::Sender<Vec<f32>>,
    input_token_list: Vec<LlamaToken>
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
enum SplitMode {
    Layer = 1,
    Row = 2
}

#[derive(Copy, Clone, Eq, PartialEq, ValueEnum)]
#[allow(non_camel_case_types)]
enum KVCacheTypes {
    F32 = GGML_TYPE_F32 as isize,
    F16 = GGML_TYPE_F16 as isize,
    BF16 = GGML_TYPE_BF16 as isize,
    Q8_0 = GGML_TYPE_Q8_0 as isize,
    Q4_0 = GGML_TYPE_Q4_0 as isize,
    Q4_1 = GGML_TYPE_Q4_1 as isize,
    IQ4_NL = GGML_TYPE_IQ4_NL as isize,
    Q5_0 = GGML_TYPE_Q5_0 as isize,
    Q5_1 = GGML_TYPE_Q5_1 as isize,
}

#[derive(Parser)]
#[command(version)]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0:30021")]
    bind_addr: SocketAddr,

    #[arg(short, long)]
    model_path: PathBuf,

    #[arg(long)]
    model_main_gpu: Option<i32>,

    #[arg(long)]
    split_mode: Option<SplitMode>,

    #[arg(long)]
    model_tensor_split_rate: Option<String>,

    #[arg(short, long)]
    draft_model_path: Option<PathBuf>,

    #[arg(long)]
    draft_model_main_gpu: Option<i32>,

    #[arg(long)]
    draft_model_tensor_split_rate: Option<String>,

    #[arg(long)]
    rpc_servers: Option<String>,

    #[arg(long)]
    model_name: String,

    #[arg(short, long, default_value_t = 4)]
    parallel_tasks: u32,

    #[arg(short, long, default_value_t = 512)]
    kv_cache_size_pre_task: u32,

    #[arg(short, long)]
    template: Option<String>,

    #[arg(long, default_value_t = 8)]
    max_unconfirmed_tokens: usize,

    #[arg(long, default_value_t = 16)]
    n_candidates: usize,

    #[arg(long, default_value_t = false)]
    disable_offload_kqv: bool,

    #[arg(long)]
    type_k: Option<KVCacheTypes>,

    #[arg(long)]
    type_v: Option<KVCacheTypes>,

    #[arg(long)]
    draft_type_k: Option<KVCacheTypes>,

    #[arg(long)]
    draft_type_v: Option<KVCacheTypes>,

    #[arg(long, default_value_t = false)]
    embedding: bool
}

fn logger_init() -> Result<()> {
    let log_level = LevelFilter::from_str(
        std::env::var("HIBIKI_LOG").as_deref().unwrap_or("INFO"),
    )?;

    let pattern = if log_level >= LevelFilter::Debug {
        "[{d(%Y-%m-%d %H:%M:%S)}] {h({l})} {f}:{L} - {m}{n}"
    } else {
        "[{d(%Y-%m-%d %H:%M:%S)}] {h({l})} {t} - {m}{n}"
    };

    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build();

    let config = log4rs::Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(
            Root::builder()
                .appender("stdout")
                .build(log_level),
        )?;

    log4rs::init_config(config)?;
    Ok(())
}

fn add_rpc_devices(servers: &str) -> Result<()> {
    type GgmlBackendRpcAddDeviceFn = fn(*const std::ffi::c_char) -> ggml_backend_dev_t;
    let servers = servers.split(",")
        .collect::<Vec<_>>();

    if servers.is_empty() {
        return Err(anyhow!("no RPC servers specified"));
    }
    unsafe {
        let rpc_reg = ggml_backend_reg_by_name(CString::new("RPC")?.as_ptr());

        if rpc_reg.is_null() {
            return Err(anyhow!("failed to find RPC backend"));
        }

        let add_device_ptr = ggml_backend_reg_get_proc_address(rpc_reg, CString::new("ggml_backend_rpc_add_device")?.as_ptr());

        if add_device_ptr.is_null() {
            return Err(anyhow!("failed to find RPC device add function"));
        }
        let add_device_fn: GgmlBackendRpcAddDeviceFn = std::mem::transmute(add_device_ptr);

        for server in servers {
            let dev = add_device_fn(CString::new(server)?.as_ptr()) ;

            if dev.is_null() {
                return Err(anyhow!("failed to register RPC device"));
            } else {
                ggml_backend_device_register(dev);
            }
        }
    }
    Ok(())
}

fn exec(args: Args) -> Result<()> {
    logger_init()?;

    let rt = tokio::runtime::Runtime::new()?;

    if let Some(rpc_servers) = args.rpc_servers {
        add_rpc_devices(&rpc_servers)?;
    }

    let backend = llama_backend::LlamaBackend::init()?;
    let backend = Arc::new(backend);

    let mut model_params = LlamaModelParams::default()
        .with_n_gpu_layers(u32::MAX);

    if let Some(gpu_idx) = args.model_main_gpu {
        model_params = model_params.with_main_gpu(gpu_idx);
    }

    if let Some(split_mode) = args.split_mode {
        model_params.params.split_mode = split_mode as llama_split_mode;
    }

    if let Some(split) = &args.model_tensor_split_rate {
        let mut split_list = Vec::new();
        for x in split.split(",") {
            split_list.push(f32::from_str(x)?);
        }
        model_params.params.tensor_split = split_list.as_ptr();
    }

    let model = LlamaModel::load_from_file(&backend, &args.model_path, &model_params)?;
    let model = Arc::new(model);

    let draft_model = if let Some(draft_model_path) = args.draft_model_path {
        let mut draft_model_params = LlamaModelParams::default()
            .with_n_gpu_layers(u32::MAX);

        if let Some(gpu_idx) = args.draft_model_main_gpu {
            draft_model_params = draft_model_params.with_main_gpu(gpu_idx);
        }

        if let Some(split_mode) = args.split_mode {
            draft_model_params.params.split_mode = split_mode as llama_split_mode;
        }

        if let Some(split) = &args.draft_model_tensor_split_rate {
            let mut split_list = Vec::new();
            for x in split.split(",") {
                split_list.push(f32::from_str(x)?);
            }
            draft_model_params.params.tensor_split = split_list.as_ptr();
        }

        let draft_model = LlamaModel::load_from_file(&backend, &draft_model_path, &draft_model_params)?;
        Some(Arc::new(draft_model))
    } else {
        None
    };

    rt.block_on(async {
        if args.embedding {
            let (tx, rx) = flume::bounded(1024);

            let infer_handle = infer::run_embedding(
                model.clone(),
                backend,
                rx,
                args.kv_cache_size_pre_task,
                args.parallel_tasks,
                !args.disable_offload_kqv,
                args.type_k,
                args.type_v,
            );

            let api_handle = api::run_embedding(
                args.bind_addr,
                model,
                args.model_name,
                args.kv_cache_size_pre_task,
                tx,
            );

            tokio::try_join!(infer_handle, api_handle)?;
        } else {
            let (tx, rx) = flume::bounded(1024);

            let infer_handle = infer::run_completions(
                model.clone(),
                draft_model,
                backend,
                rx,
                args.kv_cache_size_pre_task,
                args.parallel_tasks,
                args.max_unconfirmed_tokens,
                args.n_candidates,
                !args.disable_offload_kqv,
                args.type_k,
                args.type_v,
                args.draft_type_k,
                args.draft_type_v
            );

            let api_handle = api::run_completions(
                args.bind_addr,
                model,
                args.model_name,
                args.kv_cache_size_pre_task,
                tx,
                args.template
            );

            tokio::try_join!(infer_handle, api_handle)?;
        }

        Ok(())
    })
}

fn main() -> ExitCode {
    let args = Args::parse();

    match exec(args) {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{:?}", e);
            ExitCode::FAILURE
        }
    }
}
