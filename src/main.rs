#[macro_use]
extern crate log;

use anyhow::Result;
use clap::Parser;
use llama_cpp_2::llama_backend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use log::LevelFilter;
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Root};
use log4rs::encode::pattern::PatternEncoder;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;

mod api;
mod infer;
mod sampler;
#[allow(unused)]
mod metadata;

struct CompletionsTask {
    from_api: flume::Sender<LlamaToken>,
    input_token_list: Vec<LlamaToken>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<i64>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    maximum_tokens: Option<u32>
}

#[derive(Parser)]
#[command(version)]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0:30021")]
    bind_addr: SocketAddr,

    #[arg(short, long)]
    model_path: PathBuf,

    #[arg(short, long)]
    draft_model_path: Option<PathBuf>,

    #[arg(long)]
    model_name: String,

    #[arg(short, long, default_value_t = 4)]
    parallel_tasks: u32,

    #[arg(short, long, default_value_t = 512)]
    kv_cache_size_pre_task: u32,

    #[arg(short, long)]
    template: Option<String>,

    #[arg(long, default_value_t = 8)]
    max_unconfirmed_tokens: usize
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

fn exec(args: Args) -> Result<()> {
    logger_init()?;

    let rt = tokio::runtime::Runtime::new()?;
    let backend = llama_backend::LlamaBackend::init()?;
    let backend = Arc::new(backend);

    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(u32::MAX);

    let model = LlamaModel::load_from_file(&backend, &args.model_path, &model_params)?;
    let model = Arc::new(model);

    let draft_model = if let Some(draft_model_path) = args.draft_model_path {
        let draft_model = LlamaModel::load_from_file(&backend, &draft_model_path, &model_params)?;
        Some(Arc::new(draft_model))
    } else {
        None
    };

    let (tx, rx) = flume::bounded(1024);

    rt.block_on(async {
        let infer_handle = infer::run(
            model.clone(),
            draft_model,
            backend,
            rx,
            args.kv_cache_size_pre_task,
            args.parallel_tasks
        );

        let api_handle = api::run(
            args.bind_addr,
            model,
            args.model_name,
            args.kv_cache_size_pre_task,
            tx,
            args.template
        );

        tokio::try_join!(infer_handle, api_handle)?;
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
