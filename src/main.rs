#[macro_use]
extern crate log;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;
use llama_cpp_2::sampling::LlamaSampler;
use clap::Parser;
use llama_cpp_2::token::LlamaToken;
use anyhow::Result;
use llama_cpp_2::llama_backend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Root};
use log4rs::encode::pattern::PatternEncoder;
use log::LevelFilter;

mod api;
mod infer;

struct CompletionsTask {
    callback: flume::Sender<LlamaToken>,
    input_token_list: Vec<LlamaToken>,
    sampler: LlamaSampler,
    maximum_tokens: u32
}

#[derive(Parser)]
#[command(version)]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0:30021")]
    bind_addr: SocketAddr,

    #[arg(short, long)]
    model_path: PathBuf,

    #[arg(long)]
    model_name: String,

    #[arg(short, long, default_value_t = 512)]
    kv_cache_size_pre_task: u32,

    #[arg(short, long)]
    template: Option<String>
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

    let (tx, rx) = flume::bounded(1024);

    rt.block_on(async {
        let infer_handle = infer::run(
            model.clone(),
            backend,
            rx,
            args.kv_cache_size_pre_task
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
