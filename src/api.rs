use crate::{CompletionsTask, EmbeddingTask};
use anyhow::{anyhow, ensure, Result};
use async_openai::types::{Base64Embedding, Base64EmbeddingVector, ChatChoice, ChatChoiceStream, ChatCompletionMessageToolCall, ChatCompletionRequestMessage, ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta, ChatCompletionToolType, Choice, CreateBase64EmbeddingResponse, CreateEmbeddingResponse, Embedding, EmbeddingInput, EmbeddingUsage, EncodingFormat, FinishReason, FunctionCall, Prompt, Role};
use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::post;
use axum::{Json, Router};
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use chrono::Utc;
use futures_util::StreamExt;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_sys_2::{hibiki_body_to_chat_params, hibiki_common_chat_params_free, hibiki_common_chat_parse, hibiki_common_chat_templates_free, hibiki_common_chat_templates_from_model, hibiki_get_common_chat_params_format, hibiki_get_common_chat_params_prompt, hibiki_get_common_chat_params_prompt_length, HibikiCommonChatFormat, HibikiCommonChatParams, HibikiCommonChatTemplates};
use serde::Deserialize;
use std::ffi::{CStr, CString};
use std::io::Write;
use std::net::SocketAddr;
use std::ptr::null;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct ChatTemplates {
    inner: *mut HibikiCommonChatTemplates
}

unsafe impl Send for ChatTemplates {}

unsafe impl Sync for ChatTemplates {}

impl ChatTemplates {
    fn from_model(model: &LlamaModel, tmpl: Option<&str>) -> Self {
        let cstr;

        let tmpl_cstr_ptr = if let Some(tmpl) = tmpl {
            cstr = CString::new(tmpl).unwrap();
            cstr.as_bytes_with_nul().as_ptr()
        } else {
            null()
        };

        ChatTemplates {
            inner: unsafe { hibiki_common_chat_templates_from_model(model.as_ptr(), tmpl_cstr_ptr as *const i8) },
        }
    }

    #[allow(unused)]
    fn as_ptr(&self) -> *const HibikiCommonChatTemplates {
        self.inner
    }

    #[allow(unused)]
    fn as_mut_ptr(&self) -> *mut HibikiCommonChatTemplates {
        self.inner
    }
}

impl Drop for ChatTemplates {
    fn drop(&mut self) {
        unsafe { hibiki_common_chat_templates_free(self.inner) }
    }
}

struct ChatParams {
    inner: *mut HibikiCommonChatParams
}

impl ChatParams {
    fn get_prompt(&self) -> Result<String> {
        unsafe {
            // exclude \0
            let strlen = hibiki_get_common_chat_params_prompt_length(self.inner);
            let mut str_buff = vec![0u8; strlen + 1];
            hibiki_get_common_chat_params_prompt(self.inner, str_buff.as_mut_ptr() as *mut i8);
            let str = CStr::from_bytes_until_nul(&str_buff)?.to_str()?.to_string();
            Ok(str)
        }
    }

    fn get_chat_format(&self) -> HibikiCommonChatFormat {
        unsafe { hibiki_get_common_chat_params_format(self.inner) }
    }
}

impl Drop for ChatParams {
    fn drop(&mut self) {
        unsafe {
            hibiki_common_chat_params_free(self.inner)
        }
    }
}

fn body_json_to_chat_params(tmpl: &ChatTemplates, body_json: &str) -> Result<ChatParams> {
    unsafe {
        let body_json = CString::new(body_json).unwrap();
        let params = hibiki_body_to_chat_params(tmpl.inner, body_json.as_bytes_with_nul().as_ptr() as *const i8);

        if params.is_null() {
            return Err(anyhow!("hibiki_body_to_chat_params failed"));
        }

        Ok(ChatParams { inner: params })
    }
}

#[derive(Deserialize, Debug)]
struct CommonChatToolCall {
    name: String,
    arguments: String,
    id: String,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
struct CommonChatMsgContentPart {
    r#type: String,
    text: String,
}

#[derive(Deserialize, Debug)]
#[allow(unused)]
struct CommonChatMsg {
    role: String,
    content: String,
    content_parts: Vec<CommonChatMsgContentPart>,
    tool_calls: Vec<CommonChatToolCall>,
    reasoning_content: String,
    tool_name: String,
    tool_call_id: String,
}

fn output_parse(msg: &str, format: HibikiCommonChatFormat) -> Result<CommonChatMsg> {
    let mut buff = vec![0u8; msg.len() + 8192];
    let msg = msg
        .replace(r#"\n"#, "")
        .replace(r#"\""#, "\"");

    let cs = CString::new(msg)?;

    unsafe {
        hibiki_common_chat_parse(cs.as_bytes_with_nul().as_ptr() as *const i8, format, buff.as_mut_ptr() as *mut i8);
        let out_json = CStr::from_bytes_until_nul(&buff)?.to_str()?.to_string();
        let out_msg: CommonChatMsg = serde_json::from_str(&out_json)?;
        Ok(out_msg)
    }
}

struct Context<Task> {
    model: Arc<LlamaModel>,
    model_name: String,
    backend_bridge: flume::Sender<Task>,
    kv_cache_size_pre_task: u32,
    chat_template: Option<Arc<ChatTemplates>>,
}

async fn completion_req_to_task(
    req: async_openai::types::CreateCompletionRequest,
    model: Arc<LlamaModel>,
    callback: flume::Sender<LlamaToken>,
) -> Result<CompletionsTask> {
    tokio::task::spawn_blocking(move || {
        let input_tokens = match req.prompt {
            Prompt::String(prompt) => {
                model.str_to_token(&prompt, AddBos::Always)?
            }
            _ => return Err(anyhow!("Only string prompts are supported")),
        };

        let task = CompletionsTask {
            to_api: callback,
            maximum_tokens: req.max_tokens,
            input_token_list: input_tokens,
            frequency_penalty : req.frequency_penalty,
            presence_penalty:  req.presence_penalty,
            seed: req.seed,
            temperature: req.temperature,
            top_p: req.top_p
        };
        Result::<_, anyhow::Error>::Ok(task)
    }).await?
}

async fn tokens_to_string(
    tokens: Vec<LlamaToken>,
    model: Arc<LlamaModel>
) -> Result<String> {
    tokio::task::spawn_blocking(move || {
        let out = model.tokens_to_str(&tokens, Special::Plaintext)?;
        Result::<_, anyhow::Error>::Ok(out)
    }).await?
}

async fn send_to_backend<Task: Send + Sync + 'static>(
    task: Task,
    ctx: &Context<Task>
) -> Result<()> {
    ctx.backend_bridge.send_async(task).await?;
    Ok(())
}
// ret: (task, chat_template_format)
async fn chat_completion_req_to_task(
    mut req: async_openai::types::CreateChatCompletionRequest,
    model: Arc<LlamaModel>,
    callback: flume::Sender<LlamaToken>,
    template: Arc<ChatTemplates>
) -> Result<(CompletionsTask, HibikiCommonChatFormat)> {
    tokio::task::spawn_blocking(move || {
        for msg in req.messages.iter_mut() {
            match msg {
                ChatCompletionRequestMessage::Assistant(msg) => {
                    if msg.content.is_none() {
                        msg.content = Some("".into());
                    }
                }
                ChatCompletionRequestMessage::Function(msg) => {
                    if msg.content.is_none() {
                        msg.content = Some("".into());
                    }
                }
                _ => ()
            }
        }
        let req_json = serde_json::to_string(&req)?;
        let params = body_json_to_chat_params(&template, req_json.as_str())?;
        debug!("body_json_to_chat_params finished");

        let prompt = params.get_prompt()?;
        debug!("prompt: {:?}", prompt);

        let format = params.get_chat_format();

        let input_tokens = model.str_to_token(&prompt, AddBos::Always)?;

        let task = CompletionsTask {
            to_api: callback,
            #[allow(deprecated)]
            maximum_tokens: req.max_tokens,
            input_token_list: input_tokens,
            frequency_penalty : req.frequency_penalty,
            presence_penalty:  req.presence_penalty,
            seed: req.seed,
            temperature: req.temperature,
            top_p: req.top_p
        };
        Result::<_, anyhow::Error>::Ok((task, format))
    }).await?
}

async fn v1_chat_completions(
    State(ctx): State<Arc<Context<CompletionsTask>>>,
    Json(req): Json<async_openai::types::CreateChatCompletionRequest>
) -> Response {
    debug!("v1_chat_completions: {:?}", req);

    let is_stream = req.stream.unwrap_or(false);
    let (tx, rx) = flume::unbounded();
    let chat_completion_id = rand::random::<u64>().to_string();

    let fut = async {
        if is_stream {
            ensure!(req.tools.is_none());
        }

        let (task, format) = chat_completion_req_to_task(req, ctx.model.clone(), tx, ctx.chat_template.as_ref().unwrap().clone()).await?;
        debug!("chat_completion_req_to_task finished");
        let prompt_tokens = task.input_token_list.len() as u32;
        ensure!(prompt_tokens < ctx.kv_cache_size_pre_task, "Prompt too large, prompt tokens len: {prompt_tokens}");
        send_to_backend(task, &*ctx).await?;

        let resp = if is_stream {
            let mut single_token_bytes = Vec::new();

            let out_stream = rx.into_stream()
                .map(move |token| {
                    let token_bytes = ctx.model.token_to_bytes(token, Special::Plaintext)?;
                    single_token_bytes.extend_from_slice(&token_bytes);

                    let text = match String::from_utf8(single_token_bytes.clone()) {
                        Ok(v) => v,
                        Err(_) => return Ok(None)
                    };

                    single_token_bytes.clear();

                    if log::max_level() >= log::Level::Debug {
                        print!("{}", text);
                        std::io::stdout().flush()?;
                    }

                    let chat_completion_resp = async_openai::types::CreateChatCompletionStreamResponse {
                        id: chat_completion_id.clone(),
                        choices: vec![
                            ChatChoiceStream {
                                index: 0,
                                #[allow(deprecated)]
                                delta: ChatCompletionStreamResponseDelta {
                                    content: Some(text),
                                    refusal: None,
                                    tool_calls: None,
                                    role: Some(Role::Assistant),
                                    function_call: None
                                },
                                finish_reason: None,
                                logprobs: None,
                            }
                        ],
                        created: Utc::now().timestamp() as u32,
                        model: ctx.model_name.clone(),
                        service_tier: None,
                        system_fingerprint: None,
                        object: String::from("chat.completion.chunk"),
                        usage: None
                    };

                    let event = axum::response::sse::Event::default()
                        .json_data(&chat_completion_resp)?;

                    Result::<_, anyhow::Error>::Ok(Some(event))
                })
                .filter_map(|v| async {
                    v.transpose()
                })
                .chain(futures_util::stream::once(async {
                    debug!("v1_chat_completions stream end");

                    let event = axum::response::sse::Event::default()
                        .data("[DONE]");
                    Ok(event)
                }));

            Sse::new(out_stream)
                .keep_alive(
                    axum::response::sse::KeepAlive::new()
                        .interval(Duration::from_secs(1))
                        .text("keep-alive-text"),
                )
                .into_response()
        } else {
            let mut out_tokens = Vec::new();
            let mut last = Instant::now();
            let mut avg_time = None;
            while let Ok(token) = rx.recv_async().await {
                let now = Instant::now();

                if let Some(avg) = avg_time {
                    avg_time = Some((avg + (now - last)) / 2);
                } else {
                    avg_time = Some(now - last);
                }

                last = now;
                out_tokens.push(token);

                if log::max_level() >= log::Level::Debug {
                    let token_bytes = ctx.model.token_to_bytes(token, Special::Plaintext)?;
                    let s = String::from_utf8_lossy(&token_bytes);
                    let mut lock = std::io::stdout().lock();
                    lock.write_all(s.as_bytes())?;
                    lock.flush()?;
                }
            }

            if let Some(avg) = avg_time {
                info!("avg time to decode one token use: {:?}", avg);
            }

            let completion_tokens = out_tokens.len() as u32;
            let text = tokens_to_string(out_tokens, ctx.model.clone()).await?;
            let chat_msg = output_parse(text.as_str(), format)?;
            debug!("chat_msg: {:?}", chat_msg);

            let chat_completion_resp = async_openai::types::CreateChatCompletionResponse {
                id: chat_completion_id,
                choices: vec![
                    ChatChoice {
                        finish_reason: {
                            if !chat_msg.tool_calls.is_empty() {
                                Some(FinishReason::ToolCalls)
                            } else {
                                None
                            }
                        },
                        index: 0,
                        #[allow(deprecated)]
                        message: ChatCompletionResponseMessage {
                            content: Some(chat_msg.content),
                            refusal: None,
                            tool_calls: {
                                let list = chat_msg.tool_calls.into_iter().map(|tool_call| {
                                    ChatCompletionMessageToolCall {
                                        id: {
                                            if tool_call.id.is_empty() {
                                                format!("call_def_{}", rand::random::<u32>())
                                            } else {
                                                tool_call.id
                                            }
                                        },
                                        r#type: ChatCompletionToolType::Function,
                                        function: FunctionCall {
                                            name: tool_call.name,
                                            arguments: tool_call.arguments,
                                        }
                                    }
                                })
                                .collect::<Vec<_>>();

                                Some(list)
                            },
                            role: Role::Assistant,
                            function_call: None,
                            audio: None
                        },
                        logprobs: None,
                    }
                ],
                created: Utc::now().timestamp() as u32,
                model: ctx.model_name.clone(),
                service_tier: None,
                system_fingerprint: None,
                object: String::from("chat.completion"),
                usage: Some(async_openai::types::CompletionUsage{
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    prompt_tokens_details: None,
                    completion_tokens_details: None
                })
            };

            let body = serde_json::to_vec(&chat_completion_resp)?;
            Response::new(Body::from(body))
        };

        Result::<_, anyhow::Error>::Ok(resp)
    };

    match fut.await {
        Ok(resp) => resp,
        Err(e) => {
            error!("v1_caht_completions error: {:?}", e);

            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(e.to_string()))
                .unwrap()
        }
    }
}

async fn v1_completions(
    State(ctx): State<Arc<Context<CompletionsTask>>>,
    Json(req): Json<async_openai::types::CreateCompletionRequest>
) -> Response {
    debug!("v1_completions: {:?}", req);

    let is_stream = req.stream.unwrap_or(false);
    let (tx, rx) = flume::unbounded();
    let completion_id = rand::random::<u64>().to_string();

    let fut = async {
        let task = completion_req_to_task(req, ctx.model.clone(), tx).await?;
        let prompt_tokens = task.input_token_list.len() as u32;
        ensure!(prompt_tokens < ctx.kv_cache_size_pre_task, "Prompt too large");
        send_to_backend(task, &*ctx).await?;

        let resp = if is_stream {
            let mut single_token_bytes = Vec::new();

            let out_stream = rx.into_stream()
                .map(move |token| {
                    let token_bytes = ctx.model.token_to_bytes(token, Special::Plaintext)?;
                    single_token_bytes.extend_from_slice(&token_bytes);

                    let text = match String::from_utf8(single_token_bytes.clone()) {
                        Ok(v) => v,
                        Err(_) => return Ok(None)
                    };

                    single_token_bytes.clear();

                    let completion_resp = async_openai::types::CreateCompletionResponse {
                        id: completion_id.clone(),
                        choices: vec![Choice{
                            text,
                            index: 0,
                            logprobs: None,
                            finish_reason: None
                        }],
                        created: Utc::now().timestamp() as u32,
                        model: ctx.model_name.clone(),
                        system_fingerprint: None,
                        object: "text_completion".to_string(),
                        usage: None
                    };

                    let event = axum::response::sse::Event::default()
                        .json_data(&completion_resp)?;

                    Result::<_, anyhow::Error>::Ok(Some(event))
                })
                .filter_map(|v| async {
                    v.transpose()
                })
                .chain(futures_util::stream::once(async {
                    let event = axum::response::sse::Event::default()
                        .data("[DONE]");
                    Ok(event)
                }));
            Sse::new(out_stream).into_response()
        } else {
            let mut out_tokens = Vec::new();
            while let Ok(token) = rx.recv_async().await {
                out_tokens.push(token);
            }

            let completion_tokens = out_tokens.len() as u32;
            let text = tokens_to_string(out_tokens, ctx.model.clone()).await?;

            let completion_resp = async_openai::types::CreateCompletionResponse {
                id: completion_id,
                choices: vec![Choice{
                    text,
                    index: 0,
                    logprobs: None,
                    finish_reason: None
                }],
                created: Utc::now().timestamp() as u32,
                model: ctx.model_name.clone(),
                system_fingerprint: None,
                object: "text_completion".to_string(),
                usage: Some(async_openai::types::CompletionUsage{
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    prompt_tokens_details: None,
                    completion_tokens_details: None
                })
            };

            let body = serde_json::to_vec(&completion_resp)?;
            Response::new(Body::from(body))
        };
        Result::<_, anyhow::Error>::Ok(resp)
    };

    match fut.await {
        Ok(resp) => resp,
        Err(e) => {
            error!("v1_completions error: {:?}", e);

            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(e.to_string()))
                .unwrap()
        }
    }
}

async fn embedding_req_to_task(
    req: async_openai::types::CreateEmbeddingRequest,
    model: Arc<LlamaModel>,
    callback: flume::Sender<Vec<f32>>,
) -> Result<Vec<EmbeddingTask>> {
    tokio::task::spawn_blocking(move || {
        let input_tokens = match req.input {
            EmbeddingInput::String(prompt) => {
                vec![model.str_to_token(&prompt, AddBos::Never)?]
            },
            EmbeddingInput::StringArray(prompts) => {
                let mut token_list = Vec::with_capacity(prompts.len());

                for prompt in prompts {
                    token_list.push(model.str_to_token(&prompt, AddBos::Never)?);
                }
                token_list
            },
            _ => return Err(anyhow!("Only string prompts are supported")),
        };

        let tasks = input_tokens.into_iter()
            .map(|tokens| {
                EmbeddingTask {
                    to_api: callback.clone(),
                    input_token_list: tokens
                }
            })
            .collect::<Vec<_>>();

        Result::<_, anyhow::Error>::Ok(tasks)
    }).await?
}

async fn v1_embedding(
    State(ctx): State<Arc<Context<EmbeddingTask>>>,
    Json(req): Json<async_openai::types::CreateEmbeddingRequest>
) -> Response {
    let fut = async {
        let (tx, rx) = flume::unbounded();
        let format = req.encoding_format.clone().unwrap_or(EncodingFormat::Float);
        let tasks= embedding_req_to_task(req, ctx.model.clone(), tx).await?;

        let mut total_tokens = 0;
        for task in tasks {
            let input_tokens = task.input_token_list.len() as u32;
            ensure!(input_tokens <= ctx.kv_cache_size_pre_task, "input tokens too large, input tokens len: {input_tokens}");

            send_to_backend(task, &*ctx).await?;
            total_tokens += input_tokens;
        }

        let out = match format {
            EncodingFormat::Float => {
                let resp = CreateEmbeddingResponse  {
                    object: String::from("list"),
                    model: ctx.model_name.clone(),
                    data: {
                        let mut out = Vec::new();

                        while let Ok(embeddings) = rx.recv_async().await  {
                            out.push(Embedding {
                                index: out.len() as u32,
                                object: String::from("embedding"),
                                embedding: embeddings
                            });
                        }
                        out
                    },
                    usage: EmbeddingUsage {
                        prompt_tokens: total_tokens,
                        total_tokens
                    }
                };

                serde_json::to_vec(&resp)?
            }
            EncodingFormat::Base64 => {
                let resp = CreateBase64EmbeddingResponse {
                    object: String::from("list"),
                    model: ctx.model_name.clone(),
                    data: {
                        let mut out = Vec::new();

                        while let Ok(embeddings) = rx.recv_async().await {
                            out.push(Base64Embedding {
                                index: out.len() as u32,
                                object: String::from("embedding"),
                                embedding: Base64EmbeddingVector({
                                    let mut buff: Vec<u8> = Vec::with_capacity(embeddings.len() * 4);

                                    for x in embeddings {
                                        buff.extend_from_slice(&x.to_le_bytes());
                                    }

                                    BASE64_STANDARD.encode(&buff)
                                })
                            });
                        }
                        out
                    },
                    usage: EmbeddingUsage {
                        prompt_tokens: total_tokens,
                        total_tokens
                    }
                };
                serde_json::to_vec(&resp)?
            }
        };

        let resp = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(out))?;

        Result::<_, anyhow::Error>::Ok(resp)
    };

    match fut.await {
        Ok(resp) => resp,
        Err(e) => {
            error!("v1_embedding error: {:?}", e);

            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(e.to_string()))
                .unwrap()
        }
    }
}

pub async fn run_embedding(
    bind_addr: SocketAddr,
    model: Arc<LlamaModel>,
    model_name: String,
    kv_cache_size_pre_task: u32,
    backend_bridge: flume::Sender<EmbeddingTask>,
) -> Result<()> {
    let ctx = Context {
        model,
        model_name,
        backend_bridge,
        kv_cache_size_pre_task,
        chat_template: None
    };

    let ctx = Arc::new(ctx);
    let app = Router::new()
        .route("/v1/embeddings", post(v1_embedding))
        .with_state(ctx);

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    info!("Listening on http://{}", bind_addr);
    axum::serve(listener, app).await?;
    Ok(())
}

pub async fn run_completions(
    bind_addr: SocketAddr,
    model: Arc<LlamaModel>,
    model_name: String,
    kv_cache_size_pre_task: u32,
    backend_bridge: flume::Sender<CompletionsTask>,
    template: Option<String>,
) -> Result<()> {
    let template = ChatTemplates::from_model(&model, template.as_deref());

    let ctx = Context {
        model,
        model_name,
        backend_bridge,
        kv_cache_size_pre_task,
        chat_template: Some(Arc::new(template))
    };

    let ctx = Arc::new(ctx);

    let app = Router::new()
        .route("/v1/completions", post(v1_completions))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .with_state(ctx);

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    info!("Listening on http://{}", bind_addr);
    axum::serve(listener, app).await?;
    Ok(())
}