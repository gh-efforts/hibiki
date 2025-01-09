use crate::CompletionsTask;
use anyhow::{anyhow, ensure, Result};
use async_openai::types::{ChatChoice, ChatChoiceStream, ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessageContent, ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta, Choice, Prompt, Role};
use axum::body::Body;
use axum::extract::State;
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::post;
use axum::{Json, Router};
use chrono::Utc;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::cmp::min;
use std::net::SocketAddr;
use std::sync::Arc;
use axum::http::StatusCode;
use futures_util::StreamExt;

struct Context {
    model: Arc<LlamaModel>,
    model_name: String,
    maximum_tokens: u32,
    backend_bridge: flume::Sender<CompletionsTask>,
    kv_cache_size_pre_task: u32,
    chat_template: Option<String>,
}

async fn completion_req_to_task(
    req: async_openai::types::CreateCompletionRequest,
    model: Arc<LlamaModel>,
    callback: flume::Sender<LlamaToken>,
    maximum_tokens: u32
) -> Result<CompletionsTask> {
    tokio::task::spawn_blocking(move || {
        let input_tokens = match req.prompt {
            Prompt::String(prompt) => {
                model.str_to_token(&prompt, AddBos::Always)?
            }
            _ => return Err(anyhow!("Only string prompts are supported")),
        };

        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(req.temperature.unwrap_or(1.0)),
            LlamaSampler::top_p(req.top_p.unwrap_or(1.0), 1),
            LlamaSampler::penalties_simple(
                -1,
                1.0,
                req.frequency_penalty.unwrap_or(0.0),
                req.presence_penalty.unwrap_or(0.0)
            ),
            LlamaSampler::dist(req.seed.map(|v| v as u32).unwrap_or_else(|| rand::random()))
        ]);

        let task = CompletionsTask {
            callback,
            input_token_list: input_tokens,
            sampler,
            maximum_tokens: min(req.max_tokens.unwrap_or(maximum_tokens), maximum_tokens)
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

async fn send_to_backend(
    task: CompletionsTask,
    ctx: &Context
) -> Result<()> {
    ctx.backend_bridge.send_async(task).await?;
    Ok(())
}

async fn chat_completion_req_to_task(
    req: async_openai::types::CreateChatCompletionRequest,
    model: Arc<LlamaModel>,
    callback: flume::Sender<LlamaToken>,
    maximum_tokens: u32,
    template: Option<String>,
) -> Result<CompletionsTask> {
    tokio::task::spawn_blocking(move || {
        let mut chat_messages = Vec::new();

        for msg in req.messages {
            let chat_msg = match msg {
                ChatCompletionRequestMessage::System(v) => {
                    let content = match v.content {
                        ChatCompletionRequestSystemMessageContent::Text(content) => content,
                        ChatCompletionRequestSystemMessageContent::Array(_) => {
                            return Err(anyhow!("Array content not supported"));
                        }
                    };
                    LlamaChatMessage::new(String::from("system"), String::from(content))?
                }
                ChatCompletionRequestMessage::User(v) => {
                    let content = match v.content {
                        ChatCompletionRequestUserMessageContent::Text(content) => content,
                        ChatCompletionRequestUserMessageContent::Array(_) => {
                            return Err(anyhow!("Array content not supported"));
                        }
                    };
                    LlamaChatMessage::new(String::from("user"), String::from(content))?
                }
                ChatCompletionRequestMessage::Assistant(v) => {
                    if let Some(content) = v.content {
                        let content = match content {
                            ChatCompletionRequestAssistantMessageContent::Text(content) => content,
                            ChatCompletionRequestAssistantMessageContent::Array(_) => {
                                return Err(anyhow!("Array content not supported"));
                            }
                        };
                        LlamaChatMessage::new(String::from("user"), String::from(content))?
                    } else {
                        continue;
                    }
                }
                _ => return Err(anyhow!("Only system and user messages are supported")),
            };

            chat_messages.push(chat_msg);
        }

        let prompt = model.apply_chat_template(template, chat_messages, false)?;
        let input_tokens = model.str_to_token(&prompt, AddBos::Always)?;

        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(req.temperature.unwrap_or(1.0)),
            LlamaSampler::top_p(req.top_p.unwrap_or(1.0), 1),
            LlamaSampler::penalties_simple(
                -1,
                1.0,
                req.frequency_penalty.unwrap_or(0.0),
                req.presence_penalty.unwrap_or(0.0)
            ),
            LlamaSampler::dist(req.seed.map(|v| v as u32).unwrap_or_else(|| rand::random()))
        ]);

        let task = CompletionsTask {
            callback,
            input_token_list: input_tokens,
            sampler,
            maximum_tokens: min(req.max_tokens.unwrap_or(maximum_tokens), maximum_tokens)
        };
        Result::<_, anyhow::Error>::Ok(task)
    }).await?
}

async fn v1_chat_completions(
    State(ctx): State<Arc<Context>>,
    Json(req): Json<async_openai::types::CreateChatCompletionRequest>
) -> Response {
    let is_stream = req.stream.unwrap_or(false);
    let (tx, rx) = flume::unbounded();
    let chat_completion_id = rand::random::<u64>().to_string();

    let fut = async {
        let task = chat_completion_req_to_task(req, ctx.model.clone(), tx, ctx.maximum_tokens, ctx.chat_template.clone()).await?;
        let prompt_tokens = task.input_token_list.len() as u32;
        ensure!(prompt_tokens <= ctx.kv_cache_size_pre_task, "Prompt too large");
        send_to_backend(task, &*ctx).await?;

        let resp = if is_stream {
            let out_stream = rx.into_stream()
                .map(move |token| {
                    let text = ctx.model.token_to_str(token, Special::Plaintext)?;

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
                        .json_data(serde_json::to_vec(&chat_completion_resp)?)?;

                    Result::<_, anyhow::Error>::Ok(event)
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

            let chat_completion_resp = async_openai::types::CreateChatCompletionResponse {
                id: chat_completion_id,
                choices: vec![
                    ChatChoice {
                        index: 0,
                        #[allow(deprecated)]
                        message: ChatCompletionResponseMessage {
                            content: Some(text),
                            refusal: None,
                            tool_calls: None,
                            role: Role::Assistant,
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
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(e.to_string()))
                .unwrap()
        }
    }
}

async fn v1_completions(
    State(ctx): State<Arc<Context>>,
    Json(req): Json<async_openai::types::CreateCompletionRequest>
) -> Response {
    let is_stream = req.stream.unwrap_or(false);
    let (tx, rx) = flume::unbounded();
    let completion_id = rand::random::<u64>().to_string();

    let fut = async {
        let task = completion_req_to_task(req, ctx.model.clone(), tx, ctx.maximum_tokens,).await?;
        let prompt_tokens = task.input_token_list.len() as u32;
        ensure!(prompt_tokens <= ctx.kv_cache_size_pre_task, "Prompt too large");
        send_to_backend(task, &*ctx).await?;

        let resp = if is_stream {
            let out_stream = rx.into_stream()
                .map(move |token| {
                    let text = ctx.model.token_to_str(token, Special::Plaintext)?;

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
                        .json_data(serde_json::to_vec(&completion_resp)?)?;

                    Result::<_, anyhow::Error>::Ok(event)
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
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(e.to_string()))
                .unwrap()
        }
    }
}

pub async fn run(
    bind_addr: SocketAddr,
    model: Arc<LlamaModel>,
    model_name: String,
    maximum_tokens: u32,
    kv_cache_size_pre_task: u32,
    backend_bridge: flume::Sender<CompletionsTask>,
    template: Option<String>,
) -> Result<()> {
    let ctx = Context {
        model,
        model_name,
        maximum_tokens,
        backend_bridge,
        kv_cache_size_pre_task,
        chat_template: template
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