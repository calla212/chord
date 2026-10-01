use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as AnyhowContext, Result, anyhow};
use futures::{StreamExt, stream};
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use dynamo_runtime::pipeline::{
    AsyncEngineContext, AsyncEngineContextProvider, Context, ManyOut, Operator, PipelineOperator,
    ResponseStream, ServerStreamingEngine, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;

use crate::protocols::common::llm_backend::LLMEngineOutput;
use crate::protocols::common::preprocessor::PreprocessedRequest;

const CONTROL_KEY: &str = "dynamo_chord_control";
const REQUEST_METADATA_KEY: &str = "dynamo_chord";
const BACKEND_REQUEST_ID_KEY: &str = "backend_request_id";
const BACKEND_REQUEST_ID_SEPARATOR: &str = ":dynamo-chord-epoch:";
const DEFAULT_CONTROL_URL: &str = "http://127.0.0.1:18080";

#[derive(Clone)]
pub struct Chord {
    enabled: bool,
    client: Option<ChordControlClient>,
}

impl Chord {
    pub fn from_env() -> Result<Arc<Self>> {
        let enabled = std::env::var("DYN_CHORD_ENABLE")
            .ok()
            .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"));
        let client = enabled
            .then(ChordControlClient::from_env)
            .transpose()
            .context("create Chord control client")?;
        Ok(Arc::new(Self { enabled, client }))
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn into_operator(
        self: &Arc<Self>,
    ) -> Arc<
        PipelineOperator<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
        >,
    > {
        Operator::into_operator(self)
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for Chord
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        if !self.enabled || request.content().is_probe {
            return next.generate(request).await;
        }

        let client = self
            .client
            .clone()
            .ok_or_else(|| anyhow!("Chord enabled without a control client"))?;
        let (preprocessed, context) = request.transfer(());
        validate_request(&preprocessed)?;
        let engine_context = context.context();
        let response_context = engine_context.clone();
        let state = ChordRequest::new(
            client,
            engine_context,
            context.metadata().clone(),
            preprocessed,
            next,
        )
        .await?;
        let response_stream = stream::unfold(state, |mut state| async move {
            state.next().await.map(|response| (response, state))
        })
        .fuse();
        Ok(ResponseStream::new(
            Box::pin(response_stream),
            response_context,
        ))
    }
}

fn validate_request(request: &PreprocessedRequest) -> Result<()> {
    if request.token_ids.is_empty() {
        return Err(anyhow!("Chord requires a non-empty token prompt"));
    }
    if request.prompt_embeds.is_some() || request.multi_modal_data.is_some() {
        return Err(anyhow!(
            "Chord currently supports token-input text requests only"
        ));
    }
    if request.sampling_options.n.unwrap_or(1) != 1 {
        return Err(anyhow!("Chord requires sampling_options.n == 1"));
    }
    if request.sampling_options.temperature.unwrap_or(0.0).abs() > f32::EPSILON {
        return Err(anyhow!("Chord currently requires greedy temperature=0"));
    }
    if request.sampling_options.guided_decoding.is_some() {
        return Err(anyhow!("Chord does not support guided decoding"));
    }
    Ok(())
}

#[derive(Clone)]
struct ChordControlClient {
    base_url: Arc<str>,
    http: Client,
}

#[derive(Debug, Clone, Deserialize)]
struct DispatchLease {
    request_id: String,
    backend_request_id: String,
    epoch: u64,
    worker_id: u64,
    metadata: Value,
}

#[derive(Serialize)]
struct SubmitRequest<'a> {
    request_id: &'a str,
    prompt_tokens: usize,
}

impl ChordControlClient {
    fn from_env() -> Result<Self> {
        let base_url = std::env::var("DYN_CHORD_FRONTEND_CONTROL_URL")
            .unwrap_or_else(|_| DEFAULT_CONTROL_URL.to_string());
        let parsed = reqwest::Url::parse(&base_url)
            .with_context(|| format!("invalid DYN_CHORD_FRONTEND_CONTROL_URL={base_url:?}"))?;
        let loopback = matches!(parsed.host_str(), Some("127.0.0.1" | "localhost" | "::1"));
        if parsed.scheme() != "http"
            || !loopback
            || parsed.port().is_none()
            || parsed.path() != "/"
            || parsed.query().is_some()
            || parsed.fragment().is_some()
        {
            return Err(anyhow!(
                "Chord Frontend control URL must be an explicit loopback HTTP host:port: {base_url}"
            ));
        }
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .build()?;
        Ok(Self {
            base_url: Arc::from(base_url.trim_end_matches('/')),
            http,
        })
    }

    async fn post<T: Serialize + ?Sized, R: DeserializeOwned>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<R> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        let response = self
            .http
            .post(&url)
            .json(body)
            .send()
            .await
            .with_context(|| format!("Chord control request failed: {url}"))?;
        let status = response.status();
        let bytes = response.bytes().await?;
        if !status.is_success() {
            return Err(anyhow!(
                "Chord control {url} returned {status}: {}",
                String::from_utf8_lossy(&bytes)
            ));
        }
        serde_json::from_slice(&bytes)
            .with_context(|| format!("decode Chord control response from {url}"))
    }

    async fn submit(&self, request_id: &str, prompt_tokens: usize) -> Result<DispatchLease> {
        self.post(
            "submit",
            &SubmitRequest {
                request_id,
                prompt_tokens,
            },
        )
        .await
    }

    async fn waiting_return(&self, control: &Value) -> Result<DispatchLease> {
        self.post("waiting-return", control).await
    }

    async fn dispatch_failed(
        &self,
        request_id: &str,
        lease: &DispatchLease,
        error: &anyhow::Error,
    ) -> Result<DispatchLease> {
        self.post(
            "dispatch-failed",
            &json!({
                "request_id": request_id,
                "epoch": lease.epoch,
                "worker_id": lease.worker_id,
                "error": format!("{error:#}"),
            }),
        )
        .await
    }

    async fn finish(&self, request_id: &str, epoch: u64, reason: &str) -> Result<Value> {
        self.post(
            "finish",
            &json!({"request_id": request_id, "epoch": epoch, "reason": reason}),
        )
        .await
    }

    async fn cancel(&self, request_id: &str) -> Result<Value> {
        self.post("cancel", &json!({"request_id": request_id}))
            .await
    }
}

struct ChordRequest {
    client: ChordControlClient,
    context: Arc<dyn AsyncEngineContext>,
    context_metadata: BTreeMap<String, String>,
    request_id: String,
    request: PreprocessedRequest,
    next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    stream: Option<ManyOut<Annotated<LLMEngineOutput>>>,
    lease: DispatchLease,
    cleaned: bool,
    ended: bool,
}

impl ChordRequest {
    async fn new(
        client: ChordControlClient,
        context: Arc<dyn AsyncEngineContext>,
        context_metadata: BTreeMap<String, String>,
        request: PreprocessedRequest,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<Self> {
        let request_id = context.id().to_string();
        let lease = client.submit(&request_id, request.token_ids.len()).await?;
        let mut state = Self {
            client,
            context,
            context_metadata,
            request_id,
            request,
            next,
            stream: None,
            lease,
            cleaned: false,
            ended: false,
        };
        state.open_stream().await?;
        Ok(state)
    }

    async fn open_stream(&mut self) -> Result<()> {
        loop {
            if self.context.is_stopped() || self.context.is_killed() {
                return Err(anyhow!("Chord request {} was cancelled", self.request_id));
            }
            apply_lease(&mut self.request, &self.request_id, &self.lease)?;
            let child = Context::with_id_and_metadata(
                self.request.clone(),
                self.lease.backend_request_id.clone(),
                self.context_metadata.clone(),
            );
            self.context.link_child(child.context());
            match self.next.generate(child).await {
                Ok(stream) => {
                    self.stream = Some(stream);
                    return Ok(());
                }
                Err(error) => {
                    let error = anyhow!(error).context("open native Dynamo route");
                    tracing::warn!(
                        request_id = %self.request_id,
                        epoch = self.lease.epoch,
                        worker_id = self.lease.worker_id,
                        error = %error,
                        "Chord native dispatch failed; requesting another lease"
                    );
                    self.lease = self
                        .client
                        .dispatch_failed(&self.request_id, &self.lease, &error)
                        .await?;
                }
            }
        }
    }

    async fn next(&mut self) -> Option<Annotated<LLMEngineOutput>> {
        if self.ended {
            return None;
        }
        loop {
            if self.context.is_stopped() || self.context.is_killed() {
                self.cleanup_cancel().await;
                return None;
            }
            let response = match self.stream.as_mut() {
                Some(stream) => stream.next().await,
                None => {
                    self.ended = true;
                    return Some(Annotated::from_error("Chord native stream is missing"));
                }
            };
            let Some(response) = response else {
                self.notify_finish("stream_ended").await;
                self.ended = true;
                return None;
            };

            if let Some(commits) = response.data.as_ref()
                .and_then(|output| output.extra_args.as_ref())
                .and_then(|extra| extra.get("dynamo_migration_committed"))
                .and_then(Value::as_array)
            {
                for commit in commits {
                    let result: Result<Value> = self.client.post("migration-committed", &json!({
                        "backend_request_id": self.lease.backend_request_id,
                        "source_worker_id": commit["source_worker_id"],
                        "target_worker_id": commit["target_worker_id"],
                    })).await;
                    if let Err(error) = result {
                        self.cleanup_cancel().await;
                        return Some(Annotated::from_error(format!(
                            "Chord migration commit failed: {error:#}"
                        )));
                    }
                }
            }

            if let Some(control) = waiting_return_control(&response) {
                match self.client.waiting_return(&control).await {
                    Ok(lease) => {
                        self.lease = lease;
                        self.detach_terminal_stream();
                        if let Err(error) = self.open_stream().await {
                            self.cleanup_cancel().await;
                            return Some(Annotated::from_error(format!(
                                "Chord redispatch failed: {error:#}"
                            )));
                        }
                        continue;
                    }
                    Err(error) => {
                        self.cleanup_cancel().await;
                        return Some(Annotated::from_error(format!(
                            "Chord waiting return failed: {error:#}"
                        )));
                    }
                }
            }

            if response.error.is_some() {
                self.notify_finish("backend_error").await;
                self.detach_terminal_stream();
                self.ended = true;
            } else if let Some(reason) = response
                .data
                .as_ref()
                .and_then(|output| output.finish_reason.as_ref())
            {
                self.notify_finish(&format!("{reason:?}")).await;
                self.detach_terminal_stream();
                self.ended = true;
            }
            return Some(response);
        }
    }

    async fn notify_finish(&mut self, reason: &str) {
        if self.cleaned {
            return;
        }
        match self
            .client
            .finish(&self.request_id, self.lease.epoch, reason)
            .await
        {
            Ok(_) => self.cleaned = true,
            Err(error) => tracing::warn!(
                request_id = %self.request_id,
                error = %error,
                "Chord finish notification failed"
            ),
        }
    }

    fn detach_terminal_stream(&mut self) {
        let Some(mut stream) = self.stream.take() else {
            return;
        };
        let request_id = self.request_id.clone();
        tokio::spawn(async move {
            let mut unexpected_chunks = 0usize;
            while stream.next().await.is_some() {
                unexpected_chunks += 1;
            }
            if unexpected_chunks != 0 {
                tracing::warn!(
                    request_id = %request_id,
                    unexpected_chunks,
                    "Chord received data after a terminal backend response"
                );
            }
        });
    }

    async fn cleanup_cancel(&mut self) {
        if self.cleaned {
            self.ended = true;
            return;
        }
        match self.client.cancel(&self.request_id).await {
            Ok(_) => self.cleaned = true,
            Err(error) => tracing::warn!(
                request_id = %self.request_id,
                error = %error,
                "Chord cancel notification failed"
            ),
        }
        self.ended = true;
    }
}

impl Drop for ChordRequest {
    fn drop(&mut self) {
        if self.cleaned {
            return;
        }
        let client = self.client.clone();
        let request_id = self.request_id.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Err(error) = client.cancel(&request_id).await {
                    tracing::debug!(
                        request_id = %request_id,
                        error = %error,
                        "Chord best-effort drop cancellation failed"
                    );
                }
            });
        }
    }
}

fn apply_lease(
    request: &mut PreprocessedRequest,
    request_id: &str,
    lease: &DispatchLease,
) -> Result<()> {
    if lease.request_id != request_id {
        return Err(anyhow!(
            "Chord lease request mismatch: expected={request_id} got={}",
            lease.request_id
        ));
    }
    if lease.epoch == 0 {
        return Err(anyhow!("Chord lease epoch must be positive"));
    }
    let expected_backend_request_id =
        format!("{request_id}{BACKEND_REQUEST_ID_SEPARATOR}{}", lease.epoch);
    if lease.backend_request_id != expected_backend_request_id {
        return Err(anyhow!(
            "Chord backend request ID mismatch: expected={expected_backend_request_id} got={}",
            lease.backend_request_id
        ));
    }
    let metadata = lease
        .metadata
        .as_object()
        .ok_or_else(|| anyhow!("Chord lease metadata must be a JSON object"))?;
    if metadata.get("request_id").and_then(Value::as_str) != Some(request_id)
        || metadata.get("epoch").and_then(Value::as_u64) != Some(lease.epoch)
        || metadata.get(BACKEND_REQUEST_ID_KEY).and_then(Value::as_str)
            != Some(lease.backend_request_id.as_str())
    {
        return Err(anyhow!(
            "Chord lease metadata identity does not match its lease"
        ));
    }
    request.routing_mut().backend_instance_id = Some(lease.worker_id);
    request.routing_mut().decode_worker_id = None;
    let extra_args = request
        .extra_args
        .get_or_insert_with(|| Value::Object(Map::new()));
    if extra_args.is_null() {
        *extra_args = Value::Object(Map::new());
    }
    let object = extra_args
        .as_object_mut()
        .ok_or_else(|| anyhow!("PreprocessedRequest.extra_args must be a JSON object"))?;
    object.insert(REQUEST_METADATA_KEY.to_string(), lease.metadata.clone());
    Ok(())
}

fn waiting_return_control(response: &Annotated<LLMEngineOutput>) -> Option<Value> {
    let control = response
        .data
        .as_ref()?
        .extra_args
        .as_ref()?
        .as_object()?
        .get(CONTROL_KEY)?;
    (control.get("type")?.as_str()? == "WAITING_RETURN").then(|| control.clone())
}
