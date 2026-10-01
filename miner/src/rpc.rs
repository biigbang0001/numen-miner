use anyhow::{Context, Result, anyhow};
use codec::Encode;
use jsonrpsee::core::client::{ClientT, Error as RpcError, Subscription, SubscriptionClientT};
use jsonrpsee::rpc_params;
use jsonrpsee::ws_client::{PingConfig, WsClient, WsClientBuilder};
use primitive_types::{H256, U256};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Handle;

use crate::engine::{Share, Shared, Submit, random_limb};

/// Pause between task polls over HTTP. A rig still mining the old pre-hash
/// after a block wastes its work, and DiffPool moves on every few seconds.
const POLL: Duration = Duration::from_millis(500);

/// Longest wait for an HTTP answer.
const TIMEOUT: Duration = Duration::from_secs(10);

/// One unit of work as the node describes it.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MiningTask {
    pub pre_hash: H256,
    pub difficulty: U256,
    pub miner: String,
}

#[derive(Deserialize)]
struct Head {
    number: U256,
}

/// A node's mining RPC over a websocket, which pushes every new task.
pub struct Node {
    url: String,
    client: Arc<WsClient>,
    tasks: Subscription<MiningTask>,
    heads: Subscription<Head>,
    extranonce: u64,
}

impl Node {
    pub async fn open(url: &str) -> Result<Node> {
        let client = WsClientBuilder::new().enable_ws_ping(PingConfig::new()).build(url).await?;
        let tasks = subscribe(&client, "mining_subscribeTask", "mining_unsubscribeTask").await?;
        let heads = subscribe(&client, "chain_subscribeNewHeads", "chain_unsubscribeNewHeads").await?;
        Ok(Node { url: url.to_string(), client: Arc::new(client), tasks, heads, extranonce: random_limb() })
    }

    /// Mine the node's tasks until the connection drops.
    pub async fn run(mut self, shared: &Arc<Shared>) -> anyhow::Error {
        let client = self.client.clone();
        shared.attach(submitter(move |pre_hash, seal| submit_seal(client.clone(), pre_hash, seal)));
        let url = &self.url;
        let mut miner = String::new();
        loop {
            tokio::select! {
                // Polled first, so a dropped link reports its cause, not the
                // subscriptions it closed.
                biased;
                error = self.client.on_disconnect() => return lost(error),
                next = self.tasks.next() => match next {
                    None => return anyhow!("task subscription closed"),
                    Some(Ok(task)) => hand_out(&task, &mut miner, shared, self.extranonce),
                    Some(Err(e)) => log::warn!("{url}: malformed task: {e}"),
                },
                next = self.heads.next() => match next {
                    None => return anyhow!("header subscription closed"),
                    Some(Ok(head)) => log::info!("new block #{}", head.number),
                    Some(Err(e)) => log::warn!("{url}: malformed header: {e}"),
                },
            }
        }
    }
}

/// Why a node link dropped. The client wraps every cause in a call for a
/// restart, which the upstream makes on its own.
fn lost(error: RpcError) -> anyhow::Error {
    match error {
        RpcError::RestartNeeded(cause) => anyhow!("connection lost: {cause}"),
        error => anyhow!(error).context("connection lost"),
    }
}

async fn subscribe<T: DeserializeOwned>(client: &WsClient, method: &str, unsubscribe: &str) -> Result<Subscription<T>> {
    client.subscribe(method, rpc_params![], unsubscribe).await.with_context(|| method.to_string())
}

async fn submit_seal(client: Arc<WsClient>, pre_hash: H256, seal: String) -> Result<bool> {
    client.request("mining_submitSeal", rpc_params![pre_hash, seal]).await.context("mining_submitSeal")
}

/// The node's mining RPC over HTTP, polled. DiffPool serves it this way, with
/// the wallet and worker in the path.
#[derive(Clone)]
pub struct Http {
    url: String,
    client: reqwest::Client,
    extranonce: u64,
}

impl Http {
    pub async fn open(url: &str) -> Result<Http> {
        let client = reqwest::Client::builder().timeout(TIMEOUT).build()?;
        let http = Http { url: url.to_string(), client, extranonce: random_limb() };
        http.task().await?;
        Ok(http)
    }

    /// Mine the polled tasks until a poll fails.
    pub async fn run(self, shared: &Arc<Shared>) -> anyhow::Error {
        let http = self.clone();
        shared.attach(submitter(move |pre_hash, seal| {
            let http = http.clone();
            async move { http.call("mining_submitSeal", json!([pre_hash, seal])).await }
        }));
        self.poll(shared).await
    }

    async fn poll(&self, shared: &Shared) -> anyhow::Error {
        let mut miner = String::new();
        let mut current = None;
        loop {
            let task = match self.task().await {
                Ok(task) => task,
                Err(e) => return e,
            };
            // Most polls return the task the last one did.
            if current != Some((task.pre_hash, task.difficulty)) {
                current = Some((task.pre_hash, task.difficulty));
                hand_out(&task, &mut miner, shared, self.extranonce);
            }
            tokio::time::sleep(POLL).await;
        }
    }

    async fn task(&self) -> Result<MiningTask> {
        self.call("mining_getTask", json!([])).await
    }

    async fn call<T: DeserializeOwned>(&self, method: &str, params: Value) -> Result<T> {
        let request = json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params});
        let reply: Reply<T> = async {
            let response = self.client.post(&self.url).json(&request).send().await?;
            response.error_for_status()?.json().await
        }
        .await
        .with_context(|| method.to_string())?;
        reply.result.with_context(|| format!("{method}: {}", reply.error.unwrap_or_default()))
    }
}

/// A JSON-RPC answer, which holds a result or an error.
#[derive(Deserialize)]
struct Reply<T> {
    result: Option<T>,
    error: Option<Value>,
}

/// Hand a task to the miners, naming the reward address when it changes.
fn hand_out(task: &MiningTask, miner: &mut String, shared: &Shared, extranonce: u64) {
    if task.miner != *miner {
        log::info!("rewards go to {}", task.miner);
        miner.clone_from(&task.miner);
    }
    log::info!("new task {}, difficulty {}", task.pre_hash, task.difficulty);
    shared.set_task(task.pre_hash.0, task.difficulty, extranonce);
}

/// Submit each seal through `request` on the link's runtime and count the
/// node's verdict. Workers are plain threads, so they hand the call over.
fn submitter<F, R>(request: F) -> Submit
where
    F: Fn(H256, String) -> R + Send + Sync + 'static,
    R: Future<Output = Result<bool>> + Send + 'static,
{
    let runtime = Handle::current();
    Box::new(move |shared, share| {
        // SCALE lays out a tuple exactly like the chain's Seal struct.
        let seal = (U256(share.nonce), H256(share.work)).encode();
        let reply = request(H256(share.pre_hash), format!("0x{}", hex::encode(seal)));
        runtime.spawn(verdict(shared.clone(), share, reply));
    })
}

async fn verdict(shared: Arc<Shared>, share: Share, reply: impl Future<Output = Result<bool>>) {
    let sent = Instant::now();
    match reply.await {
        Ok(accepted) => shared.verdict(&share.meter, sent, accepted, None),
        Err(e) => log::warn!("{} submit failed: {e:#}", share.meter.name),
    }
}
