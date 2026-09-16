//! Tokio runtime helpers.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

pub use tokio::task::JoinError;

/// A join handle whose output is always `Result<T, JoinError>`.
pub struct JoinHandle<T>(tokio::task::JoinHandle<T>);

impl<T> Future for JoinHandle<T> {
    type Output = Result<T, JoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.0).poll(cx)
    }
}

/// Spawn a `Send + 'static` future onto Tokio.
pub fn spawn<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    JoinHandle(tokio::spawn(future))
}

/// The Tokio runtime a listener is serving on.
///
/// Nacelle parallelises connection handling by spawning onto the ambient
/// runtime, so a listener started on a current-thread runtime is capped to a
/// single core no matter how many cores the host has. This type makes that
/// property observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NacelleRuntimeTopology {
    flavor: &'static str,
    workers: usize,
}

impl NacelleRuntimeTopology {
    /// Describe the runtime the caller is executing on, if any.
    pub fn current() -> Option<Self> {
        let handle = tokio::runtime::Handle::try_current().ok()?;
        let flavor = match handle.runtime_flavor() {
            tokio::runtime::RuntimeFlavor::CurrentThread => "current_thread",
            tokio::runtime::RuntimeFlavor::MultiThread => "multi_thread",
            _ => "unknown",
        };
        Some(Self {
            flavor,
            workers: handle.metrics().num_workers(),
        })
    }

    /// The runtime flavor as a stable, low-cardinality label.
    pub fn flavor(&self) -> &'static str {
        self.flavor
    }

    /// The number of worker threads driving the runtime.
    pub fn workers(&self) -> usize {
        self.workers
    }

    /// Whether the runtime can only ever use one core.
    pub fn is_single_worker(&self) -> bool {
        self.workers <= 1
    }
}

static TOPOLOGY_REPORTED: AtomicBool = AtomicBool::new(false);

/// Report the runtime topology once per process when a listener starts.
///
/// Emits one `INFO` line describing the runtime, a `WARN` when the runtime is
/// capped to a single worker, and publishes the worker count as the
/// `server.runtime.workers` gauge so single-core misconfiguration is visible in
/// dashboards rather than only in a profiler.
pub fn report_runtime_topology(transport: &'static str) {
    let Some(topology) = NacelleRuntimeTopology::current() else {
        return;
    };
    metrics::gauge!("server.runtime.workers").set(topology.workers as f64);
    if TOPOLOGY_REPORTED.swap(true, Ordering::Relaxed) {
        return;
    }
    tracing::info!(
        target: "nacelle",
        transport,
        runtime = topology.flavor,
        workers = topology.workers,
        "listener started"
    );
    if topology.is_single_worker() {
        tracing::warn!(
            target: "nacelle",
            transport,
            runtime = topology.flavor,
            workers = topology.workers,
            "runtime has a single worker; throughput is capped to one core"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn current_thread_runtime_reports_single_worker() {
        let topology = NacelleRuntimeTopology::current().expect("runtime");
        assert_eq!(topology.flavor(), "current_thread");
        assert!(topology.is_single_worker());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn multi_thread_runtime_reports_worker_count() {
        let topology = NacelleRuntimeTopology::current().expect("runtime");
        assert_eq!(topology.flavor(), "multi_thread");
        assert_eq!(topology.workers(), 2);
        assert!(!topology.is_single_worker());
    }

    #[test]
    fn topology_is_absent_outside_a_runtime() {
        assert!(NacelleRuntimeTopology::current().is_none());
    }
}
