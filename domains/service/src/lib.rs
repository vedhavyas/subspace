//! Service and ServiceFactory implementation. Specialized wrapper over substrate service.

mod core_domain;
mod rpc;
mod system_domain;

pub use self::core_domain::{new_full as new_full_core, NewFull as NewFullCore};
pub use self::system_domain::{new_full, NewFull};
use domain_client_consensus_relay_chain::notification::SubspaceNotificationStream;
use domain_runtime_primitives::opaque::Block;
use domain_runtime_primitives::RelayerId;
use sc_client_api::StateBackendFor;
use sc_executor::{NativeElseWasmExecutor, NativeExecutionDispatch};
use sc_service::{
    Configuration as ServiceConfiguration, PartialComponents, TFullBackend, TFullClient,
};
use sc_telemetry::{Telemetry, TelemetryWorker, TelemetryWorkerHandle};
use sp_api::{ApiExt, ConstructRuntimeApi, NumberFor};
use sp_transaction_pool::runtime_api::TaggedTransactionQueue;
use std::sync::Arc;

/// Domain full client.
pub type FullClient<RuntimeApi, ExecutorDispatch> =
    TFullClient<Block, RuntimeApi, NativeElseWasmExecutor<ExecutorDispatch>>;

pub type FullBackend = sc_service::TFullBackend<Block>;

pub type FullPool<RuntimeApi, ExecutorDispatch> = sc_transaction_pool::BasicPool<
    sc_transaction_pool::FullChainApi<FullClient<RuntimeApi, ExecutorDispatch>, Block>,
    Block,
>;

/// Secondary chain configuration.
pub struct Configuration {
    service_config: ServiceConfiguration,
    maybe_relayer_id: Option<RelayerId>,
}

impl Configuration {
    pub fn new(service_config: ServiceConfiguration, maybe_relayer_id: Option<RelayerId>) -> Self {
        Configuration {
            service_config,
            maybe_relayer_id,
        }
    }
}

/// Starts a `ServiceBuilder` for a full service.
///
/// Use this macro if you don't actually need the full service, but just the builder in order to
/// be able to perform chain operations.
#[allow(clippy::type_complexity)]
fn new_partial<RuntimeApi, Executor>(
    config: &ServiceConfiguration,
) -> Result<
    PartialComponents<
        FullClient<RuntimeApi, Executor>,
        TFullBackend<Block>,
        (),
        sc_consensus::DefaultImportQueue<Block, FullClient<RuntimeApi, Executor>>,
        sc_transaction_pool::FullPool<Block, FullClient<RuntimeApi, Executor>>,
        (
            Option<Telemetry>,
            Option<TelemetryWorkerHandle>,
            NativeElseWasmExecutor<Executor>,
            SubspaceNotificationStream<NumberFor<Block>>,
        ),
    >,
    sc_service::Error,
>
where
    RuntimeApi:
        ConstructRuntimeApi<Block, FullClient<RuntimeApi, Executor>> + Send + Sync + 'static,
    RuntimeApi::RuntimeApi: TaggedTransactionQueue<Block>
        + ApiExt<Block, StateBackend = StateBackendFor<TFullBackend<Block>, Block>>,
    Executor: NativeExecutionDispatch + 'static,
{
    let telemetry = config
        .telemetry_endpoints
        .clone()
        .filter(|x| !x.is_empty())
        .map(|endpoints| -> Result<_, sc_telemetry::Error> {
            let worker = TelemetryWorker::new(16)?;
            let telemetry = worker.handle().new_telemetry(endpoints);
            Ok((worker, telemetry))
        })
        .transpose()?;

    let executor = NativeElseWasmExecutor::new(
        config.wasm_method,
        config.default_heap_pages,
        config.max_runtime_instances,
        config.runtime_cache_size,
    );

    let (client, backend, keystore_container, task_manager) = sc_service::new_full_parts(
        config,
        telemetry.as_ref().map(|(_, telemetry)| telemetry.handle()),
        executor.clone(),
    )?;
    let client = Arc::new(client);

    let telemetry_worker_handle = telemetry.as_ref().map(|(worker, _)| worker.handle());

    let telemetry = telemetry.map(|(worker, telemetry)| {
        task_manager
            .spawn_handle()
            .spawn("telemetry", None, worker.run());
        telemetry
    });

    let transaction_pool = sc_transaction_pool::BasicPool::new_full(
        config.transaction_pool.clone(),
        config.role.is_authority().into(),
        config.prometheus_registry(),
        task_manager.spawn_essential_handle(),
        client.clone(),
    );

    let (import_queue, import_block_notification_stream) =
        domain_client_consensus_relay_chain::import_queue(
            client.clone(),
            &task_manager.spawn_essential_handle(),
            config.prometheus_registry(),
        )?;

    let params = PartialComponents {
        backend,
        client,
        import_queue,
        keystore_container,
        task_manager,
        transaction_pool,
        select_chain: (),
        other: (
            telemetry,
            telemetry_worker_handle,
            executor,
            import_block_notification_stream,
        ),
    };

    Ok(params)
}