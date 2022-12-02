mod piece_record_store;

use crate::dsn::piece_record_store::{AuxRecordStorage, SegmentIndexGetter};
use futures::{Stream, StreamExt};
use sc_client_api::AuxStore;
use sc_consensus_subspace::ArchivedSegmentNotification;
use sc_piece_cache::AuxPieceCache;
use sp_core::traits::SpawnNamed;
use sp_runtime::traits::Block as BlockT;
use std::sync::Arc;
use subspace_core_primitives::{Piece, PieceIndex, PieceIndexHash, PIECES_IN_SEGMENT};
use subspace_networking::libp2p::{identity, Multiaddr};
use subspace_networking::{
    BootstrappedNetworkingParameters, CreationError, CustomRecordStore, MemoryProviderStorage,
    Node, NodeRunner, PieceByHashRequestHandler, PieceByHashResponse, PieceKey, ToMultihash,
};
use tracing::{debug, info, trace, Instrument};

pub type PieceGetter = Arc<dyn (Fn(&PieceIndex) -> Option<Piece>) + Send + Sync + 'static>;

/// DSN configuration parameters.
#[derive(Clone, Debug)]
pub struct DsnConfig {
    /// Where local DSN node will listen for incoming connections.
    pub listen_on: Vec<Multiaddr>,

    /// Bootstrap nodes for DSN.
    pub bootstrap_nodes: Vec<Multiaddr>,

    /// Reserved nodes for DSN.
    pub reserved_peers: Vec<Multiaddr>,

    /// Identity keypair of a node used for authenticated connections.
    pub keypair: identity::Keypair,

    /// Determines whether we allow keeping non-global (private, shared, loopback..) addresses in Kademlia DHT.
    pub allow_non_global_addresses_in_dht: bool,
}

pub(crate) async fn create_dsn_instance<Block, AS>(
    dsn_config: DsnConfig,
    piece_cache: AuxPieceCache<AS>,
    piece_getter: PieceGetter,
    segment_index_getter: SegmentIndexGetter,
) -> Result<
    (
        Node,
        NodeRunner<CustomRecordStore<AuxRecordStorage<AS>, MemoryProviderStorage>>,
    ),
    CreationError,
>
where
    Block: BlockT,
    AS: AuxStore + Sync + Send + 'static,
{
    // TODO: Combine `AuxPieceCache` with `AuxRecordStorage` and remove `PieceCache` abstraction
    let record_storage = AuxRecordStorage::new(piece_cache, segment_index_getter);

    trace!("Subspace networking starting.");

    let networking_config = subspace_networking::Config {
        keypair: dsn_config.keypair,
        listen_on: dsn_config.listen_on,
        allow_non_global_addresses_in_dht: dsn_config.allow_non_global_addresses_in_dht,
        networking_parameters_registry: BootstrappedNetworkingParameters::new(
            dsn_config.bootstrap_nodes,
        )
        .boxed(),
        request_response_protocols: vec![PieceByHashRequestHandler::create(move |req| {
            let result = if let PieceKey::PieceIndex(idx) = req.key {
                piece_getter(&idx)
            } else {
                debug!(key=?req.key, "Incorrect piece request - unsupported key type.");

                None
            };

            Some(PieceByHashResponse { piece: result })
        })],
        record_store: CustomRecordStore::new(record_storage, MemoryProviderStorage::default()),
        ..subspace_networking::Config::with_generated_keypair()
    };

    subspace_networking::create(networking_config).await
}

/// Start an archiver that will listen for archived segments and send it to DSN network using
/// pub-sub protocol.
pub(crate) async fn start_dsn_archiver<Spawner>(
    mut archived_segment_notification_stream: impl Stream<Item = ArchivedSegmentNotification> + Unpin,
    node: Node,
    spawner: Spawner,
) where
    Spawner: SpawnNamed,
{
    trace!("Subspace DSN archiver started.");

    let mut last_published_segment_index: Option<u64> = None;
    while let Some(ArchivedSegmentNotification {
        archived_segment, ..
    }) = archived_segment_notification_stream.next().await
    {
        let segment_index = archived_segment.root_block.segment_index();
        let first_piece_index = segment_index * u64::from(PIECES_IN_SEGMENT);

        info!(%segment_index, "Processing a segment.");

        // skip repeating publication
        if let Some(last_published_segment_index) = last_published_segment_index {
            if last_published_segment_index == segment_index {
                info!(?segment_index, "Archived segment skipped.");
                continue;
            }
        }
        let keys_iter = (first_piece_index..)
            .take(archived_segment.pieces.count())
            .map(|idx| (idx, PieceIndexHash::from_index(idx)))
            .map(|(idx, hash)| (idx, hash.to_multihash()));

        spawner.spawn(
            "segment-publishing",
            Some("subspace-networking"),
            Box::pin({
                let node = node.clone();

                async move {
                    for ((_idx, key), piece) in keys_iter.zip(archived_segment.pieces.as_pieces()) {
                        //TODO: restore announcing after https://github.com/libp2p/rust-libp2p/issues/3048
                        // trace!(?key, ?idx, "Announcing key...");
                        //
                        // let announcing_result = node.start_announcing(key).await;
                        //
                        // trace!(?key, "Announcing result: {:?}", announcing_result);

                        let put_value_result = node.put_value(key, piece.to_vec()).await;

                        trace!(?key, "Put value result: {:?}", put_value_result);

                        //TODO: ensure republication of failed announcements
                    }

                    info!(%segment_index, "Segment processed.");
                }
                .in_current_span()
            }),
        );

        last_published_segment_index = Some(segment_index);
    }
}