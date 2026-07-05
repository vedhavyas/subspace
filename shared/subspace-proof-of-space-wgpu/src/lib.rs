//! Subspace proof of space plotting on the GPU via wgpu (Vulkan/Metal).
//!
//! The shader half reuses abundance's `ab-proof-of-space-gpu` compute kernels and adds subspace's
//! own f7 entry point, which bins s-buckets under subspace's little-endian convention; rust-gpu
//! compiles it to SPIR-V. The host half runs those proofs on the GPU and encodes records with
//! subspace's KZG scheme, so a GPU-plotted sector reads back byte-for-byte the same as the CPU one.

#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(generic_const_exprs)]
#![cfg_attr(not(target_arch = "spirv"), feature(portable_simd))]
#![expect(incomplete_features, reason = "generic_const_exprs")]

#[cfg(not(target_arch = "spirv"))]
mod host;

#[cfg(not(target_arch = "spirv"))]
pub use ab_proof_of_space_gpu::{Backend, Device, DeviceType};
#[cfg(not(target_arch = "spirv"))]
pub use host::WgpuDevice;

/// Name of subspace's f7 entry point in the compiled shader. It has to differ from abundance's own
/// entry points, because abundance's shader is linked into this one and a shared name would let the
/// host bind abundance's raw f7 instead of ours.
#[cfg(not(target_arch = "spirv"))]
pub const F7_ENTRY_POINT: &str = "subspace_find_matches_and_compute_f7";

/// Subspace's compiled shader and its f7 entry point, handed to abundance's host as the f7 override
/// so abundance runs our binning without knowing the convention.
#[cfg(not(target_arch = "spirv"))]
pub fn f7_shader_override() -> (wgpu::ShaderModuleDescriptor<'static>, &'static str) {
    let module = wgpu::ShaderModuleDescriptor {
        label: Some("subspace-proof-of-space-wgpu"),
        source: wgpu::util::make_spirv(include_bytes!(env!("SHADER_PATH"))),
    };
    (module, F7_ENTRY_POINT)
}

#[cfg(target_arch = "spirv")]
use ab_proof_of_space_gpu::shader::constants::{
    MAX_BUCKET_SIZE, NUM_BUCKETS, NUM_MATCH_BUCKETS, NUM_S_BUCKETS, REDUCED_MATCHES_COUNT,
};
#[cfg(target_arch = "spirv")]
use ab_proof_of_space_gpu::shader::find_matches_and_compute_f7::{
    FindMatchesAndComputeF7Shared, NUM_ELEMENTS_PER_S_BUCKET, ProofTargets,
    find_matches_and_compute_f7_impl,
};
#[cfg(target_arch = "spirv")]
use ab_proof_of_space_gpu::shader::sbucket::SBucketMap;
#[cfg(target_arch = "spirv")]
use ab_proof_of_space_gpu::shader::types::{Match, Metadata, PositionR};
#[cfg(target_arch = "spirv")]
use core::mem::MaybeUninit;
#[cfg(target_arch = "spirv")]
use spirv_std::glam::UVec3;
#[cfg(target_arch = "spirv")]
use spirv_std::spirv;

/// Subspace's little-endian s-bucket convention: a proof for s-bucket `cs` has
/// `first_k_bits == cs_lo << (K - 8) | cs_hi << (K - 16)`; this inverts it, discarding entries whose
/// low `K - 16` bits are set. This is the only place the convention lives on the GPU side.
#[cfg(target_arch = "spirv")]
pub struct LittleEndianMap;

#[cfg(target_arch = "spirv")]
impl SBucketMap for LittleEndianMap {
    #[inline(always)]
    fn map(first_k_bits: u32) -> u32 {
        // K == PosProof::K
        const K: u32 = 20;
        let low_bits = K - 16;
        if first_k_bits & ((1u32 << low_bits) - 1) != 0 {
            return u32::MAX;
        }
        let cs_lo = (first_k_bits >> (K - 8)) & 0xff;
        let cs_hi = (first_k_bits >> low_bits) & 0xff;
        cs_lo | (cs_hi << 8)
    }
}

/// Subspace's f7 entry point: reuses abundance's implementation across the crate boundary.
///
/// # Safety
/// Must be dispatched from [`MAX_BUCKET_SIZE`]/2 == 256 threads with valid parent buckets.
#[cfg(target_arch = "spirv")]
#[spirv(compute(
    threads(256),
    entry_point_name = "subspace_find_matches_and_compute_f7"
))]
#[expect(clippy::too_many_arguments, reason = "Vulkan bindings")]
pub unsafe fn subspace_find_matches_and_compute_f7(
    #[spirv(local_invocation_id)] local_invocation_id: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] parent_buckets: &[[PositionR; MAX_BUCKET_SIZE];
         NUM_BUCKETS],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    parent_metadatas: &[Metadata; REDUCED_MATCHES_COUNT * NUM_MATCH_BUCKETS],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    table_6_proof_targets_sizes: &mut [u32; NUM_S_BUCKETS],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    table_6_proof_targets: &mut [[MaybeUninit<ProofTargets>; NUM_ELEMENTS_PER_S_BUCKET];
             NUM_S_BUCKETS],
    #[spirv(workgroup)] matches: &mut [MaybeUninit<Match>; MAX_BUCKET_SIZE],
    #[spirv(workgroup)] shared: &mut FindMatchesAndComputeF7Shared,
) {
    // SAFETY: contract forwarded to abundance's implementation, bound to subspace's binning.
    unsafe {
        find_matches_and_compute_f7_impl::<LittleEndianMap>(
            local_invocation_id,
            workgroup_id,
            parent_buckets,
            parent_metadatas,
            table_6_proof_targets_sizes,
            table_6_proof_targets,
            matches,
            shared,
        );
    }
}
