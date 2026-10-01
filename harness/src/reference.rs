use numen_kernel::work::derive_seed;
use poscan::{Compute, SUBDIVISIONS, TARGET_SAMPLES};
use primitive_types::{H256, U256};

/// Work value straight from the chain crate.
pub fn work(pre_hash: &[u8; 32], nonce: u64) -> Option<[u8; 32]> {
    let compute = Compute { pre_hash: H256::from(*pre_hash), nonce: U256::from(nonce) };
    compute.work().map(|h| h.0)
}

/// Raw features from the canonical generator and scan, for per dimension
/// attribution. Seed derivation is plain SHA-256 and is not under test, so it
/// is shared with the kernel side.
pub fn features(pre_hash: &[u8; 32], nonce: u64) -> Option<[f64; 23]> {
    let seed = derive_seed(pre_hash, &[nonce, 0, 0, 0]);
    let mesh = obj_asteroid::asteroid(seed, SUBDIVISIONS);
    spectral3d::scan(mesh, TARGET_SAMPLES).ok()
}
