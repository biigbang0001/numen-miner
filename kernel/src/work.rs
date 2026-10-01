use crate::pipeline::N_FEATURES;
use sha2::{Digest, Sha256};

/// Domain separation prefix. A parameter change bumps this.
pub const PROTOCOL: &[u8] = b"poscan-v1";

/// Tightens the identity bucket widths to roughly 1e-6 per dimension, filling
/// the work value with full entropy.
pub const EPS_SCALE: f64 = 1e-5;

/// Per dimension bucket widths, before `EPS_SCALE`.
pub const QUANT_STEP: [f64; N_FEATURES] = [
    0.03, 0.03, 0.08,
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
    0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
    0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12,
];

/// A 256 bit nonce as little endian limbs, matching the wire encoding.
pub type Nonce = [u64; 4];

/// Little endian bytes, the form the seed preimage takes.
#[inline(always)]
pub fn nonce_bytes(n: &Nonce) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, limb) in n.iter().enumerate() {
        out[8 * i..8 * i + 8].copy_from_slice(&limb.to_le_bytes());
    }
    out
}

/// Add a small offset, wrapping at 2^256. Each lane derives its own nonce this
/// way from one broadcast base.
#[inline(always)]
pub fn nonce_add(n: &Nonce, delta: u64) -> Nonce {
    let mut out = *n;
    let mut carry = delta;
    for limb in out.iter_mut() {
        let (v, o) = limb.overflowing_add(carry);
        *limb = v;
        carry = o as u64;
        if carry == 0 {
            break;
        }
    }
    out
}

/// Derive the generator seed from the full digest, never a truncation. A capped
/// seed domain is an enumerable one.
pub fn derive_seed(pre_hash: &[u8; 32], nonce: &Nonce) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(PROTOCOL);
    hasher.update([0u8]);
    hasher.update(pre_hash);
    hasher.update(nonce_bytes(nonce));
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&hasher.finalize());
    seed
}

/// Fine quantize the raw features into the integer bucket vector.
#[inline(always)]
pub fn quantize(features: &[f64; N_FEATURES]) -> [i64; N_FEATURES] {
    let mut buckets = [0i64; N_FEATURES];
    for i in 0..N_FEATURES {
        buckets[i] = crate::intrinsics::round(features[i] / (QUANT_STEP[i] * EPS_SCALE)) as i64;
    }
    buckets
}

/// Hash the bucket vector into the work value. The pre hash goes in directly as
/// well as through the seed, so a scanned seed table never outlives one block.
pub fn hash_buckets(buckets: &[i64; N_FEATURES], pre_hash: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(PROTOCOL);
    hasher.update([0u8]);
    hasher.update(pre_hash);
    for &bucket in buckets {
        hasher.update(bucket.to_le_bytes());
    }
    let mut work = [0u8; 32];
    work.copy_from_slice(&hasher.finalize());
    work
}

/// Does this work value clear the target?
///
/// The chain asks whether `work * difficulty` overflows 256 bits. That is the
/// same question as `work <= U256::MAX / difficulty`, and the host precomputes
/// that quotient once per task, which turns the per nonce test into a big
/// endian byte comparison.
#[inline(always)]
pub fn meets(work: &[u8; 32], threshold: &[u8; 32]) -> bool {
    for i in 0..32 {
        if work[i] != threshold[i] {
            return work[i] < threshold[i];
        }
    }
    true
}
