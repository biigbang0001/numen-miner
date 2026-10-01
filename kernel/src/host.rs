use crate::cpu::Cpu;
use crate::pipeline::N_FEATURES;
use crate::sculpt::Params;
use crate::work::{self, Nonce};

/// One worker's state. Build it once per thread and reuse it for every nonce.
pub struct Scanner {
    cpu: Cpu,
}

impl Default for Scanner {
    fn default() -> Self {
        Self::new()
    }
}

impl Scanner {
    pub fn new() -> Self {
        Scanner { cpu: Cpu::new() }
    }

    /// Raw features for one seed, before quantization.
    pub fn features(&mut self, seed: [u8; 32]) -> Option<[f64; N_FEATURES]> {
        self.cpu.features(&Params::draw(seed))
    }

    /// Full work value for one nonce.
    pub fn work(&mut self, pre_hash: &[u8; 32], nonce: &Nonce) -> Option<[u8; 32]> {
        let seed = work::derive_seed(pre_hash, nonce);
        let f = self.features(seed)?;
        Some(work::hash_buckets(&work::quantize(&f), pre_hash))
    }
}
