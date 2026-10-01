use crate::intrinsics::{sincos, sqrt};
use core::f64::consts::PI;
use rand_core::{Rng as _, SeedableRng};
use rand_pcg::Pcg64;

pub struct Rng {
    core: Pcg64,
}

impl Rng {
    pub fn new(seed: [u8; 32]) -> Self {
        Rng { core: Pcg64::from_seed(seed) }
    }

    /// f64 in [0, 1) from the top 53 bits, exact in IEEE.
    #[inline(always)]
    pub fn unit(&mut self) -> f64 {
        (self.core.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform index in `0..=i`, as `(unit() * (i + 1)) as usize` picks it.
    ///
    /// The product of the 53 bit draw and `i + 1` is taken exactly in integers.
    /// The f64 product rounds, and the two part only when that rounding
    /// carries it across a whole number, about once in 2^46 draws. A GPU runs
    /// f64 at a sliver of its integer rate, and the permutation shuffle alone
    /// makes 255 of these draws per nonce.
    #[inline(always)]
    pub fn index(&mut self, i: usize) -> usize {
        (((self.core.next_u64() >> 11) as u128 * (i as u128 + 1)) >> 53) as usize
    }

    #[inline(always)]
    pub fn range(&mut self, min: f64, max: f64) -> f64 {
        min + self.unit() * (max - min)
    }

    /// Uniform point on the unit sphere by Archimedes' z method.
    #[inline(always)]
    pub fn unit_vector(&mut self) -> [f64; 3] {
        let z = 2.0 * self.unit() - 1.0;
        let phi = 2.0 * PI * self.unit();
        let r = sqrt(1.0 - z * z);
        let (s, c) = sincos(phi);
        [r * c, r * s, z]
    }

    pub fn next_seed(&mut self) -> [u8; 32] {
        let mut seed = [0u8; 32];
        for word in seed.as_chunks_mut::<8>().0 {
            *word = self.core.next_u64().to_le_bytes();
        }
        seed
    }
}
