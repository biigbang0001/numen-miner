use numen_kernel::host::Scanner;
use numen_kernel::pipeline::N_FEATURES;
use numen_kernel::work::{EPS_SCALE, QUANT_STEP, derive_seed, hash_buckets, quantize};
use rayon::prelude::*;

use crate::dump::Dump;
use crate::reference;

pub struct Retention {
    pub count: u64,
    pub matched: u64,
    pub kernel_rejected: u64,
    pub canonical_rejected: u64,
    pub attributed: bool,
    /// Per dimension count of nonces whose bucket index differed.
    pub bucket_flips: [u64; 23],
    /// Per dimension worst drift, in bucket widths.
    pub worst_drift: [f64; 23],
}

impl Retention {
    pub fn rate(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.matched as f64 / self.count as f64
    }
}

/// One nonce as the kernel answered it, its work value and, when attributing,
/// its raw features.
type Answer = (u64, Option<[u8; 32]>, Option<[f64; N_FEATURES]>);

/// Grade the CPU kernel, run here.
pub fn measure(pre_hash: &[u8; 32], count: u64, attribute: bool) -> Retention {
    let answers = (0..count)
        .into_par_iter()
        .map_init(Scanner::new, |scanner, nonce| {
            let n = [nonce, 0, 0, 0];
            // Rescanning keeps the hash on the miner's own entry point.
            let features = if attribute { scanner.features(derive_seed(pre_hash, &n)) } else { None };
            (nonce, scanner.work(pre_hash, &n), features)
        })
        .collect();
    grade(pre_hash, answers, attribute)
}

/// Grade a device's dump. Its work values follow from its features the way
/// the finish kernel takes them.
pub fn measure_dump(dump: &Dump, attribute: bool) -> Retention {
    let answers = dump
        .rows
        .iter()
        .map(|&(nonce, features)| (nonce, features.map(|f| hash_buckets(&quantize(&f), &dump.pre_hash)), features))
        .collect();
    grade(&dump.pre_hash, answers, attribute)
}

fn grade(pre_hash: &[u8; 32], answers: Vec<Answer>, attribute: bool) -> Retention {
    let sides: Vec<_> = answers
        .into_par_iter()
        .map(|(nonce, mine, mine_features)| {
            let theirs = reference::work(pre_hash, nonce);
            let their_features = if attribute { reference::features(pre_hash, nonce) } else { None };
            (mine, theirs, mine_features.zip(their_features))
        })
        .collect();

    let mut out = Retention {
        count: sides.len() as u64,
        matched: 0,
        kernel_rejected: 0,
        canonical_rejected: 0,
        attributed: attribute,
        bucket_flips: [0; 23],
        worst_drift: [0.0; 23],
    };

    for (mine, theirs, features) in sides {
        match (&mine, &theirs) {
            (Some(mine_work), Some(their_work)) if mine_work == their_work => out.matched += 1,
            (None, _) => out.kernel_rejected += 1,
            (_, None) => out.canonical_rejected += 1,
            _ => {}
        }

        let Some((mine_features, their_features)) = features else {
            continue;
        };
        let mine_buckets = quantize(&mine_features);
        for d in 0..23 {
            let width = QUANT_STEP[d] * EPS_SCALE;
            let drift = (mine_features[d] - their_features[d]).abs() / width;
            if drift > out.worst_drift[d] {
                out.worst_drift[d] = drift;
            }
            let their_bucket = libm::round(their_features[d] / width) as i64;
            if mine_buckets[d] != their_bucket {
                out.bucket_flips[d] += 1;
            }
        }
    }
    out
}
