use numen_kernel::pipeline::N_FEATURES;
use std::path::Path;

/// A `numen-miner dump`, one device's raw features nonce by nonce.
pub struct Dump {
    pub pre_hash: [u8; 32],
    /// Each nonce with its features, `None` where the device found the body
    /// unmeasurable.
    pub rows: Vec<(u64, Option<[f64; N_FEATURES]>)>,
}

pub fn read(path: &Path) -> Dump {
    let text = std::fs::read_to_string(path).expect("read the dump");
    let mut lines = text.lines();
    let header = lines.next().and_then(|line| line.strip_prefix("pre_hash ")).expect("dump opens with its pre hash");
    let pre_hash: [u8; 32] =
        hex::decode(header).ok().and_then(|bytes| bytes.try_into().ok()).expect("pre hash is 32 bytes of hex");
    Dump { pre_hash, rows: lines.map(row).collect() }
}

fn row(line: &str) -> (u64, Option<[f64; N_FEATURES]>) {
    let mut fields = line.split(' ');
    let nonce = fields.next().and_then(|s| s.parse().ok()).expect("row opens with its nonce");
    let rest: Vec<&str> = fields.collect();
    if rest == ["-"] {
        return (nonce, None);
    }
    let features: Vec<f64> = rest.iter().map(|s| s.parse().expect("feature is a number")).collect();
    (nonce, Some(features.try_into().expect("row holds every feature")))
}
