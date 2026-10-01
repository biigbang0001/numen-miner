use crate::intrinsics::{acos, cbrt, cos, sqrt};
use crate::real::Real;

pub const N_FEATURES: usize = 23;
pub const TARGET_SAMPLES: usize = 4096;
pub const HIST_BINS: usize = 8;
pub(crate) const HIST_STEP: f64 = 0.25;

/// Kahan compensated sum, one per lane.
///
/// A plain f32 running total over thousands of faces loses the low bits of
/// every late term, which is fatal at these bucket widths.
#[derive(Clone, Copy)]
pub(crate) struct Sum<T = f32> {
    s: T,
    c: T,
}

impl<T: Real> Sum<T> {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Sum { s: T::splat(0.0), c: T::splat(0.0) }
    }

    #[inline(always)]
    pub(crate) fn add(&mut self, x: T) {
        let y = x - self.c;
        let t = self.s + y;
        self.c = (t - self.s) - y;
        self.s = t;
    }

    /// The total over every lane.
    #[inline(always)]
    pub(crate) fn get(self) -> f64 {
        self.s.sum_f64() - self.c.sum_f64()
    }
}

/// Face sums of area, volume, and the first and second moments about the
/// origin. `volume` is six times the volume, `first` 24 times and `second` 120
/// times the moments, so those factors multiply in once rather than round into
/// every face.
pub(crate) struct Body<T = f32> {
    area: Sum<T>,
    volume: Sum<T>,
    first: [Sum<T>; 3],
    second: [Sum<T>; 6],
}

impl<T: Real> Body<T> {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Body { area: Sum::zero(), volume: Sum::zero(), first: [Sum::zero(); 3], second: [Sum::zero(); 6] }
    }

    /// Add faces by the divergence theorem, one per lane, and return their
    /// areas.
    #[inline(always)]
    pub(crate) fn add(&mut self, a: [T; 3], b: [T; 3], c: [T; 3]) -> T {
        let n = cross(sub(b, a), sub(c, a));
        let area = T::splat(0.5) * dot(n, n).sqrt();
        self.area.add(area);

        let v = dot(a, n);
        self.volume.add(v);
        let s = [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]];
        // Indexed, since the GPU compiler leaves a zipped pair of iterators
        // rolled and the sums in local memory.
        #[allow(clippy::needless_range_loop)]
        for k in 0..3 {
            self.first[k].add(v * s[k]);
        }
        let mut idx = 0;
        for i in 0..3 {
            for j in i..3 {
                self.second[idx].add(v * (a[i] * a[j] + b[i] * b[j] + c[i] * c[j] + s[i] * s[j]));
                idx += 1;
            }
        }
        area
    }

    /// Area, volume, first and second moments, in that order.
    #[inline(always)]
    pub(crate) fn get(&self) -> [f64; 11] {
        let mut out = [0.0; 11];
        out[0] = self.area.get();
        out[1] = self.volume.get();
        for k in 0..3 {
            out[2 + k] = self.first[k].get();
        }
        for k in 0..6 {
            out[5 + k] = self.second[k].get();
        }
        out
    }
}

/// The body's centroid, from the face sums, which the samples are taken
/// relative to.
#[inline(always)]
pub(crate) fn centroid(body: &[f64; 11]) -> [f32; 3] {
    let vol = body[1] / 6.0;
    core::array::from_fn(|k| (body[2 + k] / 24.0 / vol) as f32)
}

/// Unit volume and the principal axes, from the face sums.
pub(crate) struct Frame {
    pub(crate) scale: f64,
    pub(crate) eigvals: [f64; 3],
}

impl Frame {
    pub(crate) fn new(body: &[f64; 11]) -> Option<Frame> {
        let vol = body[1] / 6.0;
        if !vol.is_finite() || vol.abs() < 1e-12 {
            return None;
        }

        let c: [f64; 3] = core::array::from_fn(|k| body[2 + k] / 24.0 / vol);
        let mut cov = [[0.0f64; 3]; 3];
        let mut idx = 0;
        for i in 0..3 {
            for j in i..3 {
                cov[i][j] = body[5 + idx] / 120.0 / vol - c[i] * c[j];
                cov[j][i] = cov[i][j];
                idx += 1;
            }
        }
        let eigvals = eigvals_sym3(cov);
        if eigvals.iter().any(|&x| !x.is_finite() || x <= 0.0) {
            return None;
        }
        Some(Frame { scale: 1.0 / cbrt(vol.abs()), eigvals })
    }
}

/// The feature vector from the finished sums. `sh(i)` yields harmonic `i`
/// summed with weight and with weight times radius, `hist(j)` bin `j`.
#[inline(always)]
pub(crate) fn assemble(
    frame: &Frame,
    w_total: f64,
    rbar: f64,
    mut sh: impl FnMut(usize) -> (f64, f64),
    mut hist: impl FnMut(usize) -> f64,
) -> [f64; N_FEATURES] {
    let ev = frame.eigvals;
    let mut f = [0.0f64; N_FEATURES];
    f[0] = ev[1] / ev[0];
    f[1] = ev[2] / ev[0];
    f[2] = rbar * frame.scale;
    let mut idx = 1;
    for l in 1..=6usize {
        let mut sa = 0.0;
        let mut sb = 0.0;
        for _ in 0..(2 * l + 1) {
            let (a, b) = sh(idx);
            let (x, y) = (SH_NORM[idx] * a, SH_NORM[idx] * b);
            sa += x * x;
            sb += y * y;
            idx += 1;
        }
        f[2 + l] = sqrt(sa) / w_total;
        f[8 + l] = sqrt(sb) / (w_total * rbar);
    }
    for j in 0..HIST_BINS {
        f[15 + j] = hist(j) / w_total;
    }
    f
}

#[inline(always)]
pub(crate) fn sub<T: Real>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
pub(crate) fn cross<T: Real>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
pub(crate) fn dot<T: Real>(a: [T; 3], b: [T; 3]) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Eigenvalues of a symmetric 3x3, descending, by the closed form trig
/// solution. The final sort closes a roundoff order violation near degeneracy
/// that would otherwise swap two feature dimensions between scans.
fn eigvals_sym3(a: [[f64; 3]; 3]) -> [f64; 3] {
    let p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    let p2 = (a[0][0] - q) * (a[0][0] - q)
        + (a[1][1] - q) * (a[1][1] - q)
        + (a[2][2] - q) * (a[2][2] - q)
        + 2.0 * p1;
    if p2 <= 1e-300 {
        return [q, q, q];
    }
    let p = sqrt(p2 / 6.0);
    let mut b = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = (a[i][j] - if i == j { q } else { 0.0 }) / p;
        }
    }
    let detb = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
    let r = (detb / 2.0).clamp(-1.0, 1.0);
    let phi = acos(r) / 3.0;
    let e1 = q + 2.0 * p * cos(phi);
    let e3 = q + 2.0 * p * cos(phi + 2.0 * core::f64::consts::PI / 3.0);
    let e2 = 3.0 * q - e1 - e3;
    let mut ev = [e1, e2, e3];
    ev.sort_unstable_by(|x, y| y.partial_cmp(x).unwrap_or(core::cmp::Ordering::Equal));
    ev
}

/// Prefactors of the 49 real spherical harmonics through l equals 6.
/// `real_sh6` leaves them out, and each finished sum takes its own once.
/// Baked from `sh_norm`, which a test keeps in step, so no GPU spends 49
/// f64 square roots per nonce rebuilding them.
const SH_NORM: [f64; 49] = [
    f64::from_bits(0x3fd20dd750429b6d), // 0 2.8209479177387814e-1
    f64::from_bits(0x3fdf45437857749a), // 1 4.886025119029199e-1
    f64::from_bits(0x3fdf45437857749a), // 2 4.886025119029199e-1
    f64::from_bits(0x3fdf45437857749a), // 3 4.886025119029199e-1
    f64::from_bits(0x3ff17b14102b0694), // 4 1.0925484305920792e0
    f64::from_bits(0x3ff17b14102b0694), // 5 1.0925484305920792e0
    f64::from_bits(0x3fd42f601a8c679b), // 6 3.1539156525252005e-1
    f64::from_bits(0x3ff17b14102b0694), // 7 1.0925484305920792e0
    f64::from_bits(0x3fe17b14102b0694), // 8 5.462742152960396e-1
    f64::from_bits(0x3fe2e1a3183e613b), // 9 5.900435899266435e-1
    f64::from_bits(0x40071ff8e45cad2d), // 10 2.890611442640554e0
    f64::from_bits(0x3fdd403d065e56ea), // 11 4.570457994644658e-1
    f64::from_bits(0x3fd7e21efc0768b0), // 12 3.731763325901154e-1
    f64::from_bits(0x3fdd403d065e56ea), // 13 4.570457994644658e-1
    f64::from_bits(0x3ff71ff8e45cad2d), // 14 1.445305721320277e0
    f64::from_bits(0x3fe2e1a3183e613b), // 15 5.900435899266435e-1
    f64::from_bits(0x400406d8aa0d83a4), // 16 2.5033429417967046e0
    f64::from_bits(0x3ffc5274a45d91d8), // 17 1.7701307697799304e0
    f64::from_bits(0x3fee471027d29b68), // 18 9.461746957575601e-1
    f64::from_bits(0x3fe568d44c02929c), // 19 6.690465435572892e-1
    f64::from_bits(0x3fbb14c2f863e924), // 20 1.0578554691520431e-1
    f64::from_bits(0x3fe568d44c02929c), // 21 6.690465435572892e-1
    f64::from_bits(0x3fde471027d29b68), // 22 4.7308734787878004e-1
    f64::from_bits(0x3ffc5274a45d91d8), // 23 1.7701307697799304e0
    f64::from_bits(0x3fe406d8aa0d83a4), // 24 6.258357354491761e-1
    f64::from_bits(0x3f46c2982f1c5950), // 25 6.945841871324552e-4
    f64::from_bits(0x3f61fe53cf3c8d1f), // 26 2.196468058075176e-3
    f64::from_bits(0x3f5315bf7c58bfb5), // 27 1.1648530938934535e-3
    f64::from_bits(0x3f775fcb77b03183), // 28 5.706591410682529e-3
    f64::from_bits(0x3f9eebd103a87f94), // 29 3.019644341304646e-2
    f64::from_bits(0x3fbdf074d231b11e), // 30 1.169503224534236e-1
    f64::from_bits(0x3f9eebd103a87f94), // 31 3.019644341304646e-2
    f64::from_bits(0x3f775fcb77b03183), // 32 5.706591410682529e-3
    f64::from_bits(0x3f5315bf7c58bfb5), // 33 1.1648530938934535e-3
    f64::from_bits(0x3f61fe53cf3c8d1f), // 34 2.196468058075176e-3
    f64::from_bits(0x3f46c2982f1c5950), // 35 6.945841871324552e-4
    f64::from_bits(0x3f113a8dd54a85ac), // 36 6.57223766418388e-5
    f64::from_bits(0x3f2dd74dbdf188ca), // 37 2.2766899107568565e-4
    f64::from_bits(0x3f117ef054f9133f), // 38 6.674138898528097e-5
    f64::from_bits(0x3f37f50a1ebb6f6f), // 39 3.655576426646522e-4
    f64::from_bits(0x3f61f7c7970c9394), // 40 2.1933458559879136e-3
    f64::from_bits(0x3f8c68e3e4f24b29), // 41 1.3871937202826938e-2
    f64::from_bits(0x3fb046123cc5b3b1), // 42 6.356920226762842e-2
    f64::from_bits(0x3f8c68e3e4f24b29), // 43 1.3871937202826938e-2
    f64::from_bits(0x3f61f7c7970c9394), // 44 2.1933458559879136e-3
    f64::from_bits(0x3f37f50a1ebb6f6f), // 45 3.655576426646522e-4
    f64::from_bits(0x3f117ef054f9133f), // 46 6.674138898528097e-5
    f64::from_bits(0x3f2dd74dbdf188ca), // 47 2.2766899107568565e-4
    f64::from_bits(0x3f113a8dd54a85ac), // 48 6.57223766418388e-5
];

#[cfg(test)]
fn sh_norm() -> [f64; 49] {
    let pi = core::f64::consts::PI;
    let s2 = core::f64::consts::SQRT_2;
    let k5 = |ratio: f64| sqrt(11.0 / (4.0 * pi) * ratio);
    let k6 = |ratio: f64| sqrt(13.0 / (4.0 * pi) * ratio);
    let c1 = sqrt(3.0 / (4.0 * pi));
    let c2a = 0.5 * sqrt(15.0 / pi);
    let c3a = 0.25 * sqrt(35.0 / (2.0 * pi));
    let c3b = 0.25 * sqrt(21.0 / (2.0 * pi));
    let c4a = 0.75 * sqrt(35.0 / (2.0 * pi));
    let c4b = 0.75 * sqrt(5.0 / (2.0 * pi));

    let mut k = [0.0f64; 49];
    k[0] = 0.5 * sqrt(1.0 / pi);
    k[1] = c1;
    k[2] = c1;
    k[3] = c1;
    k[4] = c2a;
    k[5] = c2a;
    k[6] = 0.25 * sqrt(5.0 / pi);
    k[7] = c2a;
    k[8] = 0.25 * sqrt(15.0 / pi);
    k[9] = c3a;
    k[10] = 0.5 * sqrt(105.0 / pi);
    k[11] = c3b;
    k[12] = 0.25 * sqrt(7.0 / pi);
    k[13] = c3b;
    k[14] = 0.25 * sqrt(105.0 / pi);
    k[15] = c3a;
    k[16] = 0.75 * sqrt(35.0 / pi);
    k[17] = c4a;
    k[18] = 0.75 * sqrt(5.0 / pi);
    k[19] = c4b;
    k[20] = (3.0 / 16.0) * sqrt(1.0 / pi);
    k[21] = c4b;
    k[22] = (3.0 / 8.0) * sqrt(5.0 / pi);
    k[23] = c4a;
    k[24] = (3.0 / 16.0) * sqrt(35.0 / pi);

    let r5 = [1.0, 1.0 / 30.0, 1.0 / 840.0, 1.0 / 20160.0, 1.0 / 362880.0, 1.0 / 3628800.0];
    k[30] = k5(1.0);
    for m in 1..=5 {
        k[30 + m] = s2 * k5(r5[m]);
        k[30 - m] = k[30 + m];
    }
    let r6 = [
        1.0,
        1.0 / 42.0,
        1.0 / 1680.0,
        1.0 / 60480.0,
        1.0 / 1814400.0,
        1.0 / 39916800.0,
        1.0 / 479001600.0,
    ];
    k[42] = k6(1.0);
    for m in 1..=6 {
        k[42 + m] = s2 * k6(r6[m]);
        k[42 - m] = k[42 + m];
    }

    // `real_sh6` leaves the 1/8 and 1/16 of its first Legendre derivatives
    // here, powers of two that scale exactly.
    for m in 0..=3 {
        k[30 + m] *= 0.125;
        k[30 - m] = k[30 + m];
    }
    for m in 0..=4 {
        k[42 + m] *= 0.0625;
        k[42 - m] = k[42 + m];
    }
    k
}

/// Real orthonormal spherical harmonics through l equals 6 without their
/// prefactors, flattened.
///
/// Every term is a polynomial, no transcendentals, which is what keeps the
/// identity reproducible across targets. The degrees come in groups a CPU sums
/// one at a time, with each group's running totals held in registers.
#[cfg(any(target_arch = "nvptx64", target_arch = "amdgpu"))]
#[inline(always)]
pub(crate) fn real_sh6<T: Real>(u: [T; 3]) -> [T; 49] {
    let mut o = [T::splat(0.0); 49];
    o[0] = T::splat(1.0);
    let (d12, d3, d4, d5, d6) = (sh12(u), sh3(u), sh4(u), sh5(u), sh6(u));
    for k in 0..8 {
        o[1 + k] = d12[k];
    }
    for k in 0..7 {
        o[9 + k] = d3[k];
    }
    for k in 0..9 {
        o[16 + k] = d4[k];
    }
    for k in 0..11 {
        o[25 + k] = d5[k];
    }
    for k in 0..13 {
        o[36 + k] = d6[k];
    }
    o
}

/// Degrees 1 and 2, harmonics 1 to 8.
#[inline(always)]
pub(crate) fn sh12<T: Real>([x, y, z]: [T; 3]) -> [T; 8] {
    let k = T::splat;
    [y, z, x, x * y, y * z, k(3.0).fma(z * z, k(-1.0)), x * z, x * x - y * y]
}

/// Degree 3, harmonics 9 to 15.
#[inline(always)]
pub(crate) fn sh3<T: Real>([x, y, z]: [T; 3]) -> [T; 7] {
    let k = T::splat;
    let (x2, y2, z2) = (x * x, y * y, z * z);
    let z5m1 = k(5.0).fma(z2, k(-1.0));
    [
        y * k(3.0).fma(x2, -y2),
        x * y * z,
        y * z5m1,
        z * k(5.0).fma(z2, k(-3.0)),
        x * z5m1,
        z * (x2 - y2),
        x * k(-3.0).fma(y2, x2),
    ]
}

/// Degree 4, harmonics 16 to 24.
#[inline(always)]
pub(crate) fn sh4<T: Real>([x, y, z]: [T; 3]) -> [T; 9] {
    let k = T::splat;
    let (x2, y2, z2) = (x * x, y * y, z * z);
    let (xy, yz, xz) = (x * y, y * z, x * z);
    let x2my2 = x2 - y2;
    let x2m3y2 = k(-3.0).fma(y2, x2);
    let z7m1 = k(7.0).fma(z2, k(-1.0));
    let z7m3 = k(7.0).fma(z2, k(-3.0));
    [
        xy * x2my2,
        yz * k(3.0).fma(x2, -y2),
        xy * z7m1,
        yz * z7m3,
        k(35.0).fma(z2, k(-30.0)).fma(z2, k(3.0)),
        xz * z7m3,
        x2my2 * z7m1,
        xz * x2m3y2,
        x2.fma(k(-6.0).fma(y2, x2), y2 * y2),
    ]
}

/// Degree 5, harmonics 25 to 35. The 1/8 of its first Legendre derivatives
/// sits in `SH_NORM`, which scales every finished sum exactly the same.
#[inline(always)]
pub(crate) fn sh5<T: Real>([x, y, z]: [T; 3]) -> [T; 11] {
    let k = T::splat;
    let z2 = z * z;
    let (am, bm) = planar(x, y);
    let d = [
        z * k(63.0).fma(z2, k(-70.0)).fma(z2, k(15.0)),
        k(315.0).fma(z2, k(-210.0)).fma(z2, k(15.0)),
        z * k(1260.0).fma(z2, k(-420.0)),
        k(3780.0).fma(z2, k(-420.0)),
        k(945.0) * z,
        k(945.0),
    ];
    let mut o = [d[0]; 11];
    for m in 1..=5 {
        o[5 + m] = am[m] * d[m];
        o[5 - m] = bm[m] * d[m];
    }
    o
}

/// Degree 6, harmonics 36 to 48, with the 1/16 of its derivatives in
/// `SH_NORM` likewise.
#[inline(always)]
pub(crate) fn sh6<T: Real>([x, y, z]: [T; 3]) -> [T; 13] {
    let k = T::splat;
    let z2 = z * z;
    let (am, bm) = planar(x, y);
    let d = [
        k(231.0).fma(z2, k(-315.0)).fma(z2, k(105.0)).fma(z2, k(-5.0)),
        z * k(1386.0).fma(z2, k(-1260.0)).fma(z2, k(210.0)),
        k(6930.0).fma(z2, k(-3780.0)).fma(z2, k(210.0)),
        z * k(27720.0).fma(z2, k(-7560.0)),
        k(83160.0).fma(z2, k(-7560.0)),
        k(10395.0) * z,
        k(10395.0),
    ];
    let mut o = [d[0]; 13];
    for m in 1..=6 {
        o[6 + m] = am[m] * d[m];
        o[6 - m] = bm[m] * d[m];
    }
    o
}

/// The real and imaginary parts of (x + iy)^m, m up to 6.
#[inline(always)]
fn planar<T: Real>(x: T, y: T) -> ([T; 7], [T; 7]) {
    let mut am = [T::splat(0.0); 7];
    let mut bm = [T::splat(0.0); 7];
    am[1] = x;
    bm[1] = y;
    for m in 2..=6 {
        am[m] = x.fma(am[m - 1], -(y * bm[m - 1]));
        bm[m] = x.fma(bm[m - 1], y * am[m - 1]);
    }
    (am, bm)
}

#[cfg(test)]
mod tests {
    #[test]
    fn baked_prefactors_match_their_derivation() {
        let derived = super::sh_norm();
        for (i, (baked, derived)) in super::SH_NORM.iter().zip(derived).enumerate() {
            assert_eq!(baked.to_bits(), derived.to_bits(), "prefactor {i}");
        }
    }
}
