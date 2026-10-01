#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

use alloc::vec;
use alloc::vec::Vec;

use crate::coop::{PATCHES, SH};
use crate::intrinsics::{ceilf, floorf};
use crate::perlin::{Fbm, Lattice, grad};
use crate::pipeline::{
    Body, Frame, HIST_BINS, HIST_STEP, N_FEATURES, Sum, TARGET_SAMPLES, assemble, centroid, cross, dot, sh3, sh4, sh5, sh6,
    sh12, sub,
};
use crate::real::Real;
use crate::sculpt::Params;
use crate::split::Split;
use crate::topology::{N_FACES, N_VERTS};

/// What the driver needs beyond arithmetic, the moves of whole vectors.
pub(crate) trait Lanes: Real {
    const N: usize;

    unsafe fn load(p: *const f32) -> Self;
    unsafe fn store(self, p: *mut f32);
    unsafe fn load_bits(p: *const u32) -> Self::Bits;
    /// `base[idx]` in each lane.
    unsafe fn gather(base: *const f32, idx: Self::Bits) -> Self;
    /// The lanes set in `mask`, stored one after another from `p`, which
    /// must have room for a whole vector. Returns how many.
    unsafe fn compress(self, mask: Self::Mask, p: *mut f32) -> usize;
    fn not(mask: Self::Mask) -> Self::Mask;
    fn bitmask(mask: Self::Mask) -> u32;
    /// The lanes below `n`.
    fn first(n: usize) -> Self::Mask;
    /// Add each lane into its own f64 total.
    fn widen(self, acc: &mut [f64]);
}

/// The permutation pairs copied out of a draw, with room past the last pair
/// for a 32 bit read that brings it in its low half.
pub(crate) struct Pairs([u16; 258]);

impl Pairs {
    #[inline(always)]
    pub(crate) fn new(fbm: &Fbm) -> Pairs {
        let mut p = [0; 258];
        p[..256].copy_from_slice(fbm.pairs());
        Pairs(p)
    }
}

impl Lattice<f32> for Pairs {
    #[inline(always)]
    fn pair(&self, i: u32) -> (u32, u32) {
        let v = self.0[(i & 255) as usize] as u32;
        (v & 0xff, v >> 8)
    }

    #[inline(always)]
    fn grad(&self, hash: u32, x: f32, y: f32, z: f32) -> f32 {
        grad(hash, x, y, z)
    }
}

impl Lanes for f32 {
    const N: usize = 1;

    #[inline(always)]
    unsafe fn load(p: *const f32) -> f32 {
        unsafe { *p }
    }

    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        unsafe { *p = self };
    }

    #[inline(always)]
    unsafe fn load_bits(p: *const u32) -> u32 {
        unsafe { *p }
    }

    #[inline(always)]
    unsafe fn gather(base: *const f32, idx: u32) -> f32 {
        unsafe { *base.add(idx as usize) }
    }

    #[inline(always)]
    unsafe fn compress(self, mask: bool, p: *mut f32) -> usize {
        unsafe { *p = self };
        mask as usize
    }

    #[inline(always)]
    fn not(mask: bool) -> bool {
        !mask
    }

    #[inline(always)]
    fn bitmask(mask: bool) -> u32 {
        mask as u32
    }

    #[inline(always)]
    fn first(n: usize) -> bool {
        n > 0
    }

    #[inline(always)]
    fn widen(self, acc: &mut [f64]) {
        acc[0] += self as f64;
    }
}

/// Vertices a whole number of patches holds.
const SLOTS: usize = PATCHES * 32;

/// Batches a vector sums in f32 before folding into its f64 totals, few
/// enough that the f32 rounding stays well below the bucket widths.
const FLUSH: usize = 32;

/// Room for four samples a face, every body measured, and a padding vector.
const STREAM: usize = 4 * N_FACES + 16;

/// Room for the faces split in two that wait on a whole vector of them.
const HELD: usize = 48;

/// Barycentric weights of samples one to three of a face split in two per
/// side, its downward triangle and then the upward ones past the first.
const SPLIT_TWO: [(f32, f32); 3] = [(2.0 / 6.0, 2.0 / 6.0), (1.0 / 6.0, 4.0 / 6.0), (4.0 / 6.0, 1.0 / 6.0)];

/// One worker's state. Build it once per thread and reuse it for every
/// nonce.
pub struct Cpu {
    /// A path this CPU runs, which the vector scans count on.
    width: Width,
    /// Blank directions in patch order, high then low halves, one array per
    /// coordinate and padded to whole patches.
    dirs: [Vec<f32>; 6],
    /// Faces as renumbered vertex indices, one array per corner.
    tris: [Vec<u32>; 3],
    patches: Vec<[f32; 4]>,
    verts: [Vec<f32>; 3],
    /// Every face as its first corner, its two edges from there and its area,
    /// kept from the face sums for the walk, which then reads them in order
    /// rather than gathering corners again.
    faces: [Vec<f32>; 10],
    stream: Stream,
    /// Faces split in two per side, collected until they fill a vector, as
    /// the first corner relative to the centroid, both edges and a quarter of
    /// the area.
    held: [Vec<f32>; 10],
}

/// A SIMD path through the scan.
#[derive(Clone, Copy)]
pub enum Width {
    #[cfg(target_arch = "x86_64")]
    Vbmi,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    Scalar,
}

/// A body's surface samples relative to its centroid, a direction, radius
/// and weight each.
struct Stream {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    r: Vec<f32>,
    w: Vec<f32>,
    len: usize,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu {
    pub fn new() -> Self {
        let (base, faces) = crate::topology::build();
        let packed = crate::coop::pack(&base, &faces);
        let dir = |k: usize| -> Vec<f32> {
            let mut v: Vec<f32> = packed.dirs.chunks(8).map(|d| d[k]).collect();
            v.resize(SLOTS, v[N_VERTS - 1]);
            v
        };
        let tri = |k: usize| -> Vec<u32> { packed.tris.iter().map(|&t| (t >> (16 * k) & 0xffff) as u32).collect() };
        let column = || vec![0.0; STREAM];
        Cpu {
            width: Width::widest(),
            dirs: [dir(0), dir(1), dir(2), dir(4), dir(5), dir(6)],
            tris: [tri(0), tri(1), tri(2)],
            patches: packed.patches.chunks(4).map(|p| [p[0], p[1], p[2], p[3]]).collect(),
            verts: [vec![0.0; SLOTS], vec![0.0; SLOTS], vec![0.0; SLOTS]],
            faces: core::array::from_fn(|_| vec![0.0; N_FACES]),
            stream: Stream { x: column(), y: column(), z: column(), r: column(), w: column(), len: 0 },
            held: core::array::from_fn(|_| vec![0.0; HELD]),
        }
    }

    /// Raw features of one drawn body, or `None` when it came out
    /// unmeasurable, which a valid seal never is.
    pub fn features(&mut self, params: &Params) -> Option<[f64; N_FEATURES]> {
        match self.width {
            #[cfg(target_arch = "x86_64")]
            Width::Vbmi => unsafe { self.scan_vbmi(params) },
            #[cfg(target_arch = "x86_64")]
            Width::Avx512 => unsafe { self.scan_avx512(params) },
            #[cfg(target_arch = "x86_64")]
            Width::Avx2 => unsafe { self.scan_avx2(params) },
            Width::Scalar => self.scan::<f32>(params, &Pairs::new(params.fbm())),
        }
    }

    /// AVX-512 with byte permutes, which look the lattice up from registers,
    /// a fifth faster overall than gathering it on a Ryzen 9 9950X3D.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    unsafe fn scan_vbmi(&mut self, params: &Params) -> Option<[f64; N_FEATURES]> {
        self.scan::<avx512::V>(params, &avx512::Bytes::new(params.fbm()))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn scan_avx512(&mut self, params: &Params) -> Option<[f64; N_FEATURES]> {
        self.scan::<avx512::V>(params, &Pairs::new(params.fbm()))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn scan_avx2(&mut self, params: &Params) -> Option<[f64; N_FEATURES]> {
        self.scan::<avx2::V>(params, &Pairs::new(params.fbm()))
    }

    #[inline(always)]
    fn scan<V: Lanes>(&mut self, params: &Params, lattice: &impl Lattice<V>) -> Option<[f64; N_FEATURES]> {
        self.sculpt::<V>(params, lattice);
        let body = self.faces::<V>();
        let frame = Frame::new(&body)?;
        let area = body[0];
        self.walk::<V>((area / TARGET_SAMPLES as f64) as f32, centroid(&body));
        let (sh, radius) = self.harmonics::<V>();
        let rbar = radius / area;
        let hist = self.histogram::<V>(rbar);
        Some(assemble(&frame, area, rbar, |i| (sh[i - 1], sh[SH / 2 + i - 1]), |j| hist[j]))
    }

    #[inline(always)]
    fn sculpt<V: Lanes>(&mut self, params: &Params, lattice: &impl Lattice<V>) {
        for (p, &[x, y, z, r]) in self.patches.iter().enumerate() {
            let near = params.near([x, y, z], r);
            let mut i = 32 * p;
            while i < 32 * (p + 1) {
                let dir: [Split<V>; 3] = core::array::from_fn(|k| unsafe {
                    Split { hi: V::load(self.dirs[k].as_ptr().add(i)), lo: V::load(self.dirs[3 + k].as_ptr().add(i)) }
                });
                let v = params.displace_with(dir, lattice, near);
                for (k, c) in v.into_iter().enumerate() {
                    unsafe { c.store(self.verts[k].as_mut_ptr().add(i)) };
                }
                i += V::N;
            }
        }
    }

    /// Faces `f` on, one per lane, as their three corners.
    #[inline(always)]
    fn tri<V: Lanes>(&self, f: usize) -> [[V; 3]; 3] {
        core::array::from_fn(|k| {
            let i = unsafe { V::load_bits(self.tris[k].as_ptr().add(f)) };
            core::array::from_fn(|q| unsafe { V::gather(self.verts[q].as_ptr(), i) })
        })
    }

    #[inline(always)]
    fn faces<V: Lanes>(&mut self) -> [f64; 11] {
        let mut body = Body::<V>::zero();
        let mut f = 0;
        while f < N_FACES {
            let [a, b, c] = self.tri::<V>(f);
            let area = body.add(a, b, c);
            let (e1, e2) = (sub(b, a), sub(c, a));
            let kept = [a[0], a[1], a[2], e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], area];
            for (v, out) in kept.into_iter().zip(self.faces.iter_mut()) {
                unsafe { v.store(out.as_mut_ptr().add(f)) };
            }
            f += V::N;
        }
        body.get()
    }

    /// Lay out the samples relative to `c`, every face's first one and then,
    /// for the faces split in two per side, the other three.
    ///
    /// About one face in seven splits in two. Those wait in `held` until they
    /// fill a vector rather than run their three samples with most lanes idle.
    #[inline(always)]
    fn walk<V: Lanes>(&mut self, per: f32, c: [f32; 3]) {
        let cv = [V::splat(c[0]), V::splat(c[1]), V::splat(c[2])];
        let mut n = 0;
        let mut held = 0;
        let mut f = 0;
        while f < N_FACES {
            self.stream.reserve(n + 4 * V::N);
            let [p0x, p0y, p0z, e1x, e1y, e1z, e2x, e2y, e2z, area] =
                core::array::from_fn(|k| unsafe { V::load(self.faces[k].as_ptr().add(f)) });
            let (a, e1, e2) = (sub([p0x, p0y, p0z], cv), [e1x, e1y, e1z], [e2x, e2y, e2z]);
            // k = ceil(sqrt(area / per)), the subdivision per side, by compares
            // for the two sizes every measured face takes.
            let one = area.le(V::splat(per));
            let two = area.le(V::splat(4.0 * per)) & V::not(one);
            let step = V::select(one, V::splat(1.0 / 3.0), V::splat(1.0 / 6.0));
            let w = V::select(one, area, V::select(two, area * V::splat(0.25), V::splat(0.0)));
            let p: [V; 3] = core::array::from_fn(|q| step.fma(e2[q], step.fma(e1[q], a[q])));
            n += self.stream.put(n, p, w, None);
            if V::any(two) {
                let fields = [a[0], a[1], a[2], e1[0], e1[1], e1[2], e2[0], e2[1], e2[2], area * V::splat(0.25)];
                let mut count = 0;
                for (v, out) in fields.into_iter().zip(self.held.iter_mut()) {
                    count = unsafe { v.compress(two, out.as_mut_ptr().add(held)) };
                }
                held += count;
                if held >= V::N {
                    n = self.extras::<V>(n, None);
                    for out in &mut self.held {
                        unsafe { V::load(out.as_ptr().add(V::N)).store(out.as_mut_ptr()) };
                    }
                    held -= V::N;
                }
            }
            let big = V::bitmask(V::not(area.le(V::splat(4.0 * per))));
            if big != 0 {
                n = self.split(f, big, per, c, n);
            }
            f += V::N;
        }
        if held > 0 {
            n = self.extras::<V>(n, Some(V::first(held)));
        }
        self.stream.reserve(n + V::N);
        self.stream.len = n;
        // Zero weights pad the last batch, and zero directions keep the
        // harmonics of the padding finite.
        let s = &mut self.stream;
        for v in [&mut s.x, &mut s.y, &mut s.z, &mut s.r, &mut s.w] {
            v[n..n + V::N].fill(0.0);
        }
    }

    /// Samples one to three of the held faces, a vector of them or those in
    /// `mask`, after the stream's first `n`. Returns the stream's new length.
    #[inline(always)]
    fn extras<V: Lanes>(&mut self, mut n: usize, mask: Option<V::Mask>) -> usize {
        let h: [V; 10] = core::array::from_fn(|k| unsafe { V::load(self.held[k].as_ptr()) });
        for (u, v) in SPLIT_TWO {
            let p: [V; 3] = core::array::from_fn(|q| V::splat(v).fma(h[6 + q], V::splat(u).fma(h[3 + q], h[q])));
            n += self.stream.put(n, p, h[9], mask);
        }
        n
    }

    /// The faces in `lanes` of the batch from `f`, split three or more ways
    /// per side, which no measured body has had, a sample at a time and after
    /// the stream's first `n`. Returns the stream's new length.
    #[cold]
    fn split(&mut self, f: usize, lanes: u32, per: f32, c: [f32; 3], mut n: usize) -> usize {
        for l in 0..32 {
            if lanes & 1 << l == 0 {
                continue;
            }
            let [p0, p1, p2] = core::array::from_fn(|k| {
                let i = self.tris[k][f + l] as usize;
                [self.verts[0][i], self.verts[1][i], self.verts[2][i]]
            });
            let (a, e1, e2) = (sub(p0, c), sub(p1, p0), sub(p2, p0));
            let nrm = cross(e1, e2);
            let area = 0.5 * Real::sqrt(dot(nrm, nrm));
            let k = (ceilf(Real::sqrt(area / per)) as usize).clamp(1, 64);
            let w = area / (k * k) as f32;
            let kf = (3 * k) as f32;
            self.stream.reserve(n + k * k + 1);
            for i in 0..k {
                for j in 0..(k - i) {
                    for d in [1, 2] {
                        if d == 2 && i + j + 1 == k {
                            continue;
                        }
                        let (u, v) = ((3 * i + d) as f32 / kf, (3 * j + d) as f32 / kf);
                        let p: [f32; 3] = core::array::from_fn(|q| v.fma(e2[q], u.fma(e1[q], a[q])));
                        n += self.stream.put(n, p, w, None);
                    }
                }
            }
        }
        n
    }

    /// Harmonic sums over the stream, weighted by area and then by area times
    /// radius, and the radius weighted by area.
    ///
    /// A pass over the stream per group of degrees, since all 96 running
    /// totals at once would not stay in registers.
    #[inline(always)]
    fn harmonics<V: Lanes>(&self) -> ([f64; SH], f64) {
        let mut wide = [0.0f64; SH * 16];
        let wide = &mut wide[..SH * V::N];
        let radius = self.degrees::<V, 8, D12, true>(wide, 0);
        self.degrees::<V, 7, D3, false>(wide, 8);
        self.degrees::<V, 9, D4, false>(wide, 15);
        self.degrees::<V, 11, D5, false>(wide, 24);
        self.degrees::<V, 13, D6, false>(wide, 35);
        (core::array::from_fn(|k| wide[k * V::N..(k + 1) * V::N].iter().sum()), radius)
    }

    /// Sums of the `K` harmonics of `G`, from harmonic `first + 1` on, into
    /// their lanes of `wide`, and with `RADIUS` the radius weighted by area.
    #[inline(always)]
    fn degrees<V: Lanes, const K: usize, G: Degrees<V, K>, const RADIUS: bool>(
        &self,
        wide: &mut [f64],
        first: usize,
    ) -> f64 {
        let s = &self.stream;
        let mut radius = Sum::<V>::zero();
        let mut i = 0;
        while i < s.len {
            let end = (i + FLUSH * V::N).min(s.len);
            let mut by_w = [V::splat(0.0); K];
            let mut by_wr = [V::splat(0.0); K];
            while i < end {
                let (x, y, z, r, w) = unsafe {
                    (V::load(s.x.as_ptr().add(i)), V::load(s.y.as_ptr().add(i)), V::load(s.z.as_ptr().add(i)),
                     V::load(s.r.as_ptr().add(i)), V::load(s.w.as_ptr().add(i)))
                };
                let wr = w * r;
                if RADIUS {
                    radius.add(wr);
                }
                let y = G::at([x, y, z]);
                for k in 0..K {
                    by_w[k] = w.fma(y[k], by_w[k]);
                    by_wr[k] = wr.fma(y[k], by_wr[k]);
                }
                i += V::N;
            }
            for k in 0..K {
                by_w[k].widen(&mut wide[(first + k) * V::N..]);
                by_wr[k].widen(&mut wide[(SH / 2 + first + k) * V::N..]);
            }
        }
        radius.get()
    }

    #[inline(always)]
    fn histogram<V: Lanes>(&self, rbar: f64) -> [f64; HIST_BINS] {
        let s = &self.stream;
        let bins = Split::new(1.0 / (HIST_STEP * rbar));
        let mut wide = [0.0f64; HIST_BINS * 16];
        let wide = &mut wide[..HIST_BINS * V::N];
        let mut acc = [V::splat(0.0); HIST_BINS];
        let mut i = 0;
        let mut batch = 0;
        while i < s.len {
            let (r, w) = unsafe { (V::load(s.r.as_ptr().add(i)), V::load(s.w.as_ptr().add(i))) };
            let x = (bins.scale(r) - V::splat(0.5)).clamp(0.0, (HIST_BINS - 1) as f32);
            for (b, h) in acc.iter_mut().enumerate() {
                *h = w.fma((V::splat(1.0) - (x - V::splat(b as f32)).abs()).max(V::splat(0.0)), *h);
            }
            batch += 1;
            if batch == FLUSH {
                flush(&mut acc, wide);
                batch = 0;
            }
            i += V::N;
        }
        flush(&mut acc, wide);
        core::array::from_fn(|b| wide[b * V::N..(b + 1) * V::N].iter().sum())
    }
}

/// A group of harmonic degrees, as `harmonics` sums them in one pass. A
/// trait rather than a function argument, since the call through a function
/// value compiles apart from the vector instructions it needs.
trait Degrees<V: Real, const K: usize> {
    fn at(u: [V; 3]) -> [V; K];
}

struct D12;
struct D3;
struct D4;
struct D5;
struct D6;

macro_rules! degrees {
    ($($g:ident $k:literal $sh:ident),*) => {
        $(impl<V: Real> Degrees<V, $k> for $g {
            #[inline(always)]
            fn at(u: [V; 3]) -> [V; $k] {
                $sh(u)
            }
        })*
    };
}

degrees!(D12 8 sh12, D3 7 sh3, D4 9 sh4, D5 11 sh5, D6 13 sh6);

/// Fold vector totals into their f64 lanes and start them over.
#[inline(always)]
fn flush<V: Lanes, const K: usize>(acc: &mut [V; K], wide: &mut [f64]) {
    for (k, a) in acc.iter_mut().enumerate() {
        a.widen(&mut wide[k * V::N..(k + 1) * V::N]);
        *a = V::splat(0.0);
    }
}

impl Stream {
    /// Store samples `p` with weight `w` at `n`, every lane or those in
    /// `mask`, as direction, radius and weight. Returns how many went in.
    #[inline(always)]
    fn put<V: Lanes>(&mut self, n: usize, p: [V; 3], w: V, mask: Option<V::Mask>) -> usize {
        let r = dot(p, p).sqrt();
        let inv = V::splat(1.0) / r;
        let fields = [p[0] * inv, p[1] * inv, p[2] * inv, r, w];
        let mut count = V::N;
        for (v, out) in fields.into_iter().zip([&mut self.x, &mut self.y, &mut self.z, &mut self.r, &mut self.w]) {
            let at = unsafe { out.as_mut_ptr().add(n) };
            match mask {
                Some(mask) => count = unsafe { v.compress(mask, at) },
                None => unsafe { v.store(at) },
            }
        }
        count
    }

    /// Room for `len` samples and a whole vector past them.
    #[inline(always)]
    fn reserve(&mut self, len: usize) {
        if len + 16 > self.x.len() {
            self.grow(len + 16);
        }
    }

    #[cold]
    fn grow(&mut self, len: usize) {
        for v in [&mut self.x, &mut self.y, &mut self.z, &mut self.r, &mut self.w] {
            v.resize(2 * len, 0.0);
        }
    }
}

impl Width {
    /// Every width built for this target, widest first.
    const ALL: &[Width] = &[
        #[cfg(target_arch = "x86_64")]
        Width::Vbmi,
        #[cfg(target_arch = "x86_64")]
        Width::Avx512,
        #[cfg(target_arch = "x86_64")]
        Width::Avx2,
        Width::Scalar,
    ];

    /// The widest this CPU runs.
    pub fn widest() -> Width {
        Width::ALL.iter().copied().find(|width| width.runs()).unwrap_or(Width::Scalar)
    }

    /// Whether this CPU has the instructions the width's scan is built for.
    pub fn runs(self) -> bool {
        match self {
            #[cfg(target_arch = "x86_64")]
            Width::Vbmi => std::is_x86_feature_detected!("avx512vbmi") && std::is_x86_feature_detected!("avx512bw"),
            #[cfg(target_arch = "x86_64")]
            Width::Avx512 => std::is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "x86_64")]
            Width::Avx2 => std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
            Width::Scalar => true,
        }
    }
}

pub(crate) const LN_2: Split = Split::new(core::f64::consts::LN_2);
pub(crate) const LOG2_E: Split = Split::new(core::f64::consts::LOG2_E);

/// `x` to the power `y` for `x` in (0, 1].
///
/// libm's f32 routines work in f64 internally, which a vector runs at half
/// its f32 width. These stay in f32, and their series run far enough that
/// what error is left is rounding, which averages out over the surface rather
/// than biasing every lobe the same way.
#[inline(always)]
pub(crate) fn pow(x: f32, y: Split) -> f32 {
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if m > core::f32::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let t = y.scale(log2(m, e as f32));
    if t < -125.0 {
        return 0.0;
    }
    let n = floorf(t + 0.5);
    exp(LN_2.scale(t - n)) * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// log2 of a value with mantissa `m` in [sqrt(1/2), sqrt(2)] and exponent
/// `e`, from the atanh series of the mantissa.
#[inline(always)]
pub(crate) fn log2<T: Real>(m: T, e: T) -> T {
    let f = m - T::splat(1.0);
    let s = f / (T::splat(2.0) + f);
    let z = s * s;
    let mut p = T::splat(1.0 / 13.0);
    for c in [1.0 / 11.0, 1.0 / 9.0, 1.0 / 7.0, 1.0 / 5.0, 1.0 / 3.0] {
        p = p.fma(z, T::splat(c));
    }
    LOG2_E.scale(T::splat(2.0) * s + T::splat(2.0) * s * z * p) + e
}

/// e^r for r within ln(2) / 2 of zero, from its Taylor series.
#[inline(always)]
pub(crate) fn exp<T: Real>(r: T) -> T {
    let mut p = T::splat(1.0 / 362880.0);
    for c in [1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
        p = p.fma(r, T::splat(c));
    }
    p
}
