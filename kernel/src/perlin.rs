use crate::real::Real;
use crate::rng::Rng;
use crate::split::Split;

/// Pinned octave count.
pub const OCTAVES: usize = 5;

const LACUNARITY: f32 = 2.0;
const PERSISTENCE: f32 = 0.5;

#[repr(C)]
pub struct Fbm {
    /// The permutation in overlapping pairs, entry `i` holding `p[i]` in its
    /// low byte and `p[i + 1]` in its high byte, indices mod 256. The lattice
    /// hash always wants a neighbouring pair, so this halves the lookups.
    pairs: [u16; 256],
    frequency: Split,
}

/// Working storage for the permutation shuffle.
///
/// A GPU lends each thread a strip of shared memory, since in local memory
/// every one of the shuffle's random swaps waits on L2.
pub trait Perm {
    fn get(&self, i: usize) -> u8;
    fn set(&mut self, i: usize, v: u8);
}

impl Perm for [u8; 256] {
    #[inline(always)]
    fn get(&self, i: usize) -> u8 {
        self[i]
    }

    #[inline(always)]
    fn set(&mut self, i: usize, v: u8) {
        self[i] = v;
    }
}

/// Where the lattice hash and gradients come from, a lane at a time or a
/// vector of lanes.
///
/// A GPU reads both from shared memory by 32 bit address, where a reference
/// into the parameters would pay 64 bit address arithmetic on every one of
/// the 75 lookups a vertex makes. A CPU gathers them.
pub trait Lattice<T: Real> {
    /// `p[i]` and `p[i + 1]` of the doubled 512 entry table, for `i` up to 510.
    fn pair(&self, i: T::Bits) -> (T::Bits, T::Bits);

    /// Dot of lattice gradient `hash` with the corner offset.
    fn grad(&self, hash: T::Bits, x: T, y: T, z: T) -> T;
}

/// Byte offset of the permutation pairs within `Fbm`.
#[cfg(any(target_arch = "nvptx64", target_arch = "amdgpu"))]
pub(crate) const PAIRS: usize = core::mem::offset_of!(Fbm, pairs);

impl Fbm {
    /// Shuffle the permutation from `seed` in `perm` and store it as pairs.
    #[inline(always)]
    pub fn fill(&mut self, seed: [u8; 32], frequency: f64, perm: &mut impl Perm) {
        shuffle(seed, perm);
        for i in 0..256 {
            self.pairs[i] = perm.get(i) as u16 | (perm.get((i + 1) & 255) as u16) << 8;
        }
        self.frequency = Split::new(frequency);
    }

    #[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
    pub(crate) fn pairs(&self) -> &[u16; 256] {
        &self.pairs
    }

    /// Fractal sum at a blank direction.
    ///
    /// Written out rather than mapped over arrays, since a CPU compiles the
    /// helpers `map` goes through without its vector instructions unless they
    /// inline. Lattice coordinates are plain f32. Their rounding moves the in-cell
    /// fraction by up to 2e-6 at the top octave, differently at every vertex,
    /// and costs about 0.3% of retention on a GPU, where carrying them split
    /// costs 6% of throughput. The frequency's low half still goes in, since
    /// its rounding would stretch the whole noise field alike.
    #[inline(always)]
    pub fn get<T: Real>(&self, dir: [Split<T>; 3], lattice: &impl Lattice<T>) -> T {
        let mut point = [self.frequency.scale(dir[0].hi), self.frequency.scale(dir[1].hi), self.frequency.scale(dir[2].hi)];
        let mut amp = 1.0;
        let mut sum = T::splat(0.0);
        let mut norm = 0.0;
        for _ in 0..OCTAVES {
            let ((xi, x), (yi, y), (zi, z)) = (cell(point[0]), cell(point[1]), cell(point[2]));
            sum = sum + T::splat(amp) * noise(lattice, xi, yi, zi, x, y, z);
            norm += amp;
            for c in &mut point {
                *c = *c * T::splat(LACUNARITY);
            }
            amp *= PERSISTENCE;
        }
        sum / T::splat(norm)
    }
}

/// One octave, a trilinear blend of eight corner gradients under a quintic
/// fade, at the fraction `x, y, z` inside lattice cell `ix, iy, iz`.
#[inline(always)]
fn noise<T: Real>(lattice: &impl Lattice<T>, ix: T::Bits, iy: T::Bits, iz: T::Bits, x: T, y: T, z: T) -> T {
    let byte = T::bits(255);
    let (xi, yi, zi) = (ix & byte, iy & byte, iz & byte);

    let u = fade(x);
    let v = fade(y);
    let w = fade(z);

    let (px, px1) = lattice.pair(xi);
    let (a, b) = (px + yi, px1 + yi);
    let (pa, pa1) = lattice.pair(a);
    let (pb, pb1) = lattice.pair(b);
    let (aa, ab, ba, bb) = (pa + zi, pa1 + zi, pb + zi, pb1 + zi);
    let (gaa, gaa1) = lattice.pair(aa);
    let (gab, gab1) = lattice.pair(ab);
    let (gba, gba1) = lattice.pair(ba);
    let (gbb, gbb1) = lattice.pair(bb);
    let grad = |h, x, y, z| lattice.grad(h, x, y, z);
    let (x1, y1, z1) = (x - T::splat(1.0), y - T::splat(1.0), z - T::splat(1.0));

    lerp(
        w,
        lerp(v, lerp(u, grad(gaa, x, y, z), grad(gba, x1, y, z)), lerp(u, grad(gab, x, y1, z), grad(gbb, x1, y1, z))),
        lerp(
            v,
            lerp(u, grad(gaa1, x, y, z1), grad(gba1, x1, y, z1)),
            lerp(u, grad(gab1, x, y1, z1), grad(gbb1, x1, y1, z1)),
        ),
    )
}

#[inline(always)]
fn shuffle(seed: [u8; 32], p: &mut impl Perm) {
    let mut rng = Rng::new(seed);
    for i in 0..256 {
        p.set(i, i as u8);
    }
    for i in (1..256).rev() {
        let j = rng.index(i);
        let (a, b) = (p.get(i), p.get(j));
        p.set(i, b);
        p.set(j, a);
    }
}

/// Lattice cell and in-cell fraction of a coordinate. The cell comes as the
/// bits of its whole number plus 1.5 * 2^23, which puts the integer in the
/// low mantissa bits, exact below 2^22 in magnitude. The lattice stays within
/// 40 of the origin and the hash only wants the low byte.
#[inline(always)]
fn cell<T: Real>(c: T) -> (T::Bits, T) {
    let floor = c.floor();
    ((floor + T::splat(12582912.0)).to_bits(), c - floor)
}

#[inline(always)]
fn fade<T: Real>(t: T) -> T {
    t * t * t * t.fma(t.fma(T::splat(6.0), T::splat(-15.0)), T::splat(10.0))
}

#[inline(always)]
fn lerp<T: Real>(t: T, a: T, b: T) -> T {
    t.fma(b - a, a)
}

/// Dot of lattice gradient `hash` with the corner offset. Every gradient is
/// two unit axes, so the reference adds two of `±x, ±y, ±z` and rounds once.
/// A GPU and a CPU vector look the gradient up in `GRADS` instead and take
/// the dot as a fused multiply add chain. One component of each row is zero,
/// so the chain rounds once, exactly where the reference's single add does.
#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
#[inline(always)]
pub(crate) fn grad(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if (h & 1) == 0 { u } else { -u }) + (if (h & 2) == 0 { v } else { -v })
}

#[repr(C, align(16))]
pub(crate) struct Row(pub(crate) [f32; 4]);

/// The gradients as rows of x, y and z weights, the table a GPU or a CPU
/// vector looks a lattice hash up in.
pub(crate) static GRADS: [Row; 16] = [
    Row([1.0, 1.0, 0.0, 0.0]),
    Row([-1.0, 1.0, 0.0, 0.0]),
    Row([1.0, -1.0, 0.0, 0.0]),
    Row([-1.0, -1.0, 0.0, 0.0]),
    Row([1.0, 0.0, 1.0, 0.0]),
    Row([-1.0, 0.0, 1.0, 0.0]),
    Row([1.0, 0.0, -1.0, 0.0]),
    Row([-1.0, 0.0, -1.0, 0.0]),
    Row([0.0, 1.0, 1.0, 0.0]),
    Row([0.0, -1.0, 1.0, 0.0]),
    Row([0.0, 1.0, -1.0, 0.0]),
    Row([0.0, -1.0, -1.0, 0.0]),
    Row([1.0, 1.0, 0.0, 0.0]),
    Row([0.0, -1.0, 1.0, 0.0]),
    Row([-1.0, 1.0, 0.0, 0.0]),
    Row([0.0, -1.0, -1.0, 0.0]),
];
