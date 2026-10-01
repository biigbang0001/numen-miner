use core::arch::x86_64::*;
use core::ops::{Add, BitAnd, Div, Mul, Neg, Sub};

use super::{LN_2, Lanes, Pairs, exp, log2};
use crate::perlin::{Fbm, GRADS, Lattice};
use crate::real::Real;
use crate::split::Split;

#[derive(Clone, Copy)]
pub struct V(__m512);

#[derive(Clone, Copy)]
pub struct B(__m512i);

#[derive(Clone, Copy)]
pub struct M(__mmask16);

macro_rules! binary {
    ($t:ident, $trait:ident, $f:ident, $op:ident) => {
        impl $trait for $t {
            type Output = $t;

            #[inline(always)]
            fn $f(self, o: $t) -> $t {
                unsafe { $t($op(self.0, o.0)) }
            }
        }
    };
}

binary!(V, Add, add, _mm512_add_ps);
binary!(V, Sub, sub, _mm512_sub_ps);
binary!(V, Mul, mul, _mm512_mul_ps);
binary!(V, Div, div, _mm512_div_ps);
binary!(B, Add, add, _mm512_add_epi32);
binary!(B, BitAnd, bitand, _mm512_and_si512);

impl Neg for V {
    type Output = V;

    #[inline(always)]
    fn neg(self) -> V {
        unsafe { V(_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(self.0), _mm512_set1_epi32(i32::MIN)))) }
    }
}

impl BitAnd for M {
    type Output = M;

    #[inline(always)]
    fn bitand(self, o: M) -> M {
        M(self.0 & o.0)
    }
}

impl Real for V {
    type Mask = M;
    type Bits = B;

    #[inline(always)]
    fn splat(x: f32) -> V {
        unsafe { V(_mm512_set1_ps(x)) }
    }

    #[inline(always)]
    fn bits(x: u32) -> B {
        unsafe { B(_mm512_set1_epi32(x as i32)) }
    }

    #[inline(always)]
    fn fma(self, b: V, c: V) -> V {
        unsafe { V(_mm512_fmadd_ps(self.0, b.0, c.0)) }
    }

    #[inline(always)]
    fn floor(self) -> V {
        unsafe { V(_mm512_roundscale_ps::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(self.0)) }
    }

    #[inline(always)]
    fn sqrt(self) -> V {
        unsafe { V(_mm512_sqrt_ps(self.0)) }
    }

    #[inline(always)]
    fn abs(self) -> V {
        unsafe { V(_mm512_abs_ps(self.0)) }
    }

    #[inline(always)]
    fn max(self, o: V) -> V {
        unsafe { V(_mm512_max_ps(self.0, o.0)) }
    }

    #[inline(always)]
    fn clamp(self, lo: f32, hi: f32) -> V {
        unsafe { V(_mm512_max_ps(_mm512_min_ps(self.0, _mm512_set1_ps(hi)), _mm512_set1_ps(lo))) }
    }

    #[inline(always)]
    fn lt(self, o: V) -> M {
        unsafe { M(_mm512_cmp_ps_mask::<_CMP_LT_OQ>(self.0, o.0)) }
    }

    #[inline(always)]
    fn le(self, o: V) -> M {
        unsafe { M(_mm512_cmp_ps_mask::<_CMP_LE_OQ>(self.0, o.0)) }
    }

    #[inline(always)]
    fn gt(self, o: V) -> M {
        unsafe { M(_mm512_cmp_ps_mask::<_CMP_GT_OQ>(self.0, o.0)) }
    }

    #[inline(always)]
    fn select(mask: M, yes: V, no: V) -> V {
        unsafe { V(_mm512_mask_blend_ps(mask.0, no.0, yes.0)) }
    }

    #[inline(always)]
    fn any(mask: M) -> bool {
        mask.0 != 0
    }

    #[inline(always)]
    fn to_bits(self) -> B {
        unsafe { B(_mm512_castps_si512(self.0)) }
    }

    #[inline(always)]
    fn sum_f64(self) -> f64 {
        unsafe {
            let (lo, hi) = halves(self.0);
            _mm512_reduce_add_pd(_mm512_add_pd(lo, hi))
        }
    }

    /// The scalar series lane by lane, bit for bit.
    #[inline(always)]
    fn pow(self, y: Split) -> V {
        unsafe {
            let bits = _mm512_castps_si512(self.0);
            let e = _mm512_sub_epi32(_mm512_srli_epi32::<23>(bits), _mm512_set1_epi32(127));
            let m = V(_mm512_castsi512_ps(_mm512_or_si512(
                _mm512_and_si512(bits, _mm512_set1_epi32(0x007f_ffff)),
                _mm512_set1_epi32(0x3f80_0000),
            )));
            let up = m.gt(V::splat(core::f32::consts::SQRT_2));
            let m = V::select(up, m * V::splat(0.5), m);
            let e = V(_mm512_cvtepi32_ps(_mm512_mask_add_epi32(e, up.0, e, _mm512_set1_epi32(1))));
            let t = y.scale(log2(m, e));
            let n = (t + V::splat(0.5)).floor();
            let scale = _mm512_slli_epi32::<23>(_mm512_add_epi32(_mm512_cvttps_epi32(n.0), _mm512_set1_epi32(127)));
            let p = exp(LN_2.scale(t - n)) * V(_mm512_castsi512_ps(scale));
            V::select(t.lt(V::splat(-125.0)), V::splat(0.0), p)
        }
    }
}

/// The low and high eight lanes widened to f64.
#[inline(always)]
unsafe fn halves(v: __m512) -> (__m512d, __m512d) {
    unsafe {
        let hi = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(v)));
        (_mm512_cvtps_pd(_mm512_castps512_ps256(v)), _mm512_cvtps_pd(hi))
    }
}

impl Lanes for V {
    const N: usize = 16;

    #[inline(always)]
    unsafe fn load(p: *const f32) -> V {
        unsafe { V(_mm512_loadu_ps(p)) }
    }

    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        unsafe { _mm512_storeu_ps(p, self.0) };
    }

    #[inline(always)]
    unsafe fn load_bits(p: *const u32) -> B {
        unsafe { B(_mm512_loadu_si512(p.cast())) }
    }

    #[inline(always)]
    unsafe fn gather(base: *const f32, idx: B) -> V {
        unsafe { V(_mm512_i32gather_ps::<4>(idx.0, base)) }
    }

    /// Packed in a register and then stored whole, which Zen runs far
    /// faster than a compressing store.
    #[inline(always)]
    unsafe fn compress(self, mask: M, p: *mut f32) -> usize {
        unsafe { _mm512_storeu_ps(p, _mm512_maskz_compress_ps(mask.0, self.0)) };
        mask.0.count_ones() as usize
    }

    #[inline(always)]
    fn not(mask: M) -> M {
        M(!mask.0)
    }

    #[inline(always)]
    fn bitmask(mask: M) -> u32 {
        mask.0 as u32
    }

    #[inline(always)]
    fn first(n: usize) -> M {
        M(((1u32 << n) - 1) as u16)
    }

    #[inline(always)]
    fn widen(self, acc: &mut [f64]) {
        assert!(acc.len() >= 16);
        unsafe {
            let (lo, hi) = halves(self.0);
            let p = acc.as_mut_ptr();
            _mm512_storeu_pd(p, _mm512_add_pd(_mm512_loadu_pd(p), lo));
            _mm512_storeu_pd(p.add(8), _mm512_add_pd(_mm512_loadu_pd(p.add(8)), hi));
        }
    }
}

/// The permutation as its 256 bytes in four registers.
pub struct Bytes([__m512i; 4]);

impl Bytes {
    /// The low byte of each pair is the permutation entry.
    #[inline(always)]
    pub(crate) fn new(fbm: &Fbm) -> Bytes {
        let words = fbm.pairs().as_ptr() as *const __m512i;
        Bytes(core::array::from_fn(|k| unsafe {
            let lo = _mm512_cvtepi16_epi8(_mm512_loadu_si512(words.add(2 * k)));
            let hi = _mm512_cvtepi16_epi8(_mm512_loadu_si512(words.add(2 * k + 1)));
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
        }))
    }
}

impl Lattice<V> for Bytes {
    /// Byte permutes in place of a gather. Byte 0 of each lane asks for
    /// entry i and byte 1 for entry i + 1, wrapping at 256 on its own, so one
    /// lookup brings the pair. The permutes pick from the low or the high
    /// half by bit 6 and the blend by bit 7.
    #[inline(always)]
    fn pair(&self, i: B) -> (B, B) {
        unsafe {
            let i = _mm512_and_si512(i.0, _mm512_set1_epi32(255));
            let at = _mm512_add_epi8(_mm512_or_si512(i, _mm512_slli_epi32::<8>(i)), _mm512_set1_epi32(0x0100));
            let low = _mm512_permutex2var_epi8(self.0[0], at, self.0[1]);
            let high = _mm512_permutex2var_epi8(self.0[2], at, self.0[3]);
            let v = _mm512_mask_blend_epi8(_mm512_movepi8_mask(at), low, high);
            let byte = _mm512_set1_epi32(255);
            (B(_mm512_and_si512(v, byte)), B(_mm512_and_si512(_mm512_srli_epi32::<8>(v), byte)))
        }
    }

    #[inline(always)]
    fn grad(&self, hash: B, x: V, y: V, z: V) -> V {
        grad(hash, x, y, z)
    }
}

impl Lattice<V> for Pairs {
    /// A 32 bit gather at the pair's byte offset, which brings the pair in
    /// its low half.
    #[inline(always)]
    fn pair(&self, i: B) -> (B, B) {
        unsafe {
            let at = _mm512_slli_epi32::<1>(_mm512_and_si512(i.0, _mm512_set1_epi32(255)));
            let v = _mm512_i32gather_epi32::<1>(at, self.0.as_ptr().cast());
            let byte = _mm512_set1_epi32(255);
            (B(_mm512_and_si512(v, byte)), B(_mm512_and_si512(_mm512_srli_epi32::<8>(v), byte)))
        }
    }

    #[inline(always)]
    fn grad(&self, hash: B, x: V, y: V, z: V) -> V {
        grad(hash, x, y, z)
    }
}

/// The gradient table's three columns each fill one register, so a lookup
/// is a permute, which reads the low four bits of the hash.
#[inline(always)]
fn grad(hash: B, x: V, y: V, z: V) -> V {
    unsafe {
        let gx = V(_mm512_permutexvar_ps(hash.0, _mm512_loadu_ps(COLUMNS[0].as_ptr())));
        let gy = V(_mm512_permutexvar_ps(hash.0, _mm512_loadu_ps(COLUMNS[1].as_ptr())));
        let gz = V(_mm512_permutexvar_ps(hash.0, _mm512_loadu_ps(COLUMNS[2].as_ptr())));
        gx.fma(x, gy.fma(y, gz * z))
    }
}

/// `GRADS` by column.
static COLUMNS: [[f32; 16]; 3] = {
    let mut c = [[0.0; 16]; 3];
    let mut i = 0;
    while i < 16 {
        let mut k = 0;
        while k < 3 {
            c[k][i] = GRADS[i].0[k];
            k += 1;
        }
        i += 1;
    }
    c
};
