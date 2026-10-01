use core::arch::x86_64::*;
use core::ops::{Add, BitAnd, Div, Mul, Neg, Sub};

use super::{LN_2, Lanes, Pairs, exp, log2};
use crate::perlin::{GRADS, Lattice};
use crate::real::Real;
use crate::split::Split;

#[derive(Clone, Copy)]
pub struct V(__m256);

#[derive(Clone, Copy)]
pub struct B(__m256i);

/// All ones in a lane that is set.
#[derive(Clone, Copy)]
pub struct M(__m256);

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

binary!(V, Add, add, _mm256_add_ps);
binary!(V, Sub, sub, _mm256_sub_ps);
binary!(V, Mul, mul, _mm256_mul_ps);
binary!(V, Div, div, _mm256_div_ps);
binary!(B, Add, add, _mm256_add_epi32);
binary!(B, BitAnd, bitand, _mm256_and_si256);
binary!(M, BitAnd, bitand, _mm256_and_ps);

impl Neg for V {
    type Output = V;

    #[inline(always)]
    fn neg(self) -> V {
        unsafe { V(_mm256_xor_ps(self.0, _mm256_set1_ps(-0.0))) }
    }
}

impl Real for V {
    type Mask = M;
    type Bits = B;

    #[inline(always)]
    fn splat(x: f32) -> V {
        unsafe { V(_mm256_set1_ps(x)) }
    }

    #[inline(always)]
    fn bits(x: u32) -> B {
        unsafe { B(_mm256_set1_epi32(x as i32)) }
    }

    #[inline(always)]
    fn fma(self, b: V, c: V) -> V {
        unsafe { V(_mm256_fmadd_ps(self.0, b.0, c.0)) }
    }

    #[inline(always)]
    fn floor(self) -> V {
        unsafe { V(_mm256_floor_ps(self.0)) }
    }

    #[inline(always)]
    fn sqrt(self) -> V {
        unsafe { V(_mm256_sqrt_ps(self.0)) }
    }

    #[inline(always)]
    fn abs(self) -> V {
        unsafe { V(_mm256_andnot_ps(_mm256_set1_ps(-0.0), self.0)) }
    }

    #[inline(always)]
    fn max(self, o: V) -> V {
        unsafe { V(_mm256_max_ps(self.0, o.0)) }
    }

    #[inline(always)]
    fn clamp(self, lo: f32, hi: f32) -> V {
        unsafe { V(_mm256_max_ps(_mm256_min_ps(self.0, _mm256_set1_ps(hi)), _mm256_set1_ps(lo))) }
    }

    #[inline(always)]
    fn lt(self, o: V) -> M {
        unsafe { M(_mm256_cmp_ps::<_CMP_LT_OQ>(self.0, o.0)) }
    }

    #[inline(always)]
    fn le(self, o: V) -> M {
        unsafe { M(_mm256_cmp_ps::<_CMP_LE_OQ>(self.0, o.0)) }
    }

    #[inline(always)]
    fn gt(self, o: V) -> M {
        unsafe { M(_mm256_cmp_ps::<_CMP_GT_OQ>(self.0, o.0)) }
    }

    #[inline(always)]
    fn select(mask: M, yes: V, no: V) -> V {
        unsafe { V(_mm256_blendv_ps(no.0, yes.0, mask.0)) }
    }

    #[inline(always)]
    fn any(mask: M) -> bool {
        unsafe { _mm256_movemask_ps(mask.0) != 0 }
    }

    #[inline(always)]
    fn to_bits(self) -> B {
        unsafe { B(_mm256_castps_si256(self.0)) }
    }

    #[inline(always)]
    fn sum_f64(self) -> f64 {
        let mut lanes = [0.0; 8];
        self.widen(&mut lanes);
        lanes.iter().sum()
    }

    /// The scalar series lane by lane, bit for bit.
    #[inline(always)]
    fn pow(self, y: Split) -> V {
        unsafe {
            let bits = _mm256_castps_si256(self.0);
            let e = _mm256_sub_epi32(_mm256_srli_epi32::<23>(bits), _mm256_set1_epi32(127));
            let m = V(_mm256_castsi256_ps(_mm256_or_si256(
                _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
                _mm256_set1_epi32(0x3f80_0000),
            )));
            let up = m.gt(V::splat(core::f32::consts::SQRT_2));
            let m = V::select(up, m * V::splat(0.5), m);
            // A set lane is all ones, minus one, so subtracting it adds one.
            let e = V(_mm256_cvtepi32_ps(_mm256_sub_epi32(e, _mm256_castps_si256(up.0))));
            let t = y.scale(log2(m, e));
            let n = (t + V::splat(0.5)).floor();
            let scale = _mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvttps_epi32(n.0), _mm256_set1_epi32(127)));
            let p = exp(LN_2.scale(t - n)) * V(_mm256_castsi256_ps(scale));
            V::select(t.lt(V::splat(-125.0)), V::splat(0.0), p)
        }
    }
}

impl Lanes for V {
    const N: usize = 8;

    #[inline(always)]
    unsafe fn load(p: *const f32) -> V {
        unsafe { V(_mm256_loadu_ps(p)) }
    }

    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        unsafe { _mm256_storeu_ps(p, self.0) };
    }

    #[inline(always)]
    unsafe fn load_bits(p: *const u32) -> B {
        unsafe { B(_mm256_loadu_si256(p.cast())) }
    }

    #[inline(always)]
    unsafe fn gather(base: *const f32, idx: B) -> V {
        unsafe { V(_mm256_i32gather_ps::<4>(base, idx.0)) }
    }

    /// No compress instruction, so a table turns the mask into the permute
    /// that packs its lanes to the front.
    #[inline(always)]
    unsafe fn compress(self, mask: M, p: *mut f32) -> usize {
        let bits = Self::bitmask(mask) as usize;
        unsafe {
            let order = _mm256_loadu_si256(PACK[bits].as_ptr().cast());
            _mm256_storeu_ps(p, _mm256_permutevar8x32_ps(self.0, order));
        }
        bits.count_ones() as usize
    }

    #[inline(always)]
    fn not(mask: M) -> M {
        unsafe { M(_mm256_xor_ps(mask.0, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))) }
    }

    #[inline(always)]
    fn bitmask(mask: M) -> u32 {
        unsafe { _mm256_movemask_ps(mask.0) as u32 }
    }

    #[inline(always)]
    fn first(n: usize) -> M {
        unsafe {
            let lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            M(_mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(n as i32), lanes)))
        }
    }

    #[inline(always)]
    fn widen(self, acc: &mut [f64]) {
        assert!(acc.len() >= 8);
        unsafe {
            let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(self.0));
            let hi = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(self.0));
            let p = acc.as_mut_ptr();
            _mm256_storeu_pd(p, _mm256_add_pd(_mm256_loadu_pd(p), lo));
            _mm256_storeu_pd(p.add(4), _mm256_add_pd(_mm256_loadu_pd(p.add(4)), hi));
        }
    }
}

impl Lattice<V> for Pairs {
    /// A 32 bit gather at the pair's byte offset, which brings the pair in
    /// its low half.
    #[inline(always)]
    fn pair(&self, i: B) -> (B, B) {
        unsafe {
            let at = _mm256_slli_epi32::<1>(_mm256_and_si256(i.0, _mm256_set1_epi32(255)));
            let v = _mm256_i32gather_epi32::<1>(self.0.as_ptr().cast(), at);
            let byte = _mm256_set1_epi32(255);
            (B(_mm256_and_si256(v, byte)), B(_mm256_and_si256(_mm256_srli_epi32::<8>(v), byte)))
        }
    }

    /// Each column of the gradient table fills two registers. A permute
    /// reads the low three bits of the hash in both, and bit three, moved up
    /// to the sign, picks between them.
    #[inline(always)]
    fn grad(&self, hash: B, x: V, y: V, z: V) -> V {
        unsafe {
            let high = _mm256_castsi256_ps(_mm256_slli_epi32::<28>(hash.0));
            let column = |k: usize| {
                let lo = _mm256_permutevar8x32_ps(_mm256_loadu_ps(COLUMNS[k].as_ptr()), hash.0);
                let hi = _mm256_permutevar8x32_ps(_mm256_loadu_ps(COLUMNS[k].as_ptr().add(8)), hash.0);
                V(_mm256_blendv_ps(lo, hi, high))
            };
            column(0).fma(x, column(1).fma(y, column(2) * z))
        }
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

/// For each 8 bit mask, the lanes it sets in order, which packs them to the
/// front under `_mm256_permutevar8x32_ps`.
static PACK: [[u32; 8]; 256] = {
    let mut t = [[0; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut l, mut n) = (0, 0);
        while l < 8 {
            if m & 1 << l != 0 {
                t[m][n] = l as u32;
                n += 1;
            }
            l += 1;
        }
        m += 1;
    }
    t
};
