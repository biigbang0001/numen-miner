use core::ops::{Add, BitAnd, Div, Mul, Neg, Sub};

use crate::split::Split;

pub trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// A truth value per lane.
    type Mask: Copy + BitAnd<Output = Self::Mask>;

    /// A u32 per lane, a lattice index or the bits of a value.
    type Bits: Copy + Add<Output = Self::Bits> + BitAnd<Output = Self::Bits>;

    fn splat(x: f32) -> Self;
    fn bits(x: u32) -> Self::Bits;

    /// `self * b + c` rounded once.
    fn fma(self, b: Self, c: Self) -> Self;
    fn floor(self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, lo: f32, hi: f32) -> Self;

    fn lt(self, other: Self) -> Self::Mask;
    fn le(self, other: Self) -> Self::Mask;
    fn gt(self, other: Self) -> Self::Mask;
    fn select(mask: Self::Mask, yes: Self, no: Self) -> Self;
    /// Whether any lane of `mask` is set, the test that lets a vector skip
    /// work none of its lanes needs.
    fn any(mask: Self::Mask) -> bool;

    fn to_bits(self) -> Self::Bits;

    /// The lanes added up in f64.
    fn sum_f64(self) -> f64;

    /// `self` to the power `y` for `self` in (0, 1].
    fn pow(self, y: Split) -> Self;
}

impl Real for f32 {
    type Mask = bool;
    type Bits = u32;

    #[inline(always)]
    fn splat(x: f32) -> f32 {
        x
    }

    #[inline(always)]
    fn bits(x: u32) -> u32 {
        x
    }

    #[inline(always)]
    fn fma(self, b: f32, c: f32) -> f32 {
        crate::intrinsics::fmaf(self, b, c)
    }

    #[inline(always)]
    fn floor(self) -> f32 {
        crate::intrinsics::floorf(self)
    }

    #[inline(always)]
    fn sqrt(self) -> f32 {
        crate::intrinsics::sqrtf(self)
    }

    #[inline(always)]
    fn abs(self) -> f32 {
        f32::abs(self)
    }

    #[inline(always)]
    fn max(self, other: f32) -> f32 {
        f32::max(self, other)
    }

    #[inline(always)]
    fn clamp(self, lo: f32, hi: f32) -> f32 {
        f32::clamp(self, lo, hi)
    }

    #[inline(always)]
    fn lt(self, other: f32) -> bool {
        self < other
    }

    #[inline(always)]
    fn le(self, other: f32) -> bool {
        self <= other
    }

    #[inline(always)]
    fn gt(self, other: f32) -> bool {
        self > other
    }

    #[inline(always)]
    fn select(mask: bool, yes: f32, no: f32) -> f32 {
        if mask { yes } else { no }
    }

    #[inline(always)]
    fn any(mask: bool) -> bool {
        mask
    }

    #[inline(always)]
    fn to_bits(self) -> u32 {
        f32::to_bits(self)
    }

    #[inline(always)]
    fn sum_f64(self) -> f64 {
        self as f64
    }

    /// The hardware's approximate log2 and exp2. Their error leans slightly
    /// one way, which costs about 0.5% of retention through the eigenvalue
    /// ratios, but the pair replaces some 45 instructions per lobe. Measured
    /// on an RTX 5070 Ti, effective hash power comes out 2% ahead.
    #[cfg(any(target_arch = "nvptx64", target_arch = "amdgpu"))]
    #[inline(always)]
    fn pow(self, y: Split) -> f32 {
        crate::simt::ex2(y.scale(crate::simt::lg2(self)))
    }

    #[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
    #[inline(always)]
    fn pow(self, y: Split) -> f32 {
        crate::cpu::pow(self, y)
    }
}
