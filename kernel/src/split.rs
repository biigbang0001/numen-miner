use core::ops::Mul;

use crate::real::Real;

/// An f64 carried as two f32, the rounded value and its rounding error.
///
/// Per nonce constants that scale a whole surface go through this. A plain f32
/// would push every vertex the same way by the same rounding error, and no
/// amount of averaging over the surface takes that back out. A blank direction
/// travels the same way, as a pair of vectors on a CPU.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Split<T = f32> {
    pub hi: T,
    pub lo: T,
}

impl Split {
    pub const fn new(x: f64) -> Split {
        let hi = x as f32;
        Split { hi, lo: (x - hi as f64) as f32 }
    }

    /// `self * x` rounded once, which keeps the error in `self` out of the
    /// result.
    #[inline(always)]
    pub fn scale<T: Real>(self, x: T) -> T {
        T::splat(self.hi).fma(x, T::splat(self.lo) * x)
    }
}

/// Product with a split constant, good to about 2^-46.
impl<T: Real> Mul<Split> for Split<T> {
    type Output = Split<T>;

    #[inline(always)]
    fn mul(self, other: Split) -> Split<T> {
        let (hi, lo) = (T::splat(other.hi), T::splat(other.lo));
        let p = self.hi * hi;
        Split { hi: p, lo: self.hi.fma(hi, -p) + (self.hi * lo + self.lo * hi) }
    }
}
