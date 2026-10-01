#[cfg(target_arch = "nvptx64")]
use core::arch::asm;

/// IEEE correctly rounded f32 square root.
///
/// IEEE 754 pins sqrt to a single correct answer, so the GPU instructions and
/// libm cannot disagree. Every sample and every crater distance takes one.
#[inline(always)]
pub fn sqrtf(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let out: f32;
        unsafe {
            asm!("sqrt.rn.f32 {}, {};", out(reg32) out, in(reg32) x,
                 options(pure, nomem, nostack));
        }
        out
    }
    #[cfg(target_arch = "amdgpu")]
    {
        core::f32::math::sqrt(x)
    }
    #[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
    {
        libm::sqrtf(x)
    }
}

/// Round an f32 toward negative infinity.
///
/// Exact in both directions, so `cvt.rmi` and libm agree. The Perlin lattice
/// alone takes 15 of these per vertex.
#[inline(always)]
pub fn floorf(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let out: f32;
        unsafe {
            asm!("cvt.rmi.f32.f32 {}, {};", out(reg32) out, in(reg32) x,
                 options(pure, nomem, nostack));
        }
        out
    }
    #[cfg(target_arch = "amdgpu")]
    {
        core::f32::math::floor(x)
    }
    #[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
    {
        libm::floorf(x)
    }
}

/// Round an f32 toward positive infinity. Runs once per face in each sample
/// pass to size the subdivision.
#[inline(always)]
pub fn ceilf(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let out: f32;
        unsafe {
            asm!("cvt.rpi.f32.f32 {}, {};", out(reg32) out, in(reg32) x,
                 options(pure, nomem, nostack));
        }
        out
    }
    #[cfg(target_arch = "amdgpu")]
    {
        core::f32::math::ceil(x)
    }
    #[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
    {
        libm::ceilf(x)
    }
}

/// `a * b + c` rounded once.
///
/// IEEE 754 pins the fused multiply add to one correctly rounded result, just
/// like the basic operations, so the GPU instructions and libm agree. On a
/// host, libm picks the FMA instruction at run time and falls back to an exact
/// software path.
#[inline(always)]
pub fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let out: f32;
        unsafe {
            asm!("fma.rn.f32 {}, {}, {}, {};", out(reg32) out, in(reg32) a, in(reg32) b,
                 in(reg32) c, options(pure, nomem, nostack));
        }
        out
    }
    #[cfg(target_arch = "amdgpu")]
    {
        core::f32::math::mul_add(a, b, c)
    }
    #[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
    {
        libm::fmaf(a, b, c)
    }
}

/// IEEE correctly rounded f64 square root. A few dozen per nonce, in the
/// parameter draw and the final features.
#[inline(always)]
pub fn sqrt(x: f64) -> f64 {
    #[cfg(target_arch = "nvptx64")]
    {
        let out: f64;
        unsafe {
            asm!("sqrt.rn.f64 {}, {};", out(reg64) out, in(reg64) x,
                 options(pure, nomem, nostack));
        }
        out
    }
    #[cfg(target_arch = "amdgpu")]
    {
        core::f64::math::sqrt(x)
    }
    #[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
    {
        libm::sqrt(x)
    }
}

/// Round half away from zero.
///
/// PTX has no instruction for this rounding mode, `cvt.rni` rounds half to even
/// and would land on a different bucket edge. Software both sides. It runs 23
/// times per nonce so the cost is noise.
#[inline(always)]
pub fn round(x: f64) -> f64 {
    libm::round(x)
}

/// Cube root. One call per nonce, in the volume normalization.
#[inline(always)]
pub fn cbrt(x: f64) -> f64 {
    libm::cbrt(x)
}

/// Arc cosine. One call per nonce, in the eigenvalue solve.
#[inline(always)]
pub fn acos(x: f64) -> f64 {
    libm::acos(x)
}

/// Cosine. Two calls per nonce, in the eigenvalue solve.
#[inline(always)]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// Sine and cosine from one range reduction, bit for bit what `sin` and
/// `cos` return. One call per random unit vector, so 15 per nonce.
#[inline(always)]
pub fn sincos(x: f64) -> (f64, f64) {
    libm::sincos(x)
}
