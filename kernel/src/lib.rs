#![cfg_attr(any(target_arch = "nvptx64", target_arch = "amdgpu"), no_std)]
#![cfg_attr(target_arch = "nvptx64", feature(abi_gpu_kernel, asm_experimental_arch))]
#![cfg_attr(
    target_arch = "amdgpu",
    feature(
        abi_gpu_kernel,
        abi_unadjusted,
        asm_experimental_arch,
        core_float_math,
        core_intrinsics,
        gpu_intrinsics,
        gpu_launch_sized_workgroup_mem,
        link_llvm_intrinsics,
        stdarch_amdgpu
    ),
    allow(internal_features)
)]

#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
extern crate alloc;

#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
pub mod cpu;
#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
pub mod host;
pub mod coop;
pub mod intrinsics;
pub mod perlin;
pub mod pipeline;
pub mod real;
#[cfg(target_arch = "nvptx64")]
#[path = "ptx.rs"]
mod simt;
#[cfg(target_arch = "amdgpu")]
#[path = "gcn.rs"]
mod simt;
pub mod rng;
pub mod sculpt;
pub mod split;
pub mod topology;
pub mod work;

pub use work::{Nonce, meets};

#[cfg(any(target_arch = "nvptx64", target_arch = "amdgpu"))]
mod gpu {
    use core::panic::PanicInfo;

    /// Abort the launch rather than spin.
    ///
    /// The bounds checks the compiler leaves in the hot loops all branch here.
    /// Spinning would hang the warp, so the kernel would never retire and the
    /// host would block forever on the result readback with nothing logged.
    /// A trap surfaces the same event as a launch failure the miner can report.
    #[cfg(target_arch = "nvptx64")]
    #[panic_handler]
    fn panic(_: &PanicInfo) -> ! {
        unsafe { core::arch::asm!("trap;", options(noreturn)) }
    }

    #[cfg(target_arch = "amdgpu")]
    #[panic_handler]
    fn panic(_: &PanicInfo) -> ! {
        core::intrinsics::abort()
    }
}
