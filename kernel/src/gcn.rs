use core::arch::amdgpu;
use core::intrinsics::gpu::{amdgpu_dispatch_ptr, gpu_launch_sized_workgroup_mem};
use core::sync::atomic::{AtomicU32, Ordering, fence};

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.amdgcn.rcp.f32"]
    safe fn rcp(x: f32) -> f32;
    #[link_name = "llvm.amdgcn.log.f32"]
    safe fn log(x: f32) -> f32;
    #[link_name = "llvm.amdgcn.exp2.f32"]
    safe fn exp(x: f32) -> f32;
}

#[inline(always)]
pub fn tid() -> u32 {
    amdgpu::workitem_id_x()
}

#[inline(always)]
pub fn ctaid() -> u32 {
    amdgpu::workgroup_id_x()
}

/// Blocks in the launch, from the dispatch packet, which gives the grid in
/// threads.
#[inline(always)]
pub fn nctaid() -> u32 {
    let packet = amdgpu_dispatch_ptr() as *const u8;
    unsafe { *(packet.add(12) as *const u32) / *(packet.add(4) as *const u16) as u32 }
}

#[inline(always)]
pub fn laneid() -> u32 {
    tid() & 31
}

#[inline(always)]
pub fn lanemask_lt() -> u32 {
    (1 << laneid()) - 1
}

/// This lane's place in its hardware wave. A warp is 32 lanes here too, so an
/// RDNA wave is one warp and a GCN wave holds two, told apart by `& 32`.
#[inline(always)]
fn wave_lane() -> u32 {
    tid() & (amdgpu::wavefrontsize() - 1)
}

/// Block wide barrier. The release fence waits out this wave's LDS writes
/// before it arrives, and the empty asm keeps the compiler from moving shared
/// reads above the barrier.
#[inline(always)]
pub fn sync() {
    fence(Ordering::Release);
    amdgpu::s_barrier();
    unsafe { core::arch::asm!("", options(nostack)) };
}

/// A wave runs its lanes in lock step and its memory accesses in order, so
/// only the compiler needs holding back.
#[inline(always)]
pub fn syncwarp() {
    unsafe { core::arch::asm!("", options(nostack)) };
}

#[inline(always)]
pub fn shfl_xor(v: u32, mask: u32) -> u32 {
    unsafe { amdgpu::ds_bpermute(4 * (wave_lane() ^ mask), v) }
}

#[inline(always)]
pub fn shfl_idx(v: u32, src: u32) -> u32 {
    unsafe { amdgpu::ds_bpermute(4 * (wave_lane() & 32 | src), v) }
}

/// Bit `l` set when lane `l` of this warp passes `p`.
#[inline(always)]
pub fn ballot(p: bool) -> u32 {
    (amdgpu::ballot(p) >> (wave_lane() & 32)) as u32
}

#[repr(C, align(16))]
struct Line([u8; 16]);

#[inline(always)]
fn window(offset: u32) -> *mut u8 {
    unsafe { (gpu_launch_sized_workgroup_mem::<Line>() as *mut u8).add(offset as usize) }
}

/// Pointer to byte `offset` of the shared window, for handing shared memory
/// to code that takes an ordinary reference.
#[inline(always)]
pub fn shared<T>(offset: u32) -> *mut T {
    window(offset) as *mut T
}

macro_rules! access {
    ($ld:ident, $st:ident, $t:ty) => {
        #[allow(dead_code)]
        #[inline(always)]
        pub unsafe fn $ld(offset: u32) -> $t {
            unsafe { *shared::<$t>(offset) }
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub unsafe fn $st(offset: u32, v: $t) {
            unsafe { *shared::<$t>(offset) = v };
        }
    };
}

access!(ld_f32, st_f32, f32);
access!(ld_f64, st_f64, f64);
access!(ld_u8, st_u8, u8);
access!(ld_u16, st_u16, u16);
access!(ld_u32, st_u32, u32);
access!(ld_u64, st_u64, u64);

#[repr(C, align(16))]
struct Quad([f32; 4]);

#[inline(always)]
pub unsafe fn ld_v4(offset: u32) -> [f32; 4] {
    unsafe { (*shared::<Quad>(offset)).0 }
}

/// Reserve one output slot. Returns the index this thread owns.
#[inline(always)]
pub unsafe fn reserve(counter: *mut u32) -> u32 {
    unsafe { AtomicU32::from_ptr(counter) }.fetch_add(1, Ordering::Relaxed)
}

/// 1 / x from the hardware's reciprocal, good to 1 ulp, and one Newton step.
#[inline(always)]
pub fn recip(x: f32) -> f32 {
    let r = rcp(x);
    crate::intrinsics::fmaf(r, crate::intrinsics::fmaf(-x, r, 1.0), r)
}

/// The hardware's log2, good to about 1 ulp.
#[inline(always)]
pub fn lg2(x: f32) -> f32 {
    log(x)
}

/// The hardware's 2^x, good to about 1 ulp.
#[inline(always)]
pub fn ex2(x: f32) -> f32 {
    exp(x)
}
