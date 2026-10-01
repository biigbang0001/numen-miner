use core::arch::{asm, global_asm};

// Rust cannot place a static in the shared state space, so shared memory is
// one dynamic window the host sizes at launch, reached by byte offset.
global_asm!(".extern .shared .align 16 .b8 smem[];");


macro_rules! special {
    ($name:ident, $reg:literal) => {
        #[inline(always)]
        pub fn $name() -> u32 {
            let v: u32;
            unsafe {
                asm!(concat!("mov.u32 {}, ", $reg, ";"), out(reg32) v, options(pure, nomem, nostack));
            }
            v
        }
    };
}

special!(tid, "%tid.x");
special!(ctaid, "%ctaid.x");
special!(nctaid, "%nctaid.x");
special!(laneid, "%laneid");
special!(lanemask_lt, "%lanemask_lt");

/// Byte address of `offset` in the shared window.
#[inline(always)]
fn window(offset: u32) -> u32 {
    let v: u32;
    unsafe { asm!("mov.u32 {}, smem;", out(reg32) v, options(pure, nomem, nostack)) };
    v + offset
}

/// Generic pointer to byte `offset` of the shared window, for handing shared
/// memory to code that takes an ordinary reference.
#[inline(always)]
pub fn shared<T>(offset: u32) -> *mut T {
    let v: u64;
    unsafe { asm!("cvta.shared.u64 {}, smem;", out(reg64) v, options(pure, nomem, nostack)) };
    (v + offset as u64) as *mut T
}

/// Block wide barrier, which also orders shared memory.
#[inline(always)]
pub fn sync() {
    unsafe { asm!("bar.sync 0;", options(nostack)) };
}

/// Warp barrier. Since Volta a warp's lanes can drift apart, so a lane reading
/// what another lane of its warp wrote to memory waits here first.
#[inline(always)]
pub fn syncwarp() {
    unsafe { asm!("bar.warp.sync -1;", options(nostack)) };
}

/// Butterfly exchange across the whole warp.
///
/// Every lane has to reach it together, so it must not read as pure. A pure
/// asm is one LLVM may sink into the branch that consumes its result, which
/// leaves the other lanes of the warp never arriving.
#[inline(always)]
pub fn shfl_xor(v: u32, mask: u32) -> u32 {
    let out: u32;
    unsafe {
        asm!("shfl.sync.bfly.b32 {}, {}, {}, 31, -1;", out(reg32) out, in(reg32) v, in(reg32) mask,
             options(nomem, nostack));
    }
    out
}

/// Value of `v` on lane `src`, which may differ per lane.
#[inline(always)]
pub fn shfl_idx(v: u32, src: u32) -> u32 {
    let out: u32;
    unsafe {
        asm!("shfl.sync.idx.b32 {}, {}, {}, 31, -1;", out(reg32) out, in(reg32) v, in(reg32) src,
             options(nomem, nostack));
    }
    out
}

/// Bit `l` set when lane `l` passes `p`.
#[inline(always)]
pub fn ballot(p: bool) -> u32 {
    let out: u32;
    unsafe {
        asm!("{{ .reg .pred bp; setp.ne.u32 bp, {}, 0; vote.sync.ballot.b32 {}, bp, -1; }}",
             in(reg32) p as u32, out(reg32) out, options(nomem, nostack));
    }
    out
}

macro_rules! access {
    ($ld:ident, $st:ident, $t:ty, $v:ty, $reg:ident, $kind:literal) => {
        #[allow(dead_code)]
        #[inline(always)]
        pub unsafe fn $ld(offset: u32) -> $t {
            let v: $v;
            unsafe {
                asm!(concat!("ld.shared.", $kind, " {}, [{}];"), out($reg) v, in(reg32) window(offset),
                     options(pure, readonly, nostack));
            }
            v as $t
        }

        #[allow(dead_code)]
        #[inline(always)]
        pub unsafe fn $st(offset: u32, v: $t) {
            unsafe {
                asm!(concat!("st.shared.", $kind, " [{}], {};"), in(reg32) window(offset), in($reg) v as $v,
                     options(nostack));
            }
        }
    };
}

access!(ld_f32, st_f32, f32, f32, reg32, "f32");
access!(ld_f64, st_f64, f64, f64, reg64, "f64");
access!(ld_u8, st_u8, u8, u16, reg16, "u8");
access!(ld_u16, st_u16, u16, u16, reg16, "u16");
access!(ld_u32, st_u32, u32, u32, reg32, "u32");
access!(ld_u64, st_u64, u64, u64, reg64, "u64");

#[inline(always)]
pub unsafe fn ld_v4(offset: u32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    unsafe {
        asm!("ld.shared.v4.f32 {{{}, {}, {}, {}}}, [{}];", out(reg32) a, out(reg32) b, out(reg32) c,
             out(reg32) d, in(reg32) window(offset), options(pure, readonly, nostack));
    }
    [a, b, c, d]
}

/// Reserve one output slot. Returns the index this thread owns.
///
/// Generic addressing on purpose. Kernel parameters arrive as generic pointers
/// and only the ones the compiler dereferences itself get a `cvta.to.global`,
/// so an `atom.global` here would be operating on an address outside the
/// window that qualifier promises.
#[inline(always)]
pub unsafe fn reserve(counter: *mut u32) -> u32 {
    let slot: u32;
    unsafe {
        asm!("atom.add.u32 {}, [{}], 1;", out(reg32) slot, in(reg64) counter, options(nostack));
    }
    slot
}

/// 1 / x from the hardware's approximate reciprocal and one Newton step,
/// within about half an ulp and without the correctly rounded divide's
/// dozen instructions.
#[inline(always)]
pub fn recip(x: f32) -> f32 {
    let r: f32;
    unsafe { asm!("rcp.approx.ftz.f32 {}, {};", out(reg32) r, in(reg32) x, options(pure, nomem, nostack)) };
    crate::intrinsics::fmaf(r, crate::intrinsics::fmaf(-x, r, 1.0), r)
}

/// The hardware's approximate log2, about 2^-22 off.
#[inline(always)]
pub fn lg2(x: f32) -> f32 {
    let v: f32;
    unsafe { asm!("lg2.approx.f32 {}, {};", out(reg32) v, in(reg32) x, options(pure, nomem, nostack)) };
    v
}

/// The hardware's approximate 2^x, about 2^-22 off.
#[inline(always)]
pub fn ex2(x: f32) -> f32 {
    let v: f32;
    unsafe { asm!("ex2.approx.f32 {}, {};", out(reg32) v, in(reg32) x, options(pure, nomem, nostack)) };
    v
}
