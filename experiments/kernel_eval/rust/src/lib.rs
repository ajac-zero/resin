//! Experimental in-process bridge. Caller owns all GPU storage and the stream.
//! Handles are single-threaded. Keep buffers/context/stream alive, with no
//! conflicting accesses until stream completion. Do not destroy during capture.
//! This is NOT a sandbox for adversarial candidate host code.
use cutile::prelude::*;
use half::bf16;
use std::ffi::c_void;
mod kernels;

struct Handle {
    stream: Arc<cuda_core::Stream>,
    g: Tensor<bf16>,
    u: Tensor<bf16>,
    out: Tensor<bf16>,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_create(g: u64, u: u64, out: u64, n: i32, stream: *mut c_void) -> *mut c_void {
    if n <= 0 || n > 1048576 || g == 0 || u == 0 || out == 0 { return std::ptr::null_mut(); }
    std::panic::catch_unwind(|| unsafe {
        let device = Device::new(0).ok()?;
        let stream = cuda_core::Stream::borrow_raw(stream, &device);
        let borrow = |ptr| Tensor::<bf16>::borrow_raw_parts(ptr, 0, vec![n], vec![1]);
        Some(Box::into_raw(Box::new(Handle {stream,g:borrow(g),u:borrow(u),out:borrow(out)})) as *mut c_void)
    }).ok().flatten().unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_launch(ptr: *mut c_void, block: i32) -> i32 {
    if ptr.is_null() { return -1; }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let h = &mut *(ptr as *mut Handle);
        macro_rules! run {
            ($b:literal) => {
                kernels::kernels::gelu_mul(&h.g,&h.u,(&mut h.out).partition([$b]))
                    .generics(vec![stringify!($b).into()]).async_on(&h.stream).map(|_| ())
            }
        }
        let result = match block {256 => run!(256), 512 => run!(512), 1024 => run!(1024), _ => return -2};
        match result {Ok(()) => 0, Err(e) => {eprintln!("resin launch: {e}"); -3}}
    })).unwrap_or(-4)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn resin_destroy(ptr: *mut c_void) -> i32 {
    if ptr.is_null() {return -1;}
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let h = Box::from_raw(ptr as *mut Handle);
        if h.stream.synchronize().is_err() {
            // Never release metadata before completion on an error path.
            Box::leak(h);
            return -3;
        }
        drop(h);
        0
    })).unwrap_or(-4)
}
