//! osrender: renders a vertical clip in a single pass without taking the
//! frames off the GPU.
//!
//!     osrender <input.mp4> <plan.json> <output.mp4>
//!     osrender burn <spec.json>        (hook + captions + mark, see burn.rs)
//!
//! NVDEC decodes into CUDA frames, the kernels in `kernels.cu` compose each
//! output frame from the plan's layers (crop + scale, optionally blurred) and
//! an optional image overlay, and NVENC encodes straight from device memory.
//! The output carries video only; the caller muxes the audio back in.
//!
//! The plan is written by `gpu_render.py`. Any error exits non-zero so the
//! caller can fall back to the ffmpeg path.

use ffmpeg_sys_next as ff;
use serde::Deserialize;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr::{null, null_mut};

extern "C" {
    fn _exit(code: c_int) -> !;
}

// ------------------------------------------------------------------ plan

#[derive(Deserialize)]
struct Plan {
    out_w: i32,
    out_h: i32,
    fps: f64,
    #[serde(default)]
    encoder: Vec<(String, String)>,
    scenes: Vec<Scene>,
    overlay: Option<ImageOverlay>,
    #[serde(default)]
    threads: i32,
}

#[derive(Deserialize)]
struct Scene {
    ss: f64,
    dur: f64,
    layers: Vec<Layer>,
}

/// Source rectangle `src` (or one per frame in `boxes`), both `[x, y, w, h]`
/// in source pixels, drawn into `dst` `[x, y, w, h]` of the output frame.
/// With `blur`, the source is first reduced to `blur.w x blur.h`, blurred
/// there with `blur.sigma` and then scaled into `dst`.
#[derive(Deserialize)]
struct Layer {
    src: Option<[f64; 4]>,
    boxes: Option<Vec<[f64; 4]>>,
    dst: [i32; 4],
    blur: Option<Blur>,
}

#[derive(Deserialize)]
struct Blur {
    sigma: f32,
    w: i32,
    h: i32,
}

#[derive(Deserialize)]
struct ImageOverlay {
    path: String,
    x: i32,
    y: i32,
    w: i32,
    opacity: f32,
}

// ------------------------------------------------------------------ CUDA (driver API, loaded at run time)

type CuPtr = u64;
type CuRes = c_int;

struct Cuda {
    _lib: libloading::Library,
    module_load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CuRes,
    get_function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CuRes,
    launch: unsafe extern "C" fn(*mut c_void, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
                                 c_uint, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> CuRes,
    alloc: unsafe extern "C" fn(*mut CuPtr, usize) -> CuRes,
    htod: unsafe extern "C" fn(CuPtr, *const c_void, usize) -> CuRes,
    memset2d: unsafe extern "C" fn(CuPtr, usize, u8, usize, usize) -> CuRes,
    ctx_push: unsafe extern "C" fn(*mut c_void) -> CuRes,
    memcpy2d: unsafe extern "C" fn(*const Memcpy2D, *mut c_void) -> CuRes,
    sync: unsafe extern "C" fn() -> CuRes,
    free: unsafe extern "C" fn(CuPtr) -> CuRes,
}

/// `CUDA_MEMCPY2D` (cuda.h).
#[repr(C)]
struct Memcpy2D {
    src_x: usize,
    src_y: usize,
    src_type: c_uint,
    src_host: *const c_void,
    src_device: CuPtr,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x: usize,
    dst_y: usize,
    dst_type: c_uint,
    dst_host: *mut c_void,
    dst_device: CuPtr,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width: usize,
    height: usize,
}

const CU_MEMORYTYPE_HOST: c_uint = 1;
const CU_MEMORYTYPE_DEVICE: c_uint = 2;

impl Memcpy2D {
    fn blank() -> Memcpy2D {
        Memcpy2D { src_x: 0, src_y: 0, src_type: 0, src_host: null(), src_device: 0, src_array: null_mut(),
                   src_pitch: 0, dst_x: 0, dst_y: 0, dst_type: 0, dst_host: null_mut(), dst_device: 0,
                   dst_array: null_mut(), dst_pitch: 0, width: 0, height: 0 }
    }

    /// Device to device, `width` bytes by `height` rows.
    fn device(src: CuPtr, sp: usize, dst: CuPtr, dp: usize, width: usize, height: usize) -> Memcpy2D {
        Memcpy2D { src_type: CU_MEMORYTYPE_DEVICE, src_device: src, src_pitch: sp,
                   dst_type: CU_MEMORYTYPE_DEVICE, dst_device: dst, dst_pitch: dp,
                   width, height, ..Memcpy2D::blank() }
    }

    /// Device to host.
    fn to_host(src: CuPtr, sp: usize, dst: *mut u8, dp: usize, width: usize, height: usize) -> Memcpy2D {
        Memcpy2D { src_type: CU_MEMORYTYPE_DEVICE, src_device: src, src_pitch: sp,
                   dst_type: CU_MEMORYTYPE_HOST, dst_host: dst as *mut c_void, dst_pitch: dp,
                   width, height, ..Memcpy2D::blank() }
    }
}

fn ck(r: CuRes, what: &str) {
    if r != 0 {
        fail(&format!("CUDA {what} failed ({r})"));
    }
}

impl Cuda {
    unsafe fn load() -> Cuda {
        let lib = libloading::Library::new("libcuda.so.1")
            .unwrap_or_else(|e| fail(&format!("libcuda.so.1: {e}")));
        macro_rules! sym {
            ($n:expr) => {
                *lib.get($n).unwrap_or_else(|e| fail(&format!("libcuda symbol: {e}")))
            };
        }
        Cuda {
            module_load: sym!(b"cuModuleLoadData\0"),
            get_function: sym!(b"cuModuleGetFunction\0"),
            launch: sym!(b"cuLaunchKernel\0"),
            alloc: sym!(b"cuMemAlloc_v2\0"),
            htod: sym!(b"cuMemcpyHtoD_v2\0"),
            memset2d: sym!(b"cuMemsetD2D8_v2\0"),
            ctx_push: sym!(b"cuCtxPushCurrent_v2\0"),
            memcpy2d: sym!(b"cuMemcpy2DAsync_v2\0"),
            sync: sym!(b"cuCtxSynchronize\0"),
            free: sym!(b"cuMemFree_v2\0"),
            _lib: lib,
        }
    }

    unsafe fn alloc(&self, n: usize) -> CuPtr {
        let mut p = 0;
        ck((self.alloc)(&mut p, n.max(1)), "alloc");
        p
    }

    /// Launch `f` over a `w x h` grid with 32x8 blocks on the default stream,
    /// the one ffmpeg's decoder and encoder use, so the work stays in order.
    unsafe fn run(&self, f: *mut c_void, w: i32, h: i32, args: &mut [*mut c_void]) {
        if w <= 0 || h <= 0 {
            return;
        }
        let gx = (w as c_uint).div_ceil(32);
        let gy = (h as c_uint).div_ceil(8);
        ck((self.launch)(f, gx, gy, 1, 32, 8, 1, 0, null_mut(), args.as_mut_ptr(), null_mut()), "launch");
    }
}

macro_rules! args {
    ($($v:expr),* $(,)?) => { [$(&mut $v as *mut _ as *mut c_void),*] };
}

mod ass;
mod burn;

// ------------------------------------------------------------------ helpers

fn fail(msg: &str) -> ! {
    eprintln!("osrender: {msg}");
    // Exit without running destructors: tearing down a CUDA context shared
    // with ffmpeg can block at process exit.
    unsafe { _exit(1) }
}

fn av(r: c_int, what: &str) -> c_int {
    if r < 0 && r != ff::AVERROR(ff::EAGAIN) && r != ff::AVERROR_EOF {
        let mut buf = [0 as c_char; 256];
        unsafe { ff::av_strerror(r, buf.as_mut_ptr(), buf.len()) };
        fail(&format!("{what}: {}", unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy()));
    }
    r
}

/// Hand `pkt` to the decoder. True when it is not taken yet (EAGAIN: the
/// decoder wants its decoded frames received first), so the caller keeps the
/// packet and sends it again; dropping it would lose its frame.
unsafe fn send_packet(dctx: *mut ff::AVCodecContext, pkt: *mut ff::AVPacket) -> bool {
    let r = ff::avcodec_send_packet(dctx, pkt);
    if r == ff::AVERROR(ff::EAGAIN) {
        return true;
    }
    av(r, "decode");
    false
}

/// How many 1 ms waits a draining decoder gets before the run gives up.
const DRAIN_WAITS: u32 = 30_000;

fn cs(s: &str) -> CString {
    CString::new(s).unwrap_or_else(|_| fail("string with NUL"))
}

unsafe extern "C" fn pick_cuda(_: *mut ff::AVCodecContext, mut f: *const ff::AVPixelFormat) -> ff::AVPixelFormat {
    while *f != ff::AVPixelFormat::AV_PIX_FMT_NONE {
        if *f == ff::AVPixelFormat::AV_PIX_FMT_CUDA {
            return *f;
        }
        f = f.add(1);
    }
    ff::AVPixelFormat::AV_PIX_FMT_NONE
}

/// The image overlay (free-plan logo) as NV12-shaped planes plus alpha, built
/// once on the CPU and uploaded; the `blend` kernel lays it on every frame.
struct Overlay {
    y: CuPtr,
    ya: CuPtr,
    uv: CuPtr,
    uva: CuPtr,
}

unsafe fn build_overlay(cu: &Cuda, o: &ImageOverlay, w: usize, h: usize) -> Overlay {
    let file = std::fs::File::open(&o.path).unwrap_or_else(|e| fail(&format!("overlay {}: {e}", o.path)));
    let mut dec = png::Decoder::new(file);
    dec.set_transformations(png::Transformations::normalize_to_color8());
    let mut rd = dec.read_info().unwrap_or_else(|e| fail(&format!("overlay png: {e}")));
    let mut buf = vec![0; rd.output_buffer_size()];
    let info = rd.next_frame(&mut buf).unwrap_or_else(|e| fail(&format!("overlay png: {e}")));
    let ch = info.color_type.samples();
    let (lw, lh) = (info.width as usize, info.height as usize);
    let ow = o.w.max(1) as usize;
    let oh = ((lh * ow) as f64 / lw as f64).round().max(1.0) as usize;
    // Premultiplied RGBA canvas the size of the frame.
    let mut c = vec![[0f32; 4]; w * h];
    for yy in 0..oh {
        for xx in 0..ow {
            let (tx, ty) = (o.x as i64 + xx as i64, o.y as i64 + yy as i64);
            if tx < 0 || ty < 0 || tx >= w as i64 || ty >= h as i64 {
                continue;
            }
            let p = &buf[((yy * lh / oh) * lw + xx * lw / ow) * ch..];
            let (r, g, b, a) = match ch {
                1 => (p[0], p[0], p[0], 255),
                2 => (p[0], p[0], p[0], p[1]),
                3 => (p[0], p[1], p[2], 255),
                _ => (p[0], p[1], p[2], p[3]),
            };
            let a = a as f32 / 255.0 * o.opacity;
            c[ty as usize * w + tx as usize] = [r as f32 * a, g as f32 * a, b as f32 * a, a];
        }
    }
    // BT.709, limited range.
    let yuv = |p: [f32; 4]| -> (f32, f32, f32) {
        if p[3] <= 0.0 {
            return (16.0, 128.0, 128.0);
        }
        let (r, g, b) = (p[0] / p[3], p[1] / p[3], p[2] / p[3]);
        (16.0 + 0.1826 * r + 0.6142 * g + 0.0620 * b,
         128.0 - 0.1006 * r - 0.3386 * g + 0.4392 * b,
         128.0 + 0.4392 * r - 0.3989 * g - 0.0403 * b)
    };
    let (mut py, mut pya) = (vec![0u8; w * h], vec![0u8; w * h]);
    for i in 0..w * h {
        py[i] = yuv(c[i]).0.round().clamp(0.0, 255.0) as u8;
        pya[i] = (c[i][3] * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    let (w2, h2) = (w / 2, h / 2);
    let (mut puv, mut puva) = (vec![0u8; w2 * h2 * 2], vec![0u8; w2 * h2]);
    for y in 0..h2 {
        for x in 0..w2 {
            let mut acc = [0f32; 4];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = c[(2 * y + dy) * w + 2 * x + dx];
                for k in 0..4 {
                    acc[k] += p[k] * 0.25;
                }
            }
            let (_, u, v) = yuv(acc);
            puv[(y * w2 + x) * 2] = u.round().clamp(0.0, 255.0) as u8;
            puv[(y * w2 + x) * 2 + 1] = v.round().clamp(0.0, 255.0) as u8;
            puva[y * w2 + x] = (acc[3] * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    let up = |v: &Vec<u8>| -> CuPtr {
        let p = cu.alloc(v.len());
        ck((cu.htod)(p, v.as_ptr() as *const c_void, v.len()), "upload");
        p
    };
    Overlay { y: up(&py), ya: up(&pya), uv: up(&puv), uva: up(&puva) }
}

/// An NV12 scratch image on the card (tightly packed planes).
#[derive(Clone, Copy)]
struct Scratch {
    y: CuPtr,
    uv: CuPtr,
}

// ------------------------------------------------------------------ main

fn main() {
    std::panic::set_hook(Box::new(|info| fail(&format!("panic: {info}"))));
    let a: Vec<String> = std::env::args().collect();
    if a.len() == 3 && a[1] == "burn" {
        let frames = burn::main(&a[2]);
        eprintln!("osrender: {frames} frames");
        unsafe { _exit(0) };
    }
    if a.len() != 4 {
        eprintln!("usage: osrender <input> <plan.json> <output>");
        unsafe { _exit(2) };
    }
    let plan: Plan = serde_json::from_str(
        &std::fs::read_to_string(&a[2]).unwrap_or_else(|e| fail(&format!("plan: {e}"))),
    )
    .unwrap_or_else(|e| fail(&format!("plan: {e}")));
    if plan.out_w <= 0 || plan.out_h <= 0 || plan.out_w % 2 != 0 || plan.out_h % 2 != 0 {
        fail("output size must be positive and even");
    }
    if plan.scenes.is_empty() {
        fail("plan has no scenes");
    }
    let frames = unsafe { run(&a[1], &plan, &a[3]) };
    eprintln!("osrender: {frames} frames");
    unsafe { _exit(0) };
}

unsafe fn run(input: &str, plan: &Plan, output: &str) -> usize {
    let (ow, oh) = (plan.out_w, plan.out_h);

    // ffmpeg creates the CUDA device on the primary context; the kernels run
    // on that same context, so device pointers are shared both ways.
    let mut hwdev: *mut ff::AVBufferRef = null_mut();
    let mut opts: *mut ff::AVDictionary = null_mut();
    ff::av_dict_set(&mut opts, cs("primary_ctx").as_ptr(), cs("1").as_ptr(), 0);
    av(ff::av_hwdevice_ctx_create(&mut hwdev, ff::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA, null(), opts, 0), "cuda device");
    ff::av_dict_free(&mut opts);
    // AVCUDADeviceContext starts with its CUcontext.
    let devctx = (*hwdev).data as *mut ff::AVHWDeviceContext;
    let cuda_ctx = *((*devctx).hwctx as *const *mut c_void);

    let cu = Cuda::load();
    ck((cu.ctx_push)(cuda_ctx), "context");
    let ptx = cs(include_str!("kernels.ptx"));
    let mut module = null_mut();
    ck((cu.module_load)(&mut module, ptx.as_ptr() as *const c_void), "module");
    let kernel = |name: &str| {
        let mut f = null_mut();
        ck((cu.get_function)(&mut f, module, cs(name).as_ptr()), name);
        f
    };
    let (k_resample, k_blur, k_blend) = (kernel("resample"), kernel("blur"), kernel("blend"));

    // Input and NVDEC decoder.
    let mut ictx: *mut ff::AVFormatContext = null_mut();
    av(ff::avformat_open_input(&mut ictx, cs(input).as_ptr(), null(), null_mut()), "open input");
    av(ff::avformat_find_stream_info(ictx, null_mut()), "stream info");
    let mut dec: *const ff::AVCodec = null();
    let vidx = av(ff::av_find_best_stream(ictx, ff::AVMediaType::AVMEDIA_TYPE_VIDEO, -1, -1, &mut dec, 0), "video stream");
    let st = *(*ictx).streams.add(vidx as usize);
    let tb = (*st).time_base;
    let tbf = tb.num as f64 / tb.den as f64;
    let dctx = ff::avcodec_alloc_context3(dec);
    av(ff::avcodec_parameters_to_context(dctx, (*st).codecpar), "decoder params");
    (*dctx).hw_device_ctx = ff::av_buffer_ref(hwdev);
    (*dctx).get_format = Some(pick_cuda);
    (*dctx).pkt_timebase = tb;
    (*dctx).thread_count = if plan.threads > 0 { plan.threads } else { 2 };
    av(ff::avcodec_open2(dctx, dec, null_mut()), "open decoder");
    let start_ts = if (*st).start_time != ff::AV_NOPTS_VALUE { (*st).start_time } else { 0 };

    // NVENC encoder fed with CUDA frames from its own pool.
    let frames_ref = ff::av_hwframe_ctx_alloc(hwdev);
    let fc = (*frames_ref).data as *mut ff::AVHWFramesContext;
    (*fc).format = ff::AVPixelFormat::AV_PIX_FMT_CUDA;
    (*fc).sw_format = ff::AVPixelFormat::AV_PIX_FMT_NV12;
    (*fc).width = ow;
    (*fc).height = oh;
    av(ff::av_hwframe_ctx_init(frames_ref), "frame pool");
    let enc = ff::avcodec_find_encoder_by_name(cs("h264_nvenc").as_ptr());
    if enc.is_null() {
        fail("h264_nvenc not available");
    }
    let ectx = ff::avcodec_alloc_context3(enc);
    let fr = if (*st).avg_frame_rate.num > 0 { (*st).avg_frame_rate } else { (*st).r_frame_rate };
    (*ectx).width = ow;
    (*ectx).height = oh;
    (*ectx).time_base = tb;
    (*ectx).framerate = fr;
    (*ectx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_CUDA;
    (*ectx).sample_aspect_ratio = ff::AVRational { num: 1, den: 1 };
    (*ectx).bit_rate = 0;
    (*ectx).hw_frames_ctx = ff::av_buffer_ref(frames_ref);
    let mut octx: *mut ff::AVFormatContext = null_mut();
    av(ff::avformat_alloc_output_context2(&mut octx, null(), null(), cs(output).as_ptr()), "output");
    if (*(*octx).oformat).flags & ff::AVFMT_GLOBALHEADER as c_int != 0 {
        (*ectx).flags |= ff::AV_CODEC_FLAG_GLOBAL_HEADER as c_int;
    }
    let mut eopts: *mut ff::AVDictionary = null_mut();
    for (k, v) in &plan.encoder {
        ff::av_dict_set(&mut eopts, cs(k).as_ptr(), cs(v).as_ptr(), 0);
    }
    av(ff::avcodec_open2(ectx, enc, &mut eopts), "open encoder");
    ff::av_dict_free(&mut eopts);
    let ost = ff::avformat_new_stream(octx, null());
    av(ff::avcodec_parameters_from_context((*ost).codecpar, ectx), "stream params");
    (*ost).time_base = tb;
    av(ff::avio_open(&mut (*octx).pb, cs(output).as_ptr(), ff::AVIO_FLAG_WRITE as c_int), "open output file");
    av(ff::avformat_write_header(octx, null_mut()), "write header");

    let overlay = plan.overlay.as_ref().map(|o| build_overlay(&cu, o, ow as usize, oh as usize));
    let mut scratch: HashMap<(i32, i32), (Scratch, Scratch)> = HashMap::new();

    // Where each scene starts on the output timeline: scenes are laid end to
    // end, as the per-scene segments were concatenated.
    let mut out_start = Vec::with_capacity(plan.scenes.len());
    let mut acc = 0.0;
    for s in &plan.scenes {
        out_start.push(acc);
        acc += s.dur;
    }
    let half = 0.5 / plan.fps;
    let scene_of = |t: f64| plan.scenes.iter().position(|s| t >= s.ss - half && t < s.ss + s.dur - half);
    let last_end = plan.scenes.iter().map(|s| s.ss + s.dur).fold(0.0, f64::max);

    let pkt = ff::av_packet_alloc();
    let opkt = ff::av_packet_alloc();
    let frame = ff::av_frame_alloc();
    let mut written = 0usize;
    let mut last_pts = i64::MIN;

    let drain = |ectx: *mut ff::AVCodecContext| loop {
        let r = ff::avcodec_receive_packet(ectx, opkt);
        if r < 0 {
            av(r, "encode");
            break;
        }
        ff::av_packet_rescale_ts(opkt, (*ectx).time_base, (*ost).time_base);
        (*opkt).stream_index = 0;
        av(ff::av_interleaved_write_frame(octx, opkt), "write");
    };

    let mut flushing = false;
    let mut done = false;
    let mut held = false;
    let mut drain_waits = 0u32;
    while !done {
        if !flushing {
            if !held && ff::av_read_frame(ictx, pkt) < 0 {
                flushing = true;
                ff::avcodec_send_packet(dctx, null());
            } else {
                held = false;
                if (*pkt).stream_index == vidx {
                    held = send_packet(dctx, pkt);
                }
                if !held {
                    ff::av_packet_unref(pkt);
                }
            }
        }
        loop {
            let r = ff::avcodec_receive_frame(dctx, frame);
            if r == ff::AVERROR_EOF {
                done = true;
                break;
            }
            if r < 0 {
                av(r, "decode");
                if flushing {
                    // Draining: EAGAIN here only means the last frames are
                    // not out of the card yet; EOF ends the stream.
                    drain_waits += 1;
                    if drain_waits > DRAIN_WAITS {
                        fail("decoder did not finish draining");
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                break;
            }
            if (*frame).format != ff::AVPixelFormat::AV_PIX_FMT_CUDA as c_int {
                fail("decoder did not return CUDA frames");
            }
            let ffc = (*(*frame).hw_frames_ctx).data as *mut ff::AVHWFramesContext;
            if (*ffc).sw_format != ff::AVPixelFormat::AV_PIX_FMT_NV12 {
                fail("source is not 8-bit 4:2:0");
            }
            let ts = (*frame).best_effort_timestamp;
            let t = (ts - start_ts) as f64 * tbf;
            if t >= last_end - half {
                ff::av_frame_unref(frame);
                done = true;
                break;
            }
            let Some(si) = scene_of(t) else {
                ff::av_frame_unref(frame);
                continue;
            };
            let scene = &plan.scenes[si];
            let k = ((t - scene.ss) * plan.fps).round().max(0.0) as usize;

            let mut out = ff::av_frame_alloc();
            av(ff::av_hwframe_get_buffer(frames_ref, out, 0), "frame buffer");
            compose(&cu, k_resample, k_blur, frame, out, scene, k, ow, oh, &mut scratch);
            if let Some(o) = &overlay {
                blend(&cu, k_blend, out, o, ow, oh);
            }
            let out_t = out_start[si] + (t - scene.ss);
            let mut pts = (out_t / tbf).round() as i64;
            if pts <= last_pts {
                pts = last_pts + 1;
            }
            last_pts = pts;
            (*out).pts = pts;
            ff::av_frame_unref(frame);
            av(ff::avcodec_send_frame(ectx, out), "encode");
            ff::av_frame_free(&mut out);
            drain(ectx);
            written += 1;
        }
    }
    ff::avcodec_send_frame(ectx, null());
    drain(ectx);
    av(ff::av_write_trailer(octx), "trailer");
    ff::avio_closep(&mut (*octx).pb);
    if written == 0 {
        fail("no frames rendered");
    }
    written
}

#[allow(clippy::too_many_arguments)]
unsafe fn compose(cu: &Cuda, k_resample: *mut c_void, k_blur: *mut c_void, src: *mut ff::AVFrame,
                  out: *mut ff::AVFrame, scene: &Scene, k: usize, ow: i32, oh: i32,
                  scratch: &mut HashMap<(i32, i32), (Scratch, Scratch)>) {
    let (sw, sh) = ((*src).width, (*src).height);
    let s_img = Img { y: (*src).data[0] as CuPtr, uv: (*src).data[1] as CuPtr, pitch: (*src).linesize[0], w: sw, h: sh };
    let d_img = Img { y: (*out).data[0] as CuPtr, uv: (*out).data[1] as CuPtr, pitch: (*out).linesize[0], w: ow, h: oh };
    let (d_y, d_uv, dp) = (d_img.y, d_img.uv, d_img.pitch);
    // Black where no layer draws (ffmpeg's pad default).
    ck((cu.memset2d)(d_y, dp as usize, 16, ow as usize, oh as usize), "fill");
    ck((cu.memset2d)(d_uv, dp as usize, 128, ow as usize, (oh / 2) as usize), "fill");

    for layer in &scene.layers {
        let r = match (&layer.boxes, &layer.src) {
            (Some(b), _) if !b.is_empty() => b[k.min(b.len() - 1)],
            (_, Some(s)) => *s,
            _ => continue,
        };
        let [dx, dy, dw, dh] = layer.dst;
        if let Some(bl) = &layer.blur {
            let (bw, bh) = (bl.w.max(2) & !1, bl.h.max(2) & !1);
            let (a, b) = *scratch.entry((bw, bh)).or_insert_with(|| {
                let n = (bw * bh) as usize;
                (Scratch { y: cu.alloc(n), uv: cu.alloc(n / 2) }, Scratch { y: cu.alloc(n), uv: cu.alloc(n / 2) })
            });
            let tmp = Img { y: a.y, uv: a.uv, pitch: bw, w: bw, h: bh };
            resample_into(cu, k_resample, s_img, tmp, r, [0, 0, bw, bh]);
            blur_planes(cu, k_blur, a, b, bw, bh, bl.sigma);
            resample_into(cu, k_resample, tmp, d_img, [0.0, 0.0, bw as f64, bh as f64], [dx, dy, dw, dh]);
        } else {
            resample_into(cu, k_resample, s_img, d_img, r, [dx, dy, dw, dh]);
        }
    }
}

/// An NV12 image on the card: luma and chroma planes sharing one pitch.
#[derive(Clone, Copy)]
struct Img {
    y: CuPtr,
    uv: CuPtr,
    pitch: i32,
    w: i32,
    h: i32,
}

/// Scale `r` (source pixels) of `s` into the rectangle `dst` of `d`.
unsafe fn resample_into(cu: &Cuda, f: *mut c_void, s: Img, d: Img, r: [f64; 4], dst: [i32; 4]) {
    let [ox, oy, ow, oh] = dst;
    // Chroma: the same rectangles at half resolution.
    let (cx0, cy0) = (ox.div_euclid(2), oy.div_euclid(2));
    let (cx1, cy1) = ((ox + ow + 1).div_euclid(2), (oy + oh + 1).div_euclid(2));
    let planes = [
        (s.y, d.y, 1, r.map(|v| v as f32), [ox, oy, ow, oh]),
        (s.uv, d.uv, 2, r.map(|v| v as f32 / 2.0), [cx0, cy0, cx1 - cx0, cy1 - cy0]),
    ];
    for (mut sp, mut dp, mut step, [mut rx, mut ry, mut rw, mut rh], [mut x, mut y, mut w, mut h]) in planes {
        let (mut spitch, mut dpitch) = (s.pitch, d.pitch);
        let (mut sw, mut sh) = (s.w / step, s.h / step);
        let (mut dw, mut dh) = (d.w / step, d.h / step);
        cu.run(f, w, h, &mut args!(sp, spitch, sw, sh, dp, dpitch, dw, dh, rx, ry, rw, rh, x, y, w, h, step));
    }
}

unsafe fn blur_planes(cu: &Cuda, f: *mut c_void, a: Scratch, b: Scratch, w: i32, h: i32, sigma: f32) {
    // Each plane: a -> b horizontally, then b -> a vertically. Chroma uses the
    // same sigma in its own (half-size) pixels, like ffmpeg's gblur.
    for (pa, pb, pw, ph, step) in [(a.y, b.y, w, h, 1i32), (a.uv, b.uv, w / 2, h / 2, 2i32)] {
        for (mut src, mut dst, mut horizontal) in [(pa, pb, 1i32), (pb, pa, 0i32)] {
            let (mut pitch, mut pw, mut ph, mut step, mut sigma) = (pw * step, pw, ph, step, sigma);
            cu.run(f, pw, ph, &mut args!(src, dst, pitch, pw, ph, step, sigma, horizontal));
        }
    }
}

unsafe fn blend(cu: &Cuda, f: *mut c_void, out: *mut ff::AVFrame, o: &Overlay, w: i32, h: i32) {
    let mut dp = (*out).linesize[0];
    let (mut y, mut uv) = ((*out).data[0] as CuPtr, (*out).data[1] as CuPtr);
    let (mut w1, mut h1, mut w2, mut h2) = (w, h, w / 2, h / 2);
    let (mut one, mut two) = (1i32, 2i32);
    let (mut oy, mut oya, mut ouv, mut ouva) = (o.y, o.ya, o.uv, o.uva);
    cu.run(f, w, h, &mut args!(y, dp, w1, h1, one, oy, oya));
    cu.run(f, w / 2, h / 2, &mut args!(uv, dp, w2, h2, two, ouv, ouva));
}
