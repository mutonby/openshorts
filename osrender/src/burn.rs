//! `osrender burn <spec.json>`: the hook + captions (+ free-plan mark) pass
//! of a finished clip, on the card.
//!
//! The CPU pass is one ffmpeg: the clip is decoded, `overlay` lays the hook
//! image on it for its first seconds, `ass` draws the captions, `overlay` adds
//! the mark on the served copy, and every picture goes back up to NVENC. Here
//! NVDEC decodes into CUDA frames, the kernels in `burn_kernels.cu` do the same
//! three blends with the same integer arithmetic (libass is the same library,
//! set up as FFmpeg sets it up), and NVENC encodes each output from device
//! memory. The audio is copied. The static images (hook, mark) are converted
//! to yuva420p beforehand by ffmpeg itself (`gpu_burn.py`), so their colour
//! conversion is swscale's, unchanged.
//!
//! Outputs are stages of the same chain: `hook` (hook only), `ass` (hook +
//! captions), `mark` (hook + captions + mark). A `raw` output writes the
//! stage's pictures as planar yuv420p instead of encoding them (used by the
//! equivalence tests). Any error exits non-zero and the caller burns with
//! ffmpeg instead.

use crate::ass::{AssImage, Subtitles};
use crate::{av, ck, cs, fail, pick_cuda, send_packet, CuPtr, Cuda, DRAIN_WAITS};
use ffmpeg_sys_next as ff;
use serde::Deserialize;
use std::ffi::{c_int, c_void};
use std::io::Write;
use std::ptr::{null, null_mut};

#[derive(Deserialize)]
struct Spec {
    input: String,
    hook: Option<StaticSpec>,
    ass: Option<AssSpec>,
    mark: Option<StaticSpec>,
    outputs: Vec<OutSpec>,
    #[serde(default)]
    encoder: Vec<(String, String)>,
    #[serde(default)]
    threads: i32,
    /// Width of FFmpeg's x86 overlay row routine (see ov_chroma).
    #[serde(default = "default_simd")]
    overlay_simd: i32,
}

fn default_simd() -> i32 {
    16
}

#[derive(Deserialize)]
struct StaticSpec {
    /// A NUT file holding one yuva420p rawvideo picture.
    planes: String,
    x: i32,
    y: i32,
    /// Drawn while 0 <= t <= until (overlay's enable='between(t,0,until)').
    until: Option<f64>,
}

#[derive(Deserialize)]
struct AssSpec {
    file: String,
    fonts_dir: String,
}

#[derive(Deserialize)]
struct OutSpec {
    path: String,
    stage: String,
    #[serde(default)]
    raw: bool,
}

#[derive(Clone, Copy, PartialEq, PartialOrd)]
enum Stage {
    Hook = 0,
    Ass = 1,
    Mark = 2,
}

/// A static overlay on the card: Y and A planes share `pitch`, U and V `cpitch`.
struct Static {
    y: CuPtr,
    a: CuPtr,
    u: CuPtr,
    v: CuPtr,
    pitch: i32,
    cpitch: i32,
    w: i32,
    h: i32,
    x: i32,
    y0: i32,
    until: Option<f64>,
}

/// kernels.cu `AssImg`.
#[repr(C)]
#[derive(Clone, Copy)]
struct AssImg {
    x0: i32,
    y0: i32,
    w: i32,
    h: i32,
    stride: i32,
    off: u32,
    alpha: u32,
    y: i32,
    u: i32,
    v: i32,
}

/// The current frame's caption images on the card, and the boxes they cover.
struct AssState {
    subs: Subtitles,
    imgs: CuPtr,
    imgs_cap: usize,
    mask: CuPtr,
    mask_cap: usize,
    n: i32,
    luma: [i32; 4],
    chroma: [i32; 4],
    coef: (f64, f64),
}

struct Output {
    stage: Stage,
    raw: Option<std::fs::File>,
    ectx: *mut ff::AVCodecContext,
    octx: *mut ff::AVFormatContext,
    vst: *mut ff::AVStream,
    ast: *mut ff::AVStream,
}

pub fn main(spec_path: &str) -> usize {
    let spec: Spec = serde_json::from_str(
        &std::fs::read_to_string(spec_path).unwrap_or_else(|e| fail(&format!("spec: {e}"))),
    )
    .unwrap_or_else(|e| fail(&format!("spec: {e}")));
    if spec.outputs.is_empty() {
        fail("spec has no outputs");
    }
    unsafe { run(&spec) }
}

fn stage_of(s: &str) -> Stage {
    match s {
        "hook" => Stage::Hook,
        "ass" => Stage::Ass,
        "mark" => Stage::Mark,
        _ => fail(&format!("unknown stage {s}")),
    }
}

/// Luma coefficients (cr, cb) as libavutil's `av_csp_luma_coeffs_from_avcsp`
/// gives them (AVR(d) = { (int)(d * 100000), 100000 }), for the colour space
/// FFmpeg's drawing code assumes: unspecified is BT.601.
fn luma_coefficients(csp: ff::AVColorSpace) -> (f64, f64) {
    use ff::AVColorSpace::*;
    let r = |d: f64| ((d * 100000.0) as i32) as f64 / 100000.0;
    match csp {
        AVCOL_SPC_UNSPECIFIED | AVCOL_SPC_SMPTE170M | AVCOL_SPC_BT470BG => (r(0.299), r(0.114)),
        AVCOL_SPC_BT709 => (r(0.2126), r(0.0722)),
        _ => fail("colour space not supported by the GPU burn"),
    }
}

/// ff_draw_color for an 8-bit, limited-range YUV picture.
fn draw_color(rgb: [u8; 3], coef: (f64, f64)) -> (i32, i32, i32) {
    let (cr, cb) = coef;
    let cg = 1.0 - cb - cr;
    let bscale = 0.5 / (cb - 1.0);
    let rscale = 0.5 / (cr - 1.0);
    let m = [
        [cr, cg, cb],
        [bscale * cr, bscale * cg, 0.5],
        [0.5, rscale * cg, rscale * cb],
    ];
    let v = [rgb[0] as f64 / 255., rgb[1] as f64 / 255., rgb[2] as f64 / 255.];
    let mut out = [0i32; 3];
    for i in 0..3 {
        let mut d = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
        let chroma = i > 0;
        d *= (if chroma { 224. } else { 219. }) / 255.;
        d += (if chroma { 128. } else { 16. }) / 255.;
        out[i] = (d * 255. + 0.5) as u32 as i32;
    }
    (out[0], out[1], out[2])
}

unsafe fn load_static(cu: &Cuda, s: &StaticSpec) -> Static {
    let mut ictx: *mut ff::AVFormatContext = null_mut();
    av(ff::avformat_open_input(&mut ictx, cs(&s.planes).as_ptr(), null(), null_mut()), "open overlay");
    av(ff::avformat_find_stream_info(ictx, null_mut()), "overlay info");
    let mut dec: *const ff::AVCodec = null();
    let idx = av(ff::av_find_best_stream(ictx, ff::AVMediaType::AVMEDIA_TYPE_VIDEO, -1, -1, &mut dec, 0), "overlay stream");
    let st = *(*ictx).streams.add(idx as usize);
    let dctx = ff::avcodec_alloc_context3(dec);
    av(ff::avcodec_parameters_to_context(dctx, (*st).codecpar), "overlay params");
    av(ff::avcodec_open2(dctx, dec, null_mut()), "overlay decoder");
    let pkt = ff::av_packet_alloc();
    let frame = ff::av_frame_alloc();
    let mut got = false;
    while !got && ff::av_read_frame(ictx, pkt) >= 0 {
        if (*pkt).stream_index == idx {
            av(ff::avcodec_send_packet(dctx, pkt), "overlay decode");
            got = ff::avcodec_receive_frame(dctx, frame) >= 0;
        }
        ff::av_packet_unref(pkt);
    }
    if !got {
        ff::avcodec_send_packet(dctx, null());
        got = ff::avcodec_receive_frame(dctx, frame) >= 0;
    }
    if !got {
        fail("overlay has no picture");
    }
    if (*frame).format != ff::AVPixelFormat::AV_PIX_FMT_YUVA420P as c_int {
        fail("overlay is not yuva420p");
    }
    let (w, h) = ((*frame).width, (*frame).height);
    let (cw, ch) = ((w + 1) / 2, (h + 1) / 2);
    // Room to the right of the A plane: the SIMD rows read one column past an
    // odd width.
    let pitch = (w + 16) & !15;
    let cpitch = cw;
    let plane = |data: *const u8, ls: i32, pw: i32, ph: i32, dpitch: i32| -> Vec<u8> {
        let mut v = vec![0u8; (dpitch * ph) as usize];
        for r in 0..ph {
            std::ptr::copy_nonoverlapping(data.add((r * ls) as usize), v.as_mut_ptr().add((r * dpitch) as usize), pw as usize);
        }
        v
    };
    let fy = plane((*frame).data[0], (*frame).linesize[0], w, h, pitch);
    let fu = plane((*frame).data[1], (*frame).linesize[1], cw, ch, cpitch);
    let fv = plane((*frame).data[2], (*frame).linesize[2], cw, ch, cpitch);
    let fa = plane((*frame).data[3], (*frame).linesize[3], w, h, pitch);
    let up = |v: &Vec<u8>| -> CuPtr {
        let p = cu.alloc(v.len());
        ck((cu.htod)(p, v.as_ptr() as *const c_void, v.len()), "upload");
        p
    };
    // vf_overlay rounds the position down to the chroma grid.
    Static {
        y: up(&fy),
        a: up(&fa),
        u: up(&fu),
        v: up(&fv),
        pitch,
        cpitch,
        w,
        h,
        x: s.x & !1,
        y0: s.y & !1,
        until: s.until,
    }
}

unsafe fn overlay(cu: &Cuda, k_luma: *mut c_void, k_chroma: *mut c_void, f: *mut ff::AVFrame,
                  o: &Static, simd: i32) {
    let (mut dw, mut dh) = ((*f).width, (*f).height);
    let (mut d, mut dp) = ((*f).data[0] as CuPtr, (*f).linesize[0]);
    let (mut s, mut a, mut sp, mut sw, mut sh, mut x, mut y) = (o.y, o.a, o.pitch, o.w, o.h, o.x, o.y0);
    cu.run(k_luma, o.w, o.h, &mut args!(d, dp, dw, dh, s, a, sp, sw, sh, x, y));
    let (mut duv, mut dpc) = ((*f).data[1] as CuPtr, (*f).linesize[1]);
    let (mut dwc, mut dhc) = ((dw + 1) / 2, (dh + 1) / 2);
    let (mut su, mut sv, mut cp, mut ap, mut simd) = (o.u, o.v, o.cpitch, o.pitch, simd);
    cu.run(k_chroma, (o.w + 1) / 2, (o.h + 1) / 2,
           &mut args!(duv, dpc, dwc, dhc, su, sv, cp, a, ap, sw, sh, x, y, simd));
}

impl AssState {
    /// Upload this frame's images (only when libass says they changed).
    unsafe fn update(&mut self, cu: &Cuda, now_ms: i64, fw: i32, fh: i32) {
        let (mut img, changed) = self.subs.render(now_ms);
        if !changed && self.imgs != 0 {
            return;
        }
        let mut table: Vec<AssImg> = Vec::new();
        let mut masks: Vec<u8> = Vec::new();
        let (mut lx0, mut ly0, mut lx1, mut ly1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        while !img.is_null() {
            let im: &AssImage = &*img;
            img = im.next;
            let a = 255 - (im.color & 0xff);
            // ff_blend_mask: clip to the frame, skip what is left empty or clear.
            let (mut x0, mut w, mut xm0) = (im.dst_x, im.w, 0);
            if x0 < 0 {
                xm0 = -x0;
                w += x0;
                x0 = 0;
            }
            if x0 + w > fw {
                w = fw - x0;
            }
            let (mut y0, mut h, mut ym0) = (im.dst_y, im.h, 0);
            if y0 < 0 {
                ym0 = -y0;
                h += y0;
                y0 = 0;
            }
            if y0 + h > fh {
                h = fh - y0;
            }
            if w <= 0 || h <= 0 || a == 0 {
                continue;
            }
            let rgb = [(im.color >> 24) as u8, (im.color >> 16) as u8, (im.color >> 8) as u8];
            let (yy, uu, vv) = draw_color(rgb, self.coef);
            let off = masks.len() as u32;
            for r in 0..h {
                let src = im.bitmap.add(((ym0 + r) * im.stride + xm0) as usize);
                masks.extend_from_slice(std::slice::from_raw_parts(src, w as usize));
            }
            table.push(AssImg {
                x0, y0, w, h, stride: w, off,
                alpha: (0x10307 * a + 0x3) >> 8,
                y: yy, u: uu, v: vv,
            });
            lx0 = lx0.min(x0);
            ly0 = ly0.min(y0);
            lx1 = lx1.max(x0 + w);
            ly1 = ly1.max(y0 + h);
        }
        self.n = table.len() as i32;
        if self.n == 0 {
            self.imgs = self.imgs.max(1);
            return;
        }
        let tbytes = table.len() * std::mem::size_of::<AssImg>();
        if tbytes > self.imgs_cap || self.imgs <= 1 {
            if self.imgs > 1 {
                (cu.free)(self.imgs);
            }
            self.imgs_cap = tbytes.max(4096) * 2;
            self.imgs = cu.alloc(self.imgs_cap);
        }
        if masks.len() > self.mask_cap || self.mask == 0 {
            if self.mask != 0 {
                (cu.free)(self.mask);
            }
            self.mask_cap = masks.len().max(1 << 16) * 2;
            self.mask = cu.alloc(self.mask_cap);
        }
        ck((cu.htod)(self.imgs, table.as_ptr() as *const c_void, tbytes), "upload captions");
        ck((cu.htod)(self.mask, masks.as_ptr() as *const c_void, masks.len()), "upload captions");
        self.luma = [lx0, ly0, lx1 - lx0, ly1 - ly0];
        let (cx0, cy0, cx1, cy1) = (lx0 >> 1, ly0 >> 1, (lx1 - 1) >> 1, (ly1 - 1) >> 1);
        self.chroma = [cx0, cy0, cx1 - cx0 + 1, cy1 - cy0 + 1];
    }

    unsafe fn draw(&self, cu: &Cuda, k_luma: *mut c_void, k_chroma: *mut c_void, f: *mut ff::AVFrame) {
        if self.n == 0 {
            return;
        }
        let (mut imgs, mut n, mut mask) = (self.imgs, self.n, self.mask);
        let (mut d, mut dp) = ((*f).data[0] as CuPtr, (*f).linesize[0]);
        let [mut bx, mut by, mut bw, mut bh] = self.luma;
        cu.run(k_luma, bw, bh, &mut args!(d, dp, bx, by, bw, bh, imgs, n, mask));
        let (mut duv, mut dpc) = ((*f).data[1] as CuPtr, (*f).linesize[1]);
        let [mut cx, mut cy, mut cw, mut ch] = self.chroma;
        cu.run(k_chroma, cw, ch, &mut args!(duv, dpc, cx, cy, cw, ch, imgs, n, mask));
    }
}

unsafe fn copy_frame(cu: &Cuda, src: *const ff::AVFrame, dst: *mut ff::AVFrame) {
    let (w, h) = ((*src).width as usize, (*src).height as usize);
    for (p, rows) in [(0usize, h), (1usize, h / 2)] {
        let c = crate::Memcpy2D::device(
            (*src).data[p] as CuPtr, (*src).linesize[p] as usize,
            (*dst).data[p] as CuPtr, (*dst).linesize[p] as usize, w, rows);
        ck((cu.memcpy2d)(&c, null_mut()), "copy");
    }
}

unsafe fn write_raw(cu: &Cuda, f: *const ff::AVFrame, out: &mut std::fs::File) {
    let (w, h) = ((*f).width as usize, (*f).height as usize);
    let mut y = vec![0u8; w * h];
    let mut uv = vec![0u8; w * h / 2];
    for (p, rows, buf) in [(0usize, h, &mut y), (1usize, h / 2, &mut uv)] {
        let c = crate::Memcpy2D::to_host((*f).data[p] as CuPtr, (*f).linesize[p] as usize,
                                         buf.as_mut_ptr(), w, w, rows);
        ck((cu.memcpy2d)(&c, null_mut()), "download");
    }
    ck((cu.sync)(), "sync");
    let (mut u, mut v) = (Vec::with_capacity(w * h / 4), Vec::with_capacity(w * h / 4));
    for pair in uv.chunks_exact(2) {
        u.push(pair[0]);
        v.push(pair[1]);
    }
    for b in [&y, &u, &v] {
        out.write_all(b).unwrap_or_else(|e| fail(&format!("raw output: {e}")));
    }
}

unsafe fn run(spec: &Spec) -> usize {
    let mut hwdev: *mut ff::AVBufferRef = null_mut();
    let mut opts: *mut ff::AVDictionary = null_mut();
    ff::av_dict_set(&mut opts, cs("primary_ctx").as_ptr(), cs("1").as_ptr(), 0);
    av(ff::av_hwdevice_ctx_create(&mut hwdev, ff::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA, null(), opts, 0), "cuda device");
    ff::av_dict_free(&mut opts);
    let devctx = (*hwdev).data as *mut ff::AVHWDeviceContext;
    let cuda_ctx = *((*devctx).hwctx as *const *mut c_void);
    let cu = Cuda::load();
    ck((cu.ctx_push)(cuda_ctx), "context");
    let ptx = cs(include_str!("burn_kernels.ptx"));
    let mut module = null_mut();
    ck((cu.module_load)(&mut module, ptx.as_ptr() as *const c_void), "module");
    let kernel = |name: &str| {
        let mut f = null_mut();
        ck((cu.get_function)(&mut f, module, cs(name).as_ptr()), name);
        f
    };
    let (k_ovl, k_ovc, k_asl, k_asc) = (kernel("ov_luma"), kernel("ov_chroma"), kernel("ass_luma"), kernel("ass_chroma"));

    // Input: video on NVDEC, audio copied.
    let mut ictx: *mut ff::AVFormatContext = null_mut();
    av(ff::avformat_open_input(&mut ictx, cs(&spec.input).as_ptr(), null(), null_mut()), "open input");
    av(ff::avformat_find_stream_info(ictx, null_mut()), "stream info");
    let mut dec: *const ff::AVCodec = null();
    let vidx = av(ff::av_find_best_stream(ictx, ff::AVMediaType::AVMEDIA_TYPE_VIDEO, -1, -1, &mut dec, 0), "video stream");
    let aidx = ff::av_find_best_stream(ictx, ff::AVMediaType::AVMEDIA_TYPE_AUDIO, -1, vidx, null_mut(), 0);
    let st = *(*ictx).streams.add(vidx as usize);
    let ain = if aidx >= 0 { *(*ictx).streams.add(aidx as usize) } else { null_mut() };
    let tb = (*st).time_base;
    let tbf = tb.num as f64 / tb.den as f64;
    let par = (*st).codecpar;
    let (fw, fh) = ((*par).width, (*par).height);
    if fw <= 0 || fh <= 0 || fw % 2 != 0 || fh % 2 != 0 {
        fail("clip size must be positive and even");
    }
    let dctx = ff::avcodec_alloc_context3(dec);
    av(ff::avcodec_parameters_to_context(dctx, par), "decoder params");
    (*dctx).hw_device_ctx = ff::av_buffer_ref(hwdev);
    (*dctx).get_format = Some(pick_cuda);
    (*dctx).pkt_timebase = tb;
    (*dctx).thread_count = if spec.threads > 0 { spec.threads } else { 2 };
    av(ff::avcodec_open2(dctx, dec, null_mut()), "open decoder");
    // ffmpeg shifts every input timestamp by the file's start time.
    let start = if (*ictx).start_time != ff::AV_NOPTS_VALUE { (*ictx).start_time } else { 0 };
    let usec = ff::AVRational { num: 1, den: 1_000_000 };
    let v_off = ff::av_rescale_q(start, usec, tb);
    let a_off = if ain.is_null() { 0 } else { ff::av_rescale_q(start, usec, (*ain).time_base) };

    let frames_ref = ff::av_hwframe_ctx_alloc(hwdev);
    let fc = (*frames_ref).data as *mut ff::AVHWFramesContext;
    (*fc).format = ff::AVPixelFormat::AV_PIX_FMT_CUDA;
    (*fc).sw_format = ff::AVPixelFormat::AV_PIX_FMT_NV12;
    (*fc).width = fw;
    (*fc).height = fh;
    av(ff::av_hwframe_ctx_init(frames_ref), "frame pool");

    // The encoder settings the ffmpeg CLI would pick for this input.
    let fr = ff::av_guess_frame_rate(ictx, st, null_mut());
    if fr.num <= 0 || fr.den <= 0 {
        fail("no frame rate");
    }
    let enc_tb = ff::AVRational { num: fr.den, den: fr.num };
    let coef = luma_coefficients((*par).color_space);
    // FFmpeg's drawing code and the blends here are for a limited-range picture.
    if (*par).color_range == ff::AVColorRange::AVCOL_RANGE_JPEG {
        fail("full-range clip not supported by the GPU burn");
    }

    let mut outs: Vec<Output> = Vec::new();
    for o in &spec.outputs {
        let stage = stage_of(&o.stage);
        if o.raw {
            let f = std::fs::File::create(&o.path).unwrap_or_else(|e| fail(&format!("{}: {e}", o.path)));
            outs.push(Output { stage, raw: Some(f), ectx: null_mut(), octx: null_mut(), vst: null_mut(), ast: null_mut() });
            continue;
        }
        let enc = ff::avcodec_find_encoder_by_name(cs("h264_nvenc").as_ptr());
        if enc.is_null() {
            fail("h264_nvenc not available");
        }
        let ectx = ff::avcodec_alloc_context3(enc);
        (*ectx).width = fw;
        (*ectx).height = fh;
        (*ectx).time_base = enc_tb;
        (*ectx).framerate = fr;
        (*ectx).pix_fmt = ff::AVPixelFormat::AV_PIX_FMT_CUDA;
        (*ectx).sample_aspect_ratio = (*par).sample_aspect_ratio;
        (*ectx).color_range = (*par).color_range;
        (*ectx).colorspace = (*par).color_space;
        (*ectx).color_primaries = (*par).color_primaries;
        (*ectx).color_trc = (*par).color_trc;
        (*ectx).chroma_sample_location = (*par).chroma_location;
        (*ectx).hw_frames_ctx = ff::av_buffer_ref(frames_ref);
        let mut octx: *mut ff::AVFormatContext = null_mut();
        av(ff::avformat_alloc_output_context2(&mut octx, null(), null(), cs(&o.path).as_ptr()), "output");
        if (*(*octx).oformat).flags & ff::AVFMT_GLOBALHEADER as c_int != 0 {
            (*ectx).flags |= ff::AV_CODEC_FLAG_GLOBAL_HEADER as c_int;
        }
        let mut eopts: *mut ff::AVDictionary = null_mut();
        for (k, v) in &spec.encoder {
            ff::av_dict_set(&mut eopts, cs(k).as_ptr(), cs(v).as_ptr(), 0);
        }
        av(ff::avcodec_open2(ectx, enc, &mut eopts), "open encoder");
        ff::av_dict_free(&mut eopts);
        let vst = ff::avformat_new_stream(octx, null());
        av(ff::avcodec_parameters_from_context((*vst).codecpar, ectx), "stream params");
        (*vst).time_base = enc_tb;
        let mut ast = null_mut();
        if !ain.is_null() {
            ast = ff::avformat_new_stream(octx, null());
            av(ff::avcodec_parameters_copy((*ast).codecpar, (*ain).codecpar), "audio params");
            (*(*ast).codecpar).codec_tag = 0;
            (*ast).time_base = (*ain).time_base;
        }
        av(ff::avio_open(&mut (*octx).pb, cs(&o.path).as_ptr(), ff::AVIO_FLAG_WRITE as c_int), "open output file");
        let mut mopts: *mut ff::AVDictionary = null_mut();
        ff::av_dict_set(&mut mopts, cs("movflags").as_ptr(), cs("+faststart").as_ptr(), 0);
        av(ff::avformat_write_header(octx, &mut mopts), "write header");
        ff::av_dict_free(&mut mopts);
        outs.push(Output { stage, raw: None, ectx, octx, vst, ast });
    }
    let last = outs.iter().map(|o| o.stage).fold(Stage::Hook, |a, b| if b > a { b } else { a });

    let hook = spec.hook.as_ref().map(|s| load_static(&cu, s));
    let mark = spec.mark.as_ref().map(|s| load_static(&cu, s));
    let mut ass = match &spec.ass {
        Some(a) if last >= Stage::Ass => {
            let subs = Subtitles::open(&a.file, &a.fonts_dir, fw, fh).unwrap_or_else(|e| fail(&e));
            Some(AssState { subs, imgs: 0, imgs_cap: 0, mask: 0, mask_cap: 0, n: 0,
                            luma: [0; 4], chroma: [0; 4], coef })
        }
        _ => None,
    };

    let pkt = ff::av_packet_alloc();
    let opkt = ff::av_packet_alloc();
    let frame = ff::av_frame_alloc();
    let mut written = 0usize;
    let mut last_pts = i64::MIN;

    let drain = |o: &Output, flush: bool| {
        if o.ectx.is_null() {
            return;
        }
        if flush {
            ff::avcodec_send_frame(o.ectx, null());
        }
        loop {
            let r = ff::avcodec_receive_packet(o.ectx, opkt);
            if r < 0 {
                av(r, "encode");
                break;
            }
            ff::av_packet_rescale_ts(opkt, (*o.ectx).time_base, (*o.vst).time_base);
            (*opkt).stream_index = (*o.vst).index;
            av(ff::av_interleaved_write_frame(o.octx, opkt), "write");
        }
    };
    let send = |outs: &mut Vec<Output>, stage: Stage, f: *mut ff::AVFrame| {
        for o in outs.iter_mut().filter(|o| o.stage == stage) {
            if let Some(raw) = o.raw.as_mut() {
                write_raw(&cu, f, raw);
            } else {
                av(ff::avcodec_send_frame(o.ectx, f), "encode");
                drain(o, false);
            }
        }
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
                } else if (*pkt).stream_index == aidx {
                    for o in outs.iter().filter(|o| !o.ast.is_null()) {
                        let c = ff::av_packet_clone(pkt);
                        if (*c).pts != ff::AV_NOPTS_VALUE {
                            (*c).pts -= a_off;
                        }
                        if (*c).dts != ff::AV_NOPTS_VALUE {
                            (*c).dts -= a_off;
                        }
                        ff::av_packet_rescale_ts(c, (*ain).time_base, (*o.ast).time_base);
                        (*c).stream_index = (*o.ast).index;
                        (*c).pos = -1;
                        av(ff::av_interleaved_write_frame(o.octx, c), "write audio");
                        let mut c = c;
                        ff::av_packet_free(&mut c);
                    }
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
            if (*frame).best_effort_timestamp == ff::AV_NOPTS_VALUE {
                fail("frame without a timestamp");
            }
            let ts = (*frame).best_effort_timestamp - v_off;
            let t = ts as f64 * tbf;
            let mut pts = ff::av_rescale_q(ts, tb, enc_tb);
            if pts <= last_pts {
                pts = last_pts + 1;
            }
            last_pts = pts;

            // Stage frames: each one NVENC may still be reading is never
            // drawn on again, the next stage starts from a copy.
            let fresh = || {
                let f = ff::av_frame_alloc();
                av(ff::av_hwframe_get_buffer(frames_ref, f, 0), "frame buffer");
                (*f).pts = pts;
                f
            };
            let mut cur = fresh();
            copy_frame(&cu, frame, cur);
            ff::av_frame_unref(frame);
            if let Some(h) = &hook {
                if t >= 0.0 && h.until.map_or(true, |u| t <= u) {
                    overlay(&cu, k_ovl, k_ovc, cur, h, spec.overlay_simd);
                }
            }
            let mut sent = false;
            for stage in [Stage::Hook, Stage::Ass, Stage::Mark] {
                if stage > last {
                    break;
                }
                if stage != Stage::Hook && sent {
                    let next = fresh();
                    copy_frame(&cu, cur, next);
                    let mut old = cur;
                    ff::av_frame_free(&mut old);
                    cur = next;
                    sent = false;
                }
                match stage {
                    Stage::Ass => {
                        if let Some(a) = ass.as_mut() {
                            let now_ms = (ts as f64 * tbf * 1000.0) as i64;
                            a.update(&cu, now_ms, fw, fh);
                            a.draw(&cu, k_asl, k_asc, cur);
                        }
                    }
                    Stage::Mark => {
                        if let Some(m) = &mark {
                            overlay(&cu, k_ovl, k_ovc, cur, m, spec.overlay_simd);
                        }
                    }
                    Stage::Hook => {}
                }
                if outs.iter().any(|o| o.stage == stage) {
                    send(&mut outs, stage, cur);
                    sent = true;
                }
            }
            ff::av_frame_free(&mut cur);
            written += 1;
        }
    }
    for o in &outs {
        if !o.ectx.is_null() {
            drain(o, true);
            av(ff::av_write_trailer(o.octx), "trailer");
            ff::avio_closep(&mut (*o.octx).pb);
        }
    }
    if written == 0 {
        fail("no frames burned");
    }
    written
}
