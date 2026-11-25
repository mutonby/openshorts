//! The few libass calls the burn needs, set up exactly as FFmpeg's `ass`
//! filter (libavfilter/vf_subtitles.c) sets them up, so the same library
//! renders the same bitmaps for every frame.

use std::ffi::{c_char, c_int, c_longlong, c_void, CString};
use std::ptr::null;

#[repr(C)]
pub struct AssLibrary {
    _p: [u8; 0],
}
#[repr(C)]
pub struct AssRenderer {
    _p: [u8; 0],
}
#[repr(C)]
pub struct AssTrack {
    _p: [u8; 0],
}

/// `ASS_Image` (ass.h): only the leading fields are read.
#[repr(C)]
pub struct AssImage {
    pub w: c_int,
    pub h: c_int,
    pub stride: c_int,
    pub bitmap: *const u8,
    pub color: u32,
    pub dst_x: c_int,
    pub dst_y: c_int,
    pub next: *const AssImage,
}

type MsgCb = unsafe extern "C" fn(c_int, *const c_char, *mut c_void, *mut c_void);

#[link(name = "ass")]
extern "C" {
    fn ass_library_init() -> *mut AssLibrary;
    fn ass_set_message_cb(lib: *mut AssLibrary, cb: Option<MsgCb>, data: *mut c_void);
    fn ass_set_fonts_dir(lib: *mut AssLibrary, dir: *const c_char);
    fn ass_set_extract_fonts(lib: *mut AssLibrary, extract: c_int);
    fn ass_renderer_init(lib: *mut AssLibrary) -> *mut AssRenderer;
    fn ass_set_frame_size(r: *mut AssRenderer, w: c_int, h: c_int);
    fn ass_set_storage_size(r: *mut AssRenderer, w: c_int, h: c_int);
    fn ass_set_fonts(r: *mut AssRenderer, font: *const c_char, family: *const c_char,
                     dfp: c_int, config: *const c_char, update: c_int);
    fn ass_read_file(lib: *mut AssLibrary, name: *const c_char, codepage: *const c_char) -> *mut AssTrack;
    fn ass_render_frame(r: *mut AssRenderer, t: *mut AssTrack, now: c_longlong,
                        change: *mut c_int) -> *const AssImage;
}

/// libass's messages are not ours to print (FFmpeg routes them to its log).
unsafe extern "C" fn quiet(_: c_int, _: *const c_char, _: *mut c_void, _: *mut c_void) {}

pub struct Subtitles {
    renderer: *mut AssRenderer,
    track: *mut AssTrack,
}

impl Subtitles {
    /// vf_subtitles.c `init` + `init_ass` + `config_input`, in that order.
    pub fn open(file: &str, fonts_dir: &str, w: i32, h: i32) -> Result<Subtitles, String> {
        let c = |s: &str| CString::new(s).map_err(|_| "string with NUL".to_string());
        let (file_c, dir_c) = (c(file)?, c(fonts_dir)?);
        unsafe {
            let lib = ass_library_init();
            if lib.is_null() {
                return Err("ass_library_init failed".into());
            }
            ass_set_message_cb(lib, Some(quiet), std::ptr::null_mut());
            ass_set_fonts_dir(lib, dir_c.as_ptr());
            ass_set_extract_fonts(lib, 1);
            let renderer = ass_renderer_init(lib);
            if renderer.is_null() {
                return Err("ass_renderer_init failed".into());
            }
            ass_set_fonts(renderer, null(), null(), 1, null(), 1);
            let track = ass_read_file(lib, file_c.as_ptr(), null());
            if track.is_null() {
                return Err(format!("could not read subtitles {file}"));
            }
            ass_set_frame_size(renderer, w, h);
            ass_set_storage_size(renderer, w, h);
            Ok(Subtitles { renderer, track })
        }
    }

    /// The images for `now_ms` and whether they differ from the last call's.
    pub fn render(&mut self, now_ms: i64) -> (*const AssImage, bool) {
        let mut change: c_int = 0;
        let img = unsafe { ass_render_frame(self.renderer, self.track, now_ms, &mut change) };
        (img, change != 0)
    }
}
