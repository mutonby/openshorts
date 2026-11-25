// CUDA kernels for `osrender burn` (burn.rs), a module of their own so the
// reframe kernels' PTX (kernels.ptx) stays exactly what it was. Planes are
// NV12. Regenerate the PTX with:
//   nvcc -ptx -O3 -arch=compute_75 src/burn_kernels.cu -o src/burn_kernels.ptx
// The kernels below reproduce, bit for bit, the 8-bit integer arithmetic of
// the ffmpeg filters the CPU burn pass runs on a yuv420p picture, so that a
// clip burned on the card decodes to the same pixels as one burned by ffmpeg.

// FFmpeg's FAST_DIV255 (libavfilter/vf_overlay.c).
__device__ __forceinline__ unsigned div255(unsigned x) { return ((x + 128u) * 257u) >> 16; }

// vf_overlay, straight alpha, onto a main picture without alpha: luma. The
// overlay is a yuva420p image whose Y and A planes share the pitch `sp`; it
// sits at (x, y) of the frame (already rounded down to even, as the filter
// does for 4:2:0).
extern "C" __global__ void ov_luma(unsigned char* d, int dp, int dw, int dh,
                                   const unsigned char* s, const unsigned char* a, int sp,
                                   int sw, int sh, int x, int y) {
    int k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= sw || j >= sh) return;
    int tx = x + k, ty = y + j;
    if (tx < 0 || ty < 0 || tx >= dw || ty >= dh) return;
    unsigned al = a[j * sp + k];
    unsigned char* p = d + ty * dp + tx;
    *p = (unsigned char)div255(*p * (255u - al) + s[j * sp + k] * al);
}

// vf_overlay chroma (NV12 destination). The alpha of a chroma sample is the
// filter's: the 2x2 average where the block is whole, a mix of the
// horizontal and vertical pairs on the last row / column. On x86 the filter
// blends each row but the last with a SIMD routine that takes the 2x2 average
// for its first `n - n % simd` samples (n = samples the row covers), the last
// column included, and the C code for the rest; `simd` is that width (16 for
// AVX2, 8 for SSE4, 0 for none). The A plane is padded with zeros to the right.
extern "C" __global__ void ov_chroma(unsigned char* d, int dp, int dwc, int dhc,
                                     const unsigned char* su, const unsigned char* sv, int cp,
                                     const unsigned char* a, int ap, int sw, int sh,
                                     int x, int y, int simd) {
    int swc = (sw + 1) >> 1, shc = (sh + 1) >> 1;
    int k = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= swc || j >= shc) return;
    int xp = x >> 1, yp = y >> 1;
    int tx = xp + k, ty = yp + j;
    if (tx < 0 || ty < 0 || tx >= dwc || ty >= dhc) return;
    const unsigned char* aa = a + (2 * j) * ap + 2 * k;
    unsigned al;
    int k0 = max(-xp, 0);
    int n = min(dwc - xp, swc) - k0;
    bool vec = simd > 0 && j + 1 < shc && (k - k0) < n - n % simd;
    if (vec || (j + 1 < shc && k + 1 < swc)) {
        al = (aa[0] + aa[ap] + aa[1] + aa[ap + 1]) >> 2;
    } else {
        unsigned ah = k + 1 < swc ? (aa[0] + aa[1]) >> 1 : aa[0];
        unsigned av = j + 1 < shc ? (aa[0] + aa[ap]) >> 1 : aa[0];
        al = (av + ah) >> 1;
    }
    unsigned char* p = d + ty * dp + 2 * tx;
    p[0] = (unsigned char)div255(p[0] * (255u - al) + su[j * cp + k] * al);
    p[1] = (unsigned char)div255(p[1] * (255u - al) + sv[j * cp + k] * al);
}

// One libass image, already clipped to the frame like ff_blend_mask does:
// its mask covers [x0, x0 + w) x [y0, y0 + h); `off` is where that clipped
// mask starts in the uploaded bitmap buffer; `alpha` is ff_blend_mask's
// (0x10307 * opacity + 3) >> 8; y, u, v the colour as ff_draw_color makes it.
struct AssImg {
    int x0, y0, w, h, stride;
    unsigned off, alpha;
    int y, u, v;
};

// ff_blend_mask (libavfilter/drawutils.c) with an 8-bit mask, luma: every
// image of the frame in libass order, over the box (bx, by, bw, bh).
extern "C" __global__ void ass_luma(unsigned char* d, int dp, int bx, int by, int bw, int bh,
                                    const AssImg* im, int n, const unsigned char* m) {
    int X = bx + blockIdx.x * blockDim.x + threadIdx.x, Y = by + blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= bx + bw || Y >= by + bh) return;
    unsigned v = d[Y * dp + X];
    bool hit = false;
    for (int i = 0; i < n; i++) {
        const AssImg g = im[i];
        int mx = X - g.x0, my = Y - g.y0;
        if (mx < 0 || my < 0 || mx >= g.w || my >= g.h) continue;
        unsigned al = m[g.off + my * g.stride + mx] * g.alpha;
        v = ((0x1010101u - al) * v + al * (unsigned)g.y) >> 24;
        hit = true;
    }
    if (hit) d[Y * dp + X] = (unsigned char)v;
}

// ff_blend_mask chroma (4:2:0, NV12 destination): a chroma sample takes the
// sum of the mask pixels its 2x2 footprint covers, shifted right by 2, the
// partial blocks at the image edges included.
extern "C" __global__ void ass_chroma(unsigned char* d, int dp, int bx, int by, int bw, int bh,
                                      const AssImg* im, int n, const unsigned char* m) {
    int CX = bx + blockIdx.x * blockDim.x + threadIdx.x, CY = by + blockIdx.y * blockDim.y + threadIdx.y;
    if (CX >= bx + bw || CY >= by + bh) return;
    unsigned char* p = d + CY * dp + 2 * CX;
    unsigned u = p[0], v = p[1];
    bool hit = false;
    for (int i = 0; i < n; i++) {
        const AssImg g = im[i];
        int xa = max(2 * CX, g.x0), xb = min(2 * CX + 2, g.x0 + g.w);
        int ya = max(2 * CY, g.y0), yb = min(2 * CY + 2, g.y0 + g.h);
        if (xa >= xb || ya >= yb) continue;
        unsigned t = 0;
        for (int yy = ya; yy < yb; yy++)
            for (int xx = xa; xx < xb; xx++)
                t += m[g.off + (yy - g.y0) * g.stride + (xx - g.x0)];
        unsigned al = (t >> 2) * g.alpha;
        u = ((0x1010101u - al) * u + al * (unsigned)g.u) >> 24;
        v = ((0x1010101u - al) * v + al * (unsigned)g.v) >> 24;
        hit = true;
    }
    if (hit) {
        p[0] = (unsigned char)u;
        p[1] = (unsigned char)v;
    }
}
