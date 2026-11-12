# Privacy blur

Opt-in. A job blurs faces, license plates, QR codes and on-screen personal
details (emails, phone numbers, card numbers, SSNs, IBANs) only when that
request set `blur_sensitive`. The dashboard toggle remembers the choice
(`localStorage.os_blur_sensitive`) and sends it both ways. The API, MCP and
CLI default to off. Autopilot does not turn it on.

`BLUR_SENSITIVE=1` is the only value the renderer honours. `app.process`
pops the name from the child environment and sets it only when this request
asked, so a deployment-wide variable cannot turn it on for every user. The
remote worker does the same before applying job options
(`worker.job_env`). The name is on `remote_worker._OPTION_NAMES`, so a job
that asked can render on the CPU worker.

## Where it runs

After reframing, before captions and hooks (`main.render_clip`). The crop
still needs the real faces; burned text stays sharp because it is drawn
afterwards. The blur is baked into the canonical file. It is the picture,
not a delivery mark: the free-plan watermark stays on the `wm_` copy only.
A fast recut cuts that canonical file and keeps the blur. A source recut or
a hand reframe paints new pixels and passes `blur_sensitive` back into
`render_clip` (`privacy_blur.from_job`: the job env wins, and metadata
counts only as a JSON boolean `true` when the env no longer says).

The choice is also in the resume manifest (`blur_sensitive`). A resume
rebuilds the environment from the server and would otherwise drop it.

## How regions are found

All of it is local. Frames are not uploaded to find the boxes.

- Faces: MediaPipe face detection, the same stack the reframer uses. The
  box is padded to the whole head.
- License plates: a wide rectangle whose interior is busy (the glyphs),
  searched inside YOLO vehicle boxes (car, motorcycle, bus, truck) and,
  more strictly, on the whole frame. The border is not counted, so a plain
  light window does not qualify. No extra model file.
- QR codes: OpenCV's QR detector.
- Personal text: `tesseract` (`tesseract-ocr` + `eng` in the image), about
  twice a second. A line is blurred only when it matches an email, a phone
  number with separators, a Luhn-valid card, an SSN or an IBAN. Dates, IPv4
  addresses and bare digit runs are left. A host without the binary still
  blurs faces, plates and QR codes and says so in the job log.

Boxes are held across the frames that are not scanned and padded, so a
blink or a small shift does not flash the region back. They are drawn in
process. Nothing from the job directory is pasted into an ffmpeg filter.

## When it fails

A blur that cannot run (no face detector, unreadable file, encode error)
deletes the clip and does not deliver it. A job that asked and delivered
nothing fails. Finding nothing is a success: the file is left as the
reframe wrote it, and the log says the clip was scanned.
