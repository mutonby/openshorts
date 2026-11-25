"""Build fonts/OpenShortsArabic-Bold.ttf from the Noto fonts in the backend image.

Arabic + Latin in one file, so PIL and libass can shape a mixed line with one
font. Run from the repo root inside the container.
"""
import os
import tempfile

from fontTools import merge, subset
from fontTools.ttLib import TTFont

NOTO = "/usr/share/fonts/truetype/noto/"
OUT = "fonts/OpenShortsArabic-Bold.ttf"
NAME = "OpenShorts Arabic"

with tempfile.TemporaryDirectory() as tmp:
    arabic = NOTO + "NotoSansArabic-Bold.ttf"
    covered = set(TTFont(arabic).getBestCmap())
    latin = TTFont(NOTO + "NotoSans-Bold.ttf")
    wanted = set(range(0x20, 0x250)) | set(range(0x2000, 0x2070)) | set(range(0x20A0, 0x20D0)) | {0xFEFF}
    opts = subset.Options()
    opts.layout_features = ["kern", "mark", "mkmk"]
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=wanted - covered)
    sub.subset(latin)
    latin.save(os.path.join(tmp, "latin.ttf"))
    font = merge.Merger().merge([arabic, os.path.join(tmp, "latin.ttf")])
    for rec in font["name"].names:
        if rec.nameID in (1, 16):
            rec.string = NAME
        elif rec.nameID in (3, 4):
            rec.string = NAME + " Bold"
        elif rec.nameID == 6:
            rec.string = "OpenShortsArabic-Bold"
        elif rec.nameID in (2, 17):
            rec.string = "Bold"
    font.save(OUT)
print("wrote", OUT, os.path.getsize(OUT))
