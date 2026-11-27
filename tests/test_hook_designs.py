import base64
import io

import pytest
from PIL import Image
from starlette.testclient import TestClient

import hooks


DESIGN = {
    "shape": "box",
    "text_color": "#FFFFFF",
    "background_color": "#DA3838",
    "outline_color": "#2139EA",
    "outline_width": 10,
    "corner_radius": 24,
    "text_case": "uppercase",
}


def artwork(text, design, size="M"):
    output = io.BytesIO()
    hooks.create_hook_image(text, 972, output, font="montserrat",
                            font_scale=hooks.HOOK_SIZE_SCALE[size], design=design)
    return output.getvalue()


def test_hook_artwork_uses_custom_box_colors_outline_corners_and_case():
    png = artwork("Mix words", DESIGN)
    assert png == artwork("MIX WORDS", DESIGN)
    assert png != artwork("Mix words", {**DESIGN, "text_case": "as_written"})
    with Image.open(io.BytesIO(png)) as image:
        pixels = set(image.get_flattened_data())
        assert (218, 56, 56, 255) in pixels  # background
        assert (33, 57, 234, 255) in pixels  # outline
        assert (255, 255, 255, 255) in pixels  # text
        assert image.getpixel((20, 20))[3] == 0

    with Image.open(io.BytesIO(artwork("Mix words", {**DESIGN, "corner_radius": 0}))) as image:
        assert image.getpixel((20, 20)) == (218, 56, 56, 255)
    with Image.open(io.BytesIO(artwork("Mix words", {**DESIGN, "shape": "none"}))) as image:
        assert (218, 56, 56, 255) not in set(image.get_flattened_data())


def test_hook_preview_returns_the_render_artwork_for_both_layers():
    app = pytest.importorskip("app")
    client = TestClient(app.app)

    response = client.post("/api/hook/design-preview", json={
        "width": 1080, "text": "Opening hook", "cta_text": "Follow here", "cta_size": "S",
        "font": "montserrat", "design": DESIGN,
    })

    assert response.status_code == 200, response.text
    assert "no-store" in response.headers["cache-control"]
    for field, text, size in [("opening", "Opening hook", "M"), ("closing", "Follow here", "S")]:
        layer = response.json()[field]
        png = base64.b64decode(layer["src"].removeprefix("data:image/png;base64,"))
        assert png == artwork(text, DESIGN, size)
        with Image.open(io.BytesIO(png)) as image:
            assert image.size == (layer["width"], layer["height"])


def test_hook_design_validation_rejects_bad_colors_and_geometry():
    app = pytest.importorskip("app")
    client = TestClient(app.app)

    for (field, value) in [("outline_color", "red"), ("outline_width", 21), ("corner_radius", -1)]:
        response = client.post("/api/hook/design-preview", json={
            "width": 1080, "text": "Opening hook", "design": {**DESIGN, field: value},
        })
        assert response.status_code == 422
