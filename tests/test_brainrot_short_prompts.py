import ast
import json
import os


def _helpers():
    path = os.path.join(os.path.dirname(__file__), "..", "saasshorts.py")
    with open(path, encoding="utf-8") as source:
        module = ast.parse(source.read())
    names = {"_brainrot_script_prompt", "_original_mascot_image_prompt"}
    functions = [node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"json": json}
    exec(compile(ast.Module(body=functions, type_ignores=[]), path, "exec"), namespace)
    return namespace


def test_brainrot_prompt_uses_verified_trends_only_as_inspiration():
    prompt = _helpers()["_brainrot_script_prompt"](
        {"trends": [{"name": "Trend Example", "evidence": "recent coverage"}]},
        2,
        "en",
    )

    assert "Trend Example" in prompt
    assert "Do not use any existing" in prompt
    assert "ORIGINAL_MASCOT:" in prompt
    assert "Do not include an advertising CTA" in prompt
    assert "Create 2 original" in prompt
    assert "each short exactly 22 seconds" in prompt


def test_brainrot_prompt_supports_spanish_scripts():
    prompt = _helpers()["_brainrot_script_prompt"]({"trends": []}, 1, "es")

    assert "in Spanish" in prompt


def test_original_mascot_prompt_explicitly_requires_distinct_kid_safe_art():
    prompt = _helpers()["_original_mascot_image_prompt"]("a smiling lemon robot")

    assert "a smiling lemon robot" in prompt
    assert "not similar to" in prompt
    assert "No text, logo, watermark, human child" in prompt
