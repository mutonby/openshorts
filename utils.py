import json
import re


def extract_json(text):
    """
    Extract the first complete JSON object from text, handling:
    - Markdown code blocks (```json ... ``` or ``` ... ```)
    - Extra text before/after the JSON object
    - Multiple JSON objects in the response
    Returns parsed JSON dict, or None if no valid JSON found.
    """
    if not text:
        return None

    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```$", "", text)
    text = text.strip()

    start_idx = text.find("{")
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end_idx = -1

    for i in range(start_idx, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\":
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx == -1:
        return None

    json_str = text[start_idx:end_idx + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None
