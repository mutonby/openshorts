"""A small TOML writer for livewatch.toml (tables of scalars/lists + [[channel]])."""
from __future__ import annotations

import math
from typing import Any, Dict, List

HEADER = ("# livewatch settings. Edited from the web UI (python -m livewatch ui) or by hand.\n"
          "# Secrets (API keys) live in .env next to this file, never here.\n")


def _str(s: str) -> str:
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("numbers must be finite")
        return repr(v)
    if isinstance(v, str):
        return _str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


_BARE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def key(k: str) -> str:
    return k if k and set(k) <= _BARE else _str(k)


def _table(d: Dict[str, Any]) -> List[str]:
    return [f"{key(k)} = {value(v)}" for k, v in d.items() if v is not None]


def dumps(data: Dict[str, Any]) -> str:
    lines = [HEADER]
    for name, table in data.items():
        if name == "channel":
            continue
        if not isinstance(table, dict):
            raise TypeError(f"[{name}] must be a table")
        lines.append(f"[{key(name)}]")
        lines += _table(table)
        lines.append("")
    for ch in data.get("channel", []) or []:
        lines.append("[[channel]]")
        lines += _table(ch)
        lines.append("")
    return "\n".join(lines)
