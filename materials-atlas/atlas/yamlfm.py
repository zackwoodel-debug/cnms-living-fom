"""A small, deterministic YAML writer for note frontmatter.

Not PyYAML's dumper, because three properties matter more here than generality:

* **Exact numbers.** A float is written as Python's shortest round-trip repr,
  always with a decimal point or exponent mark that every YAML parser reads as a
  float (``1.0e-05``, never ``1e-05``, which YAML 1.1 reads as a string). The
  check re-reads every value and compares it with the database for equality.
* **Byte-identical output.** Keys keep the order the renderer gives them; there
  is no flow-style heuristic that changes with content.
* **No surprises in strings.** Every string is double-quoted with JSON escaping,
  which is valid YAML, so a material called ``No`` or ``1e5`` stays a string.
"""

from __future__ import annotations

import json
import math


def _float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"{value!r} cannot be written as a stored value")
    text = repr(float(value))
    mantissa, _, exponent = text.partition("e")
    if exponent and "." not in mantissa:
        text = f"{mantissa}.0e{exponent}"
    return text


def scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"no YAML form for {type(value).__name__}")


def _emit(value, indent: int, out: list[str], key: str | None) -> None:
    pad = "  " * indent
    prefix = f"{pad}{key}:" if key is not None else None
    if isinstance(value, dict):
        if not value:
            out.append(f"{prefix} {{}}" if prefix else f"{pad}{{}}")
            return
        if prefix:
            out.append(prefix)
        for k, v in value.items():
            _emit(v, indent + (1 if prefix else 0), out, str(k))
        return
    if isinstance(value, (list, tuple)):
        if not value:
            out.append(f"{prefix} []" if prefix else f"{pad}[]")
            return
        if prefix:
            out.append(prefix)
        item_pad = "  " * (indent + (1 if prefix else 0))
        for item in value:
            if isinstance(item, dict) and item:
                lines: list[str] = []
                _emit(item, 0, lines, None)
                out.append(f"{item_pad}- {lines[0]}")
                out.extend(f"{item_pad}  {line}" for line in lines[1:])
            elif isinstance(item, (list, tuple)):
                raise TypeError("nested lists are not used in frontmatter")
            else:
                out.append(f"{item_pad}- {scalar(item)}")
        return
    out.append(f"{prefix} {scalar(value)}" if prefix else f"{pad}{scalar(value)}")


def dump(mapping: dict) -> str:
    """The frontmatter block, including the '---' fences and a trailing newline."""
    lines: list[str] = []
    for key, value in mapping.items():
        _emit(value, 0, lines, str(key))
    return "---\n" + "\n".join(lines) + "\n---\n"
