"""Settings that can be edited from the web UI: schema, validation, and saving
changes back into config.toml without losing its comments.

Every field says how a change takes effect (``effect``):

    live    immediately
    next    from the next utterance
    claude  the Claude process is restarted (~2 s)
    index   the notes are re-indexed (and Claude restarted: its folder / deny rules change)
    stt     the speech model is reloaded (~10 s)
    audio   the audio device is reopened
    restart only after restarting meeting-copilot
"""
from __future__ import annotations

import json
import math
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

EFFECTS = {
    "live": "instant",
    "next": "next utterance",
    "claude": "restarts Claude (~2 s)",
    "index": "re-indexes notes",
    "stt": "reloads speech model (~10 s)",
    "audio": "reopens audio",
    "restart": "after restarting the app",
}


@dataclass(frozen=True)
class Field:
    section: str
    key: str
    kind: str  # bool | int | float | str | text | list | choice
    label: str
    effect: str
    help: str = ""
    choices: tuple = ()
    suggest: tuple = ()  # free text with suggestions (datalist)
    min: Optional[float] = None
    max: Optional[float] = None

    @property
    def id(self) -> str:
        return f"{self.section}.{self.key}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["choices"] = list(self.choices)
        d["suggest"] = list(self.suggest)
        d["effect_label"] = EFFECTS[self.effect]
        return d


FIELDS: list[Field] = [
    # -- audio ------------------------------------------------------------- #
    Field("audio", "device", "str", "Audio source", "audio",
          '"loopback" = whatever the default output device plays (Windows). '
          'Or "loopback:<output name>", or part of an input device name.'),
    Field("audio", "silence_threshold", "float", "Speech threshold (RMS)", "next",
          "A 30 ms frame louder than this counts as speech. Lower it if quiet voices are missed.",
          min=0.0005, max=0.5),
    Field("audio", "end_silence_sec", "float", "End of speech after (s)", "next",
          "Silence that ends an utterance. Higher = fewer sentences cut in half, but slower answers.",
          min=0.2, max=5),
    Field("audio", "max_segment_sec", "float", "Longest utterance (s)", "next", min=2, max=60),
    Field("audio", "min_segment_sec", "float", "Shortest utterance (s)", "next",
          "Shorter blips are dropped.", min=0, max=5),
    Field("audio", "preroll_sec", "float", "Pre-roll (s)", "next",
          "Audio kept from before speech starts.", min=0, max=2),
    # -- speech to text ---------------------------------------------------- #
    Field("stt", "language", "str", "Language", "next",
          'Empty = auto-detect. "en", "zh", ...', suggest=("", "en", "zh")),
    Field("stt", "initial_prompt", "text", "Names and jargon", "next",
          "Helps Whisper spell names, products and acronyms from your meetings."),
    Field("stt", "model", "str", "Whisper model", "stt",
          suggest=("large-v3-turbo", "large-v3", "medium", "small", "mlx-community/whisper-large-v3-turbo")),
    Field("stt", "backend", "choice", "Backend", "stt", choices=("faster-whisper", "mlx")),
    Field("stt", "device", "choice", "Run on", "stt", "faster-whisper only.", choices=("auto", "cuda", "cpu")),
    Field("stt", "compute_type", "choice", "Precision", "stt",
          "int8 for Pascal GPUs (GTX 10xx, Quadro P) and CPU; float16 / int8_float16 on RTX.",
          choices=("int8", "int8_float16", "float16", "float32")),
    # -- Claude ------------------------------------------------------------ #
    Field("claude", "auto_answer", "bool", "Answer detected questions automatically", "live",
          "Off = only answer when you click a line, press Space or type a question."),
    Field("claude", "about_me", "text", "About me", "live",
          "Your role, so answers sound like you."),
    Field("claude", "context_minutes", "float", "Transcript context (min)", "live",
          "How much recent transcript is sent with each question.", min=0.5, max=60),
    Field("claude", "model", "str", "Model", "claude", "haiku is fastest; sonnet gives better answers.",
          suggest=("haiku", "sonnet", "opus")),
    Field("claude", "effort", "choice", "Effort", "claude", choices=("", "low", "medium", "high")),
    Field("claude", "thinking", "bool", "Extended thinking", "claude",
          "Better reasoning, but 5-10 s before the first word."),
    Field("claude", "allow_vault_tools", "bool", "Let Claude search the notes itself", "claude",
          "Read/Grep/Glob in the notes folder when the retrieved sections are not enough. "
          "Excluded folders stay blocked."),
    Field("claude", "mode", "choice", "Process mode", "claude",
          "persistent = one long-lived process (fast); oneshot = a new process per answer (+1.5 s).",
          choices=("auto", "persistent", "oneshot")),
    Field("claude", "session_max_minutes", "float", "Recycle session after (min)", "claude", min=1, max=600),
    Field("claude", "turn_timeout_sec", "float", "Answer timeout (s)", "claude", min=10, max=900),
    Field("claude", "cli", "str", "Claude CLI", "claude", "Path to the claude command."),
    # -- notes ------------------------------------------------------------- #
    Field("vault", "path", "str", "Notes folder", "index",
          "Any folder of .md files, e.g. an Obsidian vault. Relative paths are relative to the config file."),
    Field("vault", "exclude", "list", "Exclude (one glob per line)", "index",
          'Never indexed, and Claude cannot read them. e.g. "NDA/**", "Private/**".'),
    Field("vault", "name", "str", "Obsidian vault name", "live",
          "For the obsidian:// links. Empty = the folder name."),
    Field("vault", "embedding_model", "str", "Embedding model", "index",
          'Local semantic search. Empty = keyword search (BM25) only.',
          suggest=("", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "BAAI/bge-m3")),
    Field("vault", "top_k", "int", "Notes per question", "live", min=1, max=30),
    Field("vault", "per_note", "int", "Sections per note", "live", min=1, max=10),
    # -- web UI ------------------------------------------------------------ #
    Field("ui", "host", "str", "Host", "restart", "Keep 127.0.0.1 so only this computer can open the UI."),
    Field("ui", "port", "int", "Port", "restart", min=1, max=65535),
]

BY_ID = {f.id: f for f in FIELDS}
SECTION_TITLES = {"audio": "Audio", "stt": "Speech to text", "claude": "Claude", "vault": "Notes", "ui": "Web UI"}


class SettingsError(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def coerce(f: Field, value: Any) -> Any:
    """Return value converted to the field's type, or raise ValueError."""
    k = f.kind
    if k == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError("must be true or false")
    if k in ("int", "float"):
        if isinstance(value, bool):
            raise ValueError("must be a number")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("must be a number")
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError("must be a number") from None
        if not math.isfinite(num):
            raise ValueError("must be a number")
        if k == "int":
            if num != int(num):
                raise ValueError("must be a whole number")
            num = int(num)
        if f.min is not None and num < f.min:
            raise ValueError(f"must be at least {f.min:g}")
        if f.max is not None and num > f.max:
            raise ValueError(f"must be at most {f.max:g}")
        return num
    if k in ("str", "text", "choice"):
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError("must be text")
        if len(value) > 4000:
            raise ValueError("is too long")
        if k == "str":
            value = value.strip()
        if k == "choice" and value not in f.choices:
            raise ValueError(f"must be one of {', '.join(repr(c) for c in f.choices)}")
        return value
    if k == "list":
        if isinstance(value, str):
            value = value.splitlines()
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError("must be a list of text")
        return [v.strip() for v in value if v.strip()]
    raise ValueError(f"unknown field type {k}")


def validate_changes(changes: Any) -> dict[str, dict[str, Any]]:
    """{"claude": {"model": "sonnet"}, ...} -> the same, with values coerced.

    Raises SettingsError with one message per bad field (nothing is applied then).
    """
    if not isinstance(changes, dict):
        raise SettingsError({"changes": "must be an object"})
    out: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for section, values in changes.items():
        if not isinstance(values, dict):
            errors[str(section)] = "must be an object"
            continue
        for key, value in values.items():
            f = BY_ID.get(f"{section}.{key}")
            if f is None:
                errors[f"{section}.{key}"] = "is not an editable setting"
                continue
            try:
                out.setdefault(section, {})[key] = coerce(f, value)
            except ValueError as e:
                errors[f.id] = f"{f.label} {e}"
    if errors:
        raise SettingsError(errors)
    return out


# --------------------------------------------------------------------------- #
# Writing config.toml (keeps comments, order and formatting of other lines)
# --------------------------------------------------------------------------- #
_SECTION_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")


def toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v) if v != int(v) or abs(v) >= 1e16 else f"{v:.1f}"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)  # JSON strings are valid TOML basic strings
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def _key_re(key: str) -> re.Pattern:
    return re.compile(r"^(\s*)(" + re.escape(key) + r'|"' + re.escape(key) + r'")\s*=')


def _value_extent(lines: list[str], i: int, key: str) -> tuple[int, str]:
    """For `key = value  # comment` starting at line i, return (number of lines the
    value spans, trailing comment). Uses tomllib to find where the value ends."""
    for n in range(1, min(len(lines) - i, 200) + 1):
        chunk = "\n".join(lines[i:i + n])
        try:
            tomllib.loads(chunk)
        except tomllib.TOMLDecodeError:
            continue
        last = lines[i + n - 1]
        # the comment, if any, is the shortest '#'-suffix whose removal still parses
        comment = ""
        for m in re.finditer("#", last):
            head = "\n".join(lines[i:i + n - 1] + [last[:m.start()]])
            try:
                tomllib.loads(head)
            except tomllib.TOMLDecodeError:
                continue
            comment = last[m.start():]
            break
        return n, comment
    return 1, ""


def update_toml_text(text: str, updates: dict[str, dict[str, Any]]) -> str:
    """Set section.key = value in TOML text, editing only the affected lines."""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    for section, values in updates.items():
        for key, value in values.items():
            lines = _set_one(lines, section, key, value)
    out = "\n".join(lines)
    tomllib.loads(out)  # never write a file we can't read back
    return out.replace("\n", newline)


def _set_one(lines: list[str], section: str, key: str, value: Any) -> list[str]:
    start = end = None
    for i, line in enumerate(lines):
        m = _SECTION_RE.match(line)
        if m:
            if start is not None:
                end = i
                break
            if m.group(1).strip() == section:
                start = i + 1
    if start is None:  # section missing: append it
        while lines and not lines[-1].strip():
            lines.pop()
        return lines + ["", f"[{section}]", f"{key} = {toml_value(value)}", ""]
    if end is None:
        end = len(lines)
    key_re = _key_re(key)
    for i in range(start, end):
        m = key_re.match(lines[i])
        if not m:
            continue
        n, comment = _value_extent(lines, i, key)
        new = f"{m.group(1)}{key} = {toml_value(value)}"
        if comment:
            # keep the comment aligned where it was when the new value fits
            old_col = len(lines[i + n - 1]) - len(comment)
            new = new.ljust(old_col) if len(new) < old_col else new + "  "
            new += comment
        return lines[:i] + [new] + lines[i + n:]
    # key missing: insert after the section's last non-blank line
    j = end
    while j > start and not lines[j - 1].strip():
        j -= 1
    return lines[:j] + [f"{key} = {toml_value(value)}"] + lines[j:]


def save_changes(path: Path, updates: dict[str, dict[str, Any]], template: Optional[Path] = None) -> Path:
    """Write updates into path (created from template, or empty, if it doesn't exist)."""
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    elif template is not None and template.is_file():
        text = template.read_text(encoding="utf-8")
    else:
        text = ""
    new = update_toml_text(text, updates)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(new.encode("utf-8"))
    os.replace(tmp, path)
    return path


def current_values(cfg: dict, raw_vault_path: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in FIELDS:
        v = cfg.get(f.section, {}).get(f.key)
        if f.id == "vault.path":
            v = raw_vault_path
        if f.kind == "list":
            v = list(v or [])
        elif v is None:
            v = "" if f.kind in ("str", "text", "choice") else v
        out.setdefault(f.section, {})[f.key] = v
    return out


def schema() -> dict:
    return {
        "fields": [f.to_dict() for f in FIELDS],
        "sections": SECTION_TITLES,
        "effects": EFFECTS,
    }
