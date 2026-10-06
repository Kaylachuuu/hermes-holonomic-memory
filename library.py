"""Reference libraries: stores of material to work from, kept apart from her own memory.

A library is built from a folder of files (notes, manuals, source code) and has its own plates, beside the
memory store and never mixed into it.  What is in a library does not fade, is not reflected on, is not dreamt
about and belongs to nobody: it is reference material, not experience.

Nothing from a library reaches a conversation unless it has been opened for that conversation.  A new
conversation starts with none open, so working on a project one evening does not put its details into the
next morning's talk.

    <data>/libraries/<name>/holonomic.db ...   the library's own store
    <data>/libraries/<name>/library.json       where it came from and which files are in it
    <data>/libraries/open.json                 which libraries each conversation has open
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

LIBRARY_DEFAULTS: Dict[str, Any] = {
    "library_k": 4,                   # pieces given to her per message from the open libraries
    "library_min_score": 0.3,         # below this a piece is not given
    "library_score_band": 0.2,        # and it must score within this much of the best piece
    "library_lexical_weight": 0.35,   # exact words (a register, an interrupt number) count for more than in memory
    "library_context_chars": 3600,    # budget for the block of reference material in a message
    "library_piece_chars": 1100,      # a file is cut into pieces of about this size, at headings and paragraphs
    "library_max_file_mb": 40,        # a larger file is left out, and said to be
    "library_max_files": 5000,        # a folder with more files than this is refused: it is probably the wrong folder
    "library_remember_days": 30,      # how long "last time you used X" is worth mentioning
}

PIECE = "reference"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")

# Read as plain text.  Anything else is tried as text and left out if it turns out not to be.
_SKIP_DIRS = {".git", ".svn", ".hg", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode"}
_BINARY = {".exe", ".com", ".dll", ".so", ".dylib", ".bin", ".img", ".iso", ".o", ".obj", ".lib", ".a", ".class", ".jar", ".pyc",
           ".zip", ".7z", ".rar", ".gz", ".tar", ".xz", ".bz2", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".heic",
           ".ico", ".tif", ".tiff", ".mp3", ".wav", ".flac", ".ogg", ".mp4", ".mkv", ".avi", ".mov", ".ttf", ".otf",
           ".woff", ".woff2", ".db", ".sqlite", ".npy", ".npz", ".pth", ".safetensors", ".onnx", ".doc", ".xls", ".ppt",
           ".xlsx", ".pptx", ".epub", ".chm", ".lnk", ".ovl", ".ovr", ".dsk", ".ima", ".vhd", ".rom", ".tpu", ".scr", ".msi", ".cab"}
_MARKUP = {".html", ".htm", ".xhtml"}
_HEADED = {".md", ".markdown", ".txt", ".rst", ".text", ""}

_LOCK = threading.RLock()
_STORES: Dict[str, Any] = {}           # folder of a library -> its open store
_BUILDS: Dict[str, Dict[str, Any]] = {}  # folder of a library -> progress of a build running in the background


class LibraryError(ValueError):
    """Something the person asking can put right: a name that will not do, a folder that is not there."""


# ----------------------------------------------------------------- names and places

def clean_name(name: str) -> str:
    """A library's name as it is kept: small letters, digits, - and _.  'x86 Assembly' becomes 'x86-assembly'."""
    name = re.sub(r"[^a-z0-9_-]+", "-", str(name or "").strip().lower()).strip("-_")
    if not _NAME_RE.match(name):
        raise LibraryError("A library needs a short name made of letters, digits, - or _ (for example 'x86' or 'fat-filesystems').")
    return name


def root_of(engine) -> Path:
    return Path(engine.path) / "libraries"


def names(root: Path) -> List[str]:
    try:
        return sorted(p.name for p in Path(root).iterdir() if (p / "library.json").exists())
    except OSError:
        return []


def _info_path(root: Path, name: str) -> Path:
    return Path(root) / name / "library.json"


def info(root: Path, name: str) -> Optional[dict]:
    try:
        data = json.loads(_info_path(root, name).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _save_info(root: Path, name: str, data: dict) -> None:
    path = _info_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _need(root: Path, name: str) -> Tuple[str, dict]:
    name = clean_name(name)
    data = info(root, name)
    if data is None:
        have = names(root)
        raise LibraryError(f"There is no library called '{name}'." + (f" There are: {', '.join(have)}." if have else " None has been made yet."))
    return name, data


def store(root: Path, name: str, embedder, cfg: Dict[str, Any]):
    """The library's own store, opened once and kept."""
    from .engine import HolonomicMemory
    key = str((Path(root) / name).resolve())
    with _LOCK:
        got = _STORES.get(key)
        if got is None:
            got = _STORES[key] = HolonomicMemory(Path(root) / name, embedder, dim=int(cfg.get("dim", 4096)),
                                                 plate_capacity=float(cfg.get("plate_capacity", 128.0)))
        return got


def close_all(root: Optional[Path] = None) -> None:
    with _LOCK:
        for key in [k for k in _STORES if root is None or k.startswith(str(Path(root).resolve()))]:
            try:
                _STORES.pop(key).close()
            except Exception:
                pass


def summary(root: Path, name: str) -> dict:
    data = info(root, name) or {}
    files = data.get("files") or {}
    out = {"name": name, "about": data.get("about", ""), "folder": data.get("folder", ""), "files": len(files),
           "pieces": sum(len(f.get("ids") or []) for f in files.values()),
           "built": time.strftime("%Y-%m-%d %H:%M", time.localtime(data["built"])) if data.get("built") else "not yet"}
    running = build_state(root, name)
    if running:
        out["building"] = running
    return out


# ----------------------------------------------------------------- reading files

def _html_text(source: str) -> str:
    from html.parser import HTMLParser

    class Reader(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.out: List[str] = []
            self.skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "nav", "head"):
                self.skip += 1
            elif tag in ("h1", "h2", "h3", "h4"):
                self.out.append("\n\n" + "#" * int(tag[1]) + " ")
            elif tag in ("p", "div", "br", "li", "tr", "pre", "table", "section", "ul", "ol", "dt", "dd"):
                self.out.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "head"):
                self.skip = max(0, self.skip - 1)
            elif tag in ("h1", "h2", "h3", "h4", "p", "pre", "table", "tr"):
                self.out.append("\n")

        def handle_data(self, data):
            if not self.skip:
                self.out.append(data)

    reader = Reader()
    reader.feed(source)
    return "".join(reader.out)


def _pdf_text(path: Path) -> str:
    try:
        import pypdf
        return "\n\n".join((page.extract_text() or "") for page in pypdf.PdfReader(str(path)).pages)
    except ImportError:
        pass
    try:
        import fitz                                   # PyMuPDF
        with fitz.open(str(path)) as doc:
            return "\n\n".join(page.get_text() for page in doc)
    except ImportError:
        raise LibraryError("reading a PDF needs the pypdf package in Hermes' Python (pip install pypdf)")


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    xml = re.sub(r"</w:p>", "\n\n", xml)
    xml = re.sub(r"<w:tab/>", "\t", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


def looks_binary(raw: bytes) -> bool:
    """Whether a file's bytes are a program or data and not something written to be read.

    Two mistakes to avoid, both made here before.  A small DOS .COM program can have no zero byte in it, so that
    test alone lets programs in.  And old DOS text is full of bytes above 127 (box-drawing lines, block shading,
    accented letters in the code page of the day) with the odd control code used as a symbol, so counting those
    against a file throws out a project's own notes and source.  What tells them apart is that text has lines,
    and few control codes among its characters."""
    sample = raw[:16000]
    if not sample:
        return False
    if b"\x00" in sample or sample[:2] in (b"MZ", b"ZM") or sample[:4] == b"\x7fELF":
        return True
    control = sum(1 for c in sample if c < 32 and c not in (9, 10, 13, 12, 26, 27))     # 26: the old DOS end-of-file mark
    if control > max(8, len(sample) // 20):                  # more than 1 in 20 are control codes
        return True
    lines = sample.count(b"\n") + sample.count(b"\r")
    high = sum(1 for c in sample if c >= 127)
    if len(sample) >= 200 and lines * 400 < len(sample) and (control + high) * 10 > len(sample):
        return True                                          # hardly a line break in it, and much that is not plain characters
    return False


def decode(raw: bytes) -> str:
    """Bytes to text.  UTF-8 if it is; otherwise the old DOS code page when the file draws boxes with it, and the
    Windows one when it does not."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    high = [c for c in raw[:16000] if c >= 128]
    boxes = sum(1 for c in high if 0xB0 <= c <= 0xDF)        # shading, lines and corners in code page 437
    try:
        return raw.decode("cp437" if high and boxes * 2 >= len(high) else "cp1252")
    except UnicodeDecodeError:
        return raw.decode("cp437")


def read_file(path: Path, max_bytes: int) -> str:
    """The text of a file.  Raises LibraryError, with the reason, for one that cannot be used."""
    ext = path.suffix.lower()
    if ext in _BINARY:
        raise LibraryError("not a text file")
    size = path.stat().st_size
    if size > max_bytes:
        raise LibraryError(f"larger than {max_bytes // 1_000_000} MB")
    if size == 0:
        raise LibraryError("empty")
    if ext == ".pdf":
        text = _pdf_text(path)
        if len(text.strip()) < 20:
            raise LibraryError("a PDF with no text in it (scanned pages are pictures)")
        return text
    if ext == ".docx":
        return _docx_text(path)
    raw = path.read_bytes()
    # DOS marked the end of a text file with one byte, and editors left whatever was in memory after it.
    body = raw.split(b"\x1a", 1)[0]
    if len(raw) > 200 and len(body) < len(raw) // 10:
        raise LibraryError("not a text file")
    raw = body
    if looks_binary(raw):
        raise LibraryError("not a text file")
    text = decode(raw).replace("\x1a", "")                    # without the old end-of-file mark
    return _html_text(text) if ext in _MARKUP else text


# ----------------------------------------------------------------- cutting into pieces

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_UNDERLINE = re.compile(r"^(=+|-+|~+|\^+)\s*$")


def _sections(text: str, headed: bool) -> List[Tuple[str, str]]:
    """(heading path, text under it).  Markdown and underlined headings open a section; other files are one."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not headed:
        return [("", "\n".join(lines))]
    out: List[Tuple[str, List[str]]] = [("", [])]
    trail: List[Tuple[int, str]] = []
    fenced = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            fenced = not fenced
        level, title = 0, ""
        if not fenced:
            m = _HEADING.match(line)
            if m:
                level, title = len(m.group(1)), m.group(2)
            elif (i + 1 < len(lines) and line.strip() and len(line.strip()) <= 100 and _UNDERLINE.match(lines[i + 1])
                  and len(lines[i + 1].strip()) >= max(3, len(line.strip()) - 2) and (i == 0 or not lines[i - 1].strip())):
                level, title = {"=": 1, "-": 2, "~": 3, "^": 4}[lines[i + 1].strip()[0]], line.strip()
                i += 1
        if level:
            trail = [t for t in trail if t[0] < level] + [(level, title)]
            out.append((" > ".join(t[1] for t in trail), []))
        else:
            out[-1][1].append(line)
        i += 1
    return [(h, "\n".join(body)) for h, body in out if "".join(body).strip()]


def _pack(text: str, size: int) -> List[str]:
    """Paragraphs gathered into pieces of about `size`.  A fenced block of code is kept whole where it fits; a
    paragraph too long for one piece is cut at line ends."""
    blocks, current, fenced = [], [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not line.strip() and not fenced:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line.rstrip())
    if current:
        blocks.append("\n".join(current))
    parts: List[str] = []
    for block in blocks:
        while len(block) > size:                      # cut at a line end, or failing that at a space
            cut = block.rfind("\n", size // 2, size)
            if cut < 0:
                cut = block.rfind(" ", size // 2, size)
            if cut < 0:
                cut = size
            parts.append(block[:cut].rstrip())
            block = block[cut:].lstrip("\n")
        if block.strip():
            parts.append(block)
    pieces, held = [], ""
    for part in parts:
        if held and len(held) + len(part) + 2 > size:
            pieces.append(held)
            held = part
        else:
            held = f"{held}\n\n{part}" if held else part
    if held:
        pieces.append(held)
    return pieces


# Where a routine begins, by kind of source.  The name it captures becomes the piece's heading, so a piece of
# BOOT.ASM is "BOOT.ASM > ReadSectors" and not merely somewhere in BOOT.ASM.
_ASM = {".asm", ".inc", ".s", ".nasm", ".mac", ".a86", ".a"}
_ROUTINE = {
    "asm": re.compile(r"^(?!\.)([A-Za-z_@$?][\w@$?.]*):(?!:)"),                       # a label at the margin; .local ones are inside a routine
    "pascal": re.compile(r"^\s*(?:procedure|function|constructor|destructor)\s+([\w.]+)", re.IGNORECASE),
    "basic": re.compile(r"^\s*(?:DECLARE\s+)?(?:STATIC\s+)?(?:SUB|FUNCTION)\s+([\w.$%&!#]+)", re.IGNORECASE),
    "python": re.compile(r"^(?:async\s+)?(?:def|class)\s+(\w+)"),
    "c": re.compile(r"^[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{?\s*$"),
}
_SOURCE = {**{e: "asm" for e in _ASM}, ".pas": "pascal", ".pp": "pascal", ".dpr": "pascal", ".bas": "basic", ".bi": "basic",
           ".frm": "basic", ".py": "python", ".c": "c", ".h": "c", ".cpp": "c", ".hpp": "c", ".cc": "c"}
_COMMENT = {"asm": ";", "pascal": "{", "basic": "'", "python": "#", "c": "/"}
_BANNER = re.compile(r"^\s*[;'#/*{(]+\s*[-=*_~]{2,}\s*(?:PROCEDURE|PROC|FUNCTION|ROUTINE|CMD|SUB|MACRO)?\s*([A-Za-z_][\w ]{1,40}?)\s*[-=*_~]{2,}", re.IGNORECASE)


def _routines(text: str, kind: str, least: int = 350) -> List[Tuple[str, str]]:
    """(routine name, its lines) for a source file.  A routine starts at its name; comments standing just above the
    name go with it.  A run of short routines or data labels is kept together until it is `least` characters long,
    so that a table of one-line labels does not become a hundred pieces."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start, mark = _ROUTINE[kind], _COMMENT[kind]
    out: List[Tuple[str, List[str]]] = [("", [])]
    for i, line in enumerate(lines):
        m = start.match(line) if not (kind == "c" and line.lstrip().startswith(("if", "for", "while", "switch", "else", "return"))) else None
        if m and sum(len(x) + 1 for x in out[-1][1]) >= least:
            body = out[-1][1]
            carried: List[str] = []
            while body and (not body[-1].strip() or body[-1].lstrip().startswith(mark)):      # the comments above it are its own
                carried.insert(0, body.pop())
            if not "".join(body).strip():                 # nothing but those comments came before: no cut after all
                body.extend(carried)
                body.append(line)
                if not out[-1][0]:
                    out[-1] = (m.group(1), body)
                continue
            name = m.group(1)
            banner = next((b.group(1).strip() for b in (_BANNER.match(c) for c in carried) if b), "")
            out.append((banner if banner and kind == "asm" and len(banner) > len(name) and name.lower() in banner.lower() else name, carried + [line]))
        else:
            if m and not out[-1][0]:
                out[-1] = (m.group(1), out[-1][1])
            out[-1][1].append(line)
    return [(name, "\n".join(body)) for name, body in out if "".join(body).strip()]


_MNEMONIC = re.compile(r"^\s*(?:[A-Za-z_.@$?][\w@$?.]*:)?\s*(?:mov|call|jmp|jn?[zecbla]e?|int|push|pop|ret|xor|cmp|add|sub|lodsb|stosb|"
                       r"cli|sti|db|dw|dd|equ|org|inc|dec|loop|test|shl|shr)\b", re.IGNORECASE)


def _reads_as_assembly(text: str) -> bool:
    lines = [ln for ln in text.split("\n")[:400] if ln.strip() and not ln.lstrip().startswith(";")]
    return len(lines) >= 5 and sum(1 for ln in lines if _MNEMONIC.match(ln)) >= 0.5 * len(lines)


def pieces_of(text: str, name: str, size: int = 1100) -> List[Tuple[str, str]]:
    """(heading path, piece) for a file's text, in order.  Writing is cut at its headings, source at its routines."""
    ext = Path(name).suffix.lower()
    out = []
    kind = _SOURCE.get(ext)
    if kind is None and ext not in _HEADED | _MARKUP | {".pdf", ".docx"} and _reads_as_assembly(text):
        kind = "asm"                                  # BOOT.BAK, DIR.OLD: an earlier copy of a source file, whatever it is called
    if kind:
        for routine, body in _routines(text, kind):
            out += [(routine, piece) for piece in _pack(body, max(200, size))]
        return out
    headed = ext in _HEADED | _MARKUP | {".pdf", ".docx"}
    for heading, body in _sections(text, headed):
        out += [(heading, piece) for piece in _pack(body, max(200, size))]
    return out


# ----------------------------------------------------------------- building

def _files_in(folder: Path, limit: int) -> List[Path]:
    found = []
    for base, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for f in sorted(files):
            if f.startswith(".") or f.startswith("~$"):
                continue
            found.append(Path(base) / f)
            if len(found) > limit:
                raise LibraryError(f"That folder holds more than {limit} files. If it is the right folder, raise library_max_files.")
    return found


def _digest(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_folder(folder: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(folder or "").strip().strip('"'))))
    if not str(folder or "").strip() or not path.is_dir():
        raise LibraryError(f"There is no folder at {folder!r}. Give the full path of the folder that holds the material.")
    path = path.resolve()
    if path == Path(path.anchor) or path == Path.home().resolve():
        raise LibraryError("That is a whole drive or home folder. Put the material in a folder of its own and give that.")
    return path


def create(root: Path, name: str, folder: str, about: str = "") -> dict:
    """Make an empty library that will be built from a folder.  Building is a separate step."""
    name = clean_name(name)
    if info(root, name) is not None:
        raise LibraryError(f"There is already a library called '{name}'. To bring it up to date with its folder, update it.")
    path = check_folder(folder)
    if Path(root).resolve() in [path, *path.parents]:
        raise LibraryError("That folder is inside the memory store itself.")
    data = {"name": name, "folder": str(path), "about": " ".join(str(about or "").split())[:300],
            "created": time.time(), "built": None, "files": {}}
    _save_info(root, name, data)
    return data


def build(root: Path, name: str, embedder, cfg: Dict[str, Any], *, progress: Optional[Callable[[dict], None]] = None,
          stop: Optional[threading.Event] = None, fresh: bool = False) -> dict:
    """Bring a library up to date with its folder: new and changed files are read in, files that are gone are
    taken out, files that have not changed are left alone.  Safe to stop and run again.

    `fresh` reads every file again whether it has changed or not: for when the way files are cut has changed."""
    name, data = _need(root, name)
    folder = Path(data["folder"])
    if not folder.is_dir():
        raise LibraryError(f"The folder this library was built from is not there any more: {folder}")
    lib = store(root, name, embedder, cfg)
    size = int(cfg.get("library_piece_chars", 1100))
    max_bytes = int(float(cfg.get("library_max_file_mb", 40)) * 1_000_000)
    found = _files_in(folder, int(cfg.get("library_max_files", 5000)))
    files: Dict[str, dict] = data.setdefault("files", {})
    report = {"name": name, "added": 0, "changed": 0, "removed": 0, "unchanged": 0, "pieces": 0, "skipped": [],
              "total": len(found), "done": 0, "stopped": False}
    here = {p.relative_to(folder).as_posix(): p for p in found}

    def drop(rel: str) -> None:
        for mid in files.get(rel, {}).get("ids") or []:
            try:
                lib.forget(int(mid))
            except Exception:
                pass
        files.pop(rel, None)

    if fresh:
        for rel in list(files):
            drop(rel)
        _save_info(root, name, data)
    for rel in [r for r in files if r not in here]:
        drop(rel)
        report["removed"] += 1
    for rel, path in here.items():
        if stop is not None and stop.is_set():
            report["stopped"] = True
            break
        report["done"] += 1
        try:
            st = path.stat()
            had = files.get(rel)
            if had and had.get("size") == st.st_size and abs(float(had.get("mtime", 0)) - st.st_mtime) < 1.0:
                report["unchanged"] += 1
                continue
            digest = _digest(path) if st.st_size <= max_bytes else ""
            if had and digest and had.get("sha") == digest:          # touched, not changed
                had.update(size=st.st_size, mtime=st.st_mtime)
                report["unchanged"] += 1
                continue
            text = read_file(path, max_bytes)
            cut = pieces_of(text, path.name, size)
            if not cut:
                raise LibraryError("nothing to read in it")
        except LibraryError as exc:
            if rel in files:
                drop(rel)
            report["skipped"].append({"file": rel, "why": str(exc)})
            continue
        except OSError as exc:
            report["skipped"].append({"file": rel, "why": f"could not be read ({exc.__class__.__name__})"})
            continue
        if rel in files:
            drop(rel)
            report["changed"] += 1
        else:
            report["added"] += 1
        ids: List[int] = []
        try:
            for n, (heading, piece) in enumerate(cut):
                # The heading leads the piece: it is what a question about the piece would mention.
                lead = f"{rel} > {heading}" if heading else rel
                ids += lib.remember(f"{lead}\n{piece}", kind=PIECE, session=f"{rel}#{heading}", whole=True, trust=1.0,
                                    meta={"file": rel, "section": heading, "n": n})
        except Exception as exc:                      # the embedding server went away: keep what is done, say so
            for mid in ids:
                lib.forget(int(mid))
            _save_info(root, name, data)
            raise LibraryError(f"Stopped at {rel}: {exc}. What was read before it is kept; update the library to carry on.") from exc
        files[rel] = {"size": st.st_size, "mtime": st.st_mtime, "sha": digest, "ids": ids}
        report["pieces"] += len(ids)
        _save_info(root, name, data)                  # after each file, so a stopped build loses nothing
        if progress:
            progress(report)
    data["left_out"] = report["skipped"][:2000]       # kept, so that it can be looked at after the build has scrolled away
    if not report["stopped"]:
        data["built"] = time.time()
    _save_info(root, name, data)
    report["in_library"] = {"files": len(files), "pieces": sum(len(f.get("ids") or []) for f in files.values())}
    return report


def build_in_background(root: Path, name: str, embedder, cfg: Dict[str, Any],
                        spawn: Optional[Callable[[Callable[[], None], str], Any]] = None) -> dict:
    """Start a build and come straight back.  `build_state` says how far it has got."""
    name, _ = _need(root, name)
    key = str((Path(root) / name).resolve())
    with _LOCK:
        state = _BUILDS.get(key)
        if state and state.get("running"):
            return dict(state, already=True)
        state = _BUILDS[key] = {"running": True, "started": time.time(), "done": 0, "total": 0, "pieces": 0, "error": ""}

    def run() -> None:
        try:
            report = build(root, name, embedder, cfg, progress=lambda r: state.update(
                done=r["done"], total=r["total"], pieces=r["pieces"]))
            state.update(report=report, done=report["done"], total=report["total"], pieces=report["pieces"])
        except Exception as exc:
            state["error"] = str(exc)[:300]
        finally:
            state.update(running=False, finished=time.time())

    maker = spawn or (lambda target, label: threading.Thread(target=target, name=label, daemon=True))
    thread = maker(run, f"holonomic-library-{name}")
    if getattr(thread, "ident", 0) is None:       # made but not started, as Hermes' own maker leaves it
        thread.start()
    return dict(state)


def build_state(root: Path, name: str) -> Optional[dict]:
    with _LOCK:
        state = _BUILDS.get(str((Path(root) / name).resolve()))
        return {k: v for k, v in state.items() if k != "report"} | ({"report": state["report"]} if state.get("report") else {}) if state else None


def delete(root: Path, name: str) -> bool:
    """Remove a library altogether.  The folder it was built from is not touched."""
    name, _ = _need(root, name)
    key = str((Path(root) / name).resolve())
    with _LOCK:
        if (_BUILDS.get(key) or {}).get("running"):
            raise LibraryError("That library is being built. Wait for it to finish.")
        lib = _STORES.pop(key, None)
        if lib is not None:
            lib.close()
    shutil.rmtree(Path(root) / name)
    data = _read_open(root)
    for entry in list(data["sessions"].values()) + [data.get("last") or {}]:
        if name in (entry.get("names") or []):
            entry["names"] = [n for n in entry["names"] if n != name]
    _write_open(root, data)
    return True


# ----------------------------------------------------------------- which conversation has what open

_OPEN_HERE: Dict[str, List[str]] = {}       # for a conversation with no id of its own: this run of the program only


def _read_open(root: Path) -> dict:
    try:
        data = json.loads((Path(root) / "open.json").read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("sessions"), dict):
            return data
    except (OSError, ValueError):
        pass
    return {"sessions": {}, "last": None}


def _write_open(root: Path, data: dict) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    keep = sorted(data["sessions"].items(), key=lambda kv: kv[1].get("at", 0))[-200:]      # old conversations fall away
    data["sessions"] = {k: v for k, v in keep if v.get("names")}
    tmp = Path(root) / "open.json.part"
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, Path(root) / "open.json")


def opened(root: Path, session: str) -> List[str]:
    """The libraries this conversation has open, leaving out any that no longer exist."""
    with _LOCK:
        got = _OPEN_HERE.get(str(root), []) if not session else (_read_open(root)["sessions"].get(session) or {}).get("names") or []
        have = set(names(root))
        return [n for n in got if n in have]


def open_for(root: Path, session: str, wanted: List[str]) -> List[str]:
    """Open libraries for one conversation, in addition to any it already has open."""
    wanted = [_need(root, n)[0] for n in wanted]
    with _LOCK:
        now = list(dict.fromkeys(opened(root, session) + wanted))
        data = _read_open(root)
        if session:
            data["sessions"][session] = {"names": now, "at": time.time()}
        else:
            _OPEN_HERE[str(root)] = now
        data["last"] = {"names": now, "at": time.time(), "session": session}
        _write_open(root, data)
        return now


def close_for(root: Path, session: str, which: Optional[List[str]] = None) -> List[str]:
    """Close some or all of a conversation's libraries.  Returns what is still open."""
    with _LOCK:
        gone = {clean_name(n) for n in which} if which else None
        left = [n for n in opened(root, session) if gone is not None and n not in gone]
        data = _read_open(root)
        if session:
            data["sessions"][session] = {"names": left, "at": time.time()}
        else:
            _OPEN_HERE[str(root)] = left
        _write_open(root, data)
        return left


def carry_over(root: Path, new: str, parent: str) -> None:
    """The same conversation under a new id (its context was compressed) keeps what it had open."""
    if not new or not parent or new == parent:
        return
    with _LOCK:
        data = _read_open(root)
        had = data["sessions"].get(parent)
        if had and had.get("names") and new not in data["sessions"]:
            data["sessions"][new] = {"names": list(had["names"]), "at": time.time()}
            _write_open(root, data)


def last_used(root: Path, session: str, days: float = 30) -> Optional[dict]:
    """What another, recent conversation last had open: worth a mention at the start of a new one."""
    last = _read_open(root).get("last")
    if not last or not last.get("names") or (session and last.get("session") == session):
        return None
    if time.time() - float(last.get("at", 0)) > days * 86400:
        return None
    have = set(names(root))
    kept = [n for n in last["names"] if n in have]
    return {"names": kept, "when": time.strftime("%Y-%m-%d", time.localtime(last["at"]))} if kept else None


# ----------------------------------------------------------------- whether the user asked

_CUE = re.compile(r"\b(librar(?:y|ies)|data\s?stores?|reference(?:s| material)?)\b", re.IGNORECASE)
_YES = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok|okay|please|please do|go ahead|do it|go for it|open it|open them|use it|use them|"
                  r"yes please|sounds good|that would be great|absolutely|definitely)\b", re.IGNORECASE)
_NO = re.compile(r"\b(no|not|don'?t|do not|never|nope|close|stop)\b", re.IGNORECASE)
_ASKED: Dict[str, List[str]] = {}       # conversation -> libraries she tried to open unasked, and may be told yes to


def _mentions(message: str, name: str) -> bool:
    plain = re.sub(r"[^a-z0-9]+", " ", message.lower())
    return bool(re.search(r"\b" + r"\s*".join(re.escape(part) for part in re.split(r"[-_]+", name) if part) + r"\b", plain))


def user_asked(root: Path, session: str, message: str, name: str) -> bool:
    """Whether the user's own message asks for this library to be opened.

    Being told in her instructions to open a library only when asked was not enough: asked a question the library
    could answer, she reasoned that opening it was 'logical' and did.  An open library feeds every message of the
    conversation, so that is the user's to decide, and it is decided here from the user's words: the message names
    the library and speaks of a library (or data store, or reference material); or it speaks of all the libraries;
    or it says yes after she asked whether to open this one."""
    message = " ".join(str(message or "").split())
    if not message:
        return False
    if _CUE.search(message) and (_mentions(message, name) or re.search(r"\b(all|every|both)\b", message, re.IGNORECASE)):
        return not re.search(r"\b(don'?t|do not|never)\s+(open|use|load)\b", message, re.IGNORECASE)
    with _LOCK:
        waiting = name in _ASKED.get(f"{root}|{session}", [])
    return waiting and len(message) <= 80 and bool(_YES.search(message)) and not _NO.search(message)


def note_refused(root: Path, session: str, wanted: List[str]) -> None:
    with _LOCK:
        _ASKED[f"{root}|{session}"] = list(dict.fromkeys(wanted))


def clear_refused(root: Path, session: str) -> None:
    with _LOCK:
        _ASKED.pop(f"{root}|{session}", None)


# ----------------------------------------------------------------- looking things up

def _source(meta: dict) -> str:
    return meta.get("file", "") + (f" > {meta['section']}" if meta.get("section") else "")


def search(root: Path, which: List[str], query: str, embedder, cfg: Dict[str, Any], *, k: Optional[int] = None,
           floor: Optional[float] = None) -> List[dict]:
    """The pieces of the given libraries that best answer a query, best first.  A piece linked on the plates to a
    good match (the next part of the same section) comes with it."""
    query = " ".join(str(query or "").split())
    if not query or not which:
        return []
    k = int(k or cfg.get("library_k", 4))
    floor = float(cfg.get("library_min_score", 0.3)) if floor is None else float(floor)
    found: List[dict] = []
    for name in which:
        if info(root, name) is None:
            continue
        lib = store(root, name, embedder, cfg)
        for hit in lib.recall(query[:2000], k=k * 4, min_score=floor, lexical=float(cfg.get("library_lexical_weight", 0.35)),
                              only_kinds=(PIECE,)):
            meta = hit.meta or {}
            text = hit.text.split("\n", 1)[1] if "\n" in hit.text else hit.text       # without the heading that leads it
            found.append({"library": name, "file": meta.get("file", ""), "section": meta.get("section", ""),
                          "source": _source(meta), "text": text, "score": round(float(hit.score), 3),
                          "linked": float(hit.direct) < floor <= float(hit.score), "id": hit.id})
    found.sort(key=lambda f: -f["score"])
    if found:
        best = found[0]["score"]
        found = [f for f in found if f["score"] >= best - float(cfg.get("library_score_band", 0.2))]
    # The same passage kept in several files (a backup, an earlier version of the project) is given once, with a
    # note of where else it is.  Four copies of one boot sector are one answer, and would leave no room for a second.
    kept: List[dict] = []
    for f in found:
        words = set(re.findall(r"[a-z0-9_]+", f["text"].lower()))
        twin = next((k for k in kept if len(words & k["_words"]) >= 0.8 * max(1, len(words | k["_words"]))), None)
        if twin is not None:
            rank = lambda x: (bool(x["section"]), Path(x["file"]).suffix.lower() in _SOURCE, -x["file"].count("/"))
            if rank(f) > rank(twin):                      # of two copies, show the one that names its routine, in the real source file
                for key in ("file", "section", "source", "library", "id"):
                    f[key], twin[key] = twin[key], f[key]
            if f["source"] != twin["source"] and f["source"] not in twin["also_in"]:
                twin["also_in"].append(f["source"])
            continue
        kept.append(dict(f, _words=words, also_in=[]))
    for f in kept:
        f.pop("_words")
    return kept[:k]


def context_block(root: Path, session: str, query: str, embedder, cfg: Dict[str, Any]) -> Tuple[str, List[str]]:
    """What the open libraries have on a message, as a block for her context, and notes for checking."""
    which = opened(root, session)
    if not which:
        return "", []
    notes = [f"reference libraries open in this conversation: {', '.join(which)}"]
    found = search(root, which, query, embedder, cfg)
    head = (f"## Reference material (from the librar{'ies' if len(which) > 1 else 'y'} open in this conversation: "
            f"{', '.join(which)})\n")
    if not found:
        notes.append("nothing in them matched this message")
        return head + "Nothing in it matched this message. Look something up with holonomic_library (action 'search') if you need it.", notes
    budget, lines = int(cfg.get("library_context_chars", 3600)), []
    for f in found:
        text = f["text"].strip()
        room = budget - sum(len(x) for x in lines)
        if room < 200:
            break
        if len(text) > room:
            text = text[:room].rsplit("\n", 1)[0].rstrip() + "\n[...]"
        also = f" (the same passage is also in: {', '.join(f['also_in'][:4])})" if f.get("also_in") else ""
        lines.append(f"[{f['library']}: {f['source']}]{also}\n{text}\n")
    notes.append("pieces given: " + "; ".join(f"{f['library']}: {f['source']} ({f['score']})" for f in found[:len(lines)]))
    return (head + "This is reference material, not something you remember. Rely on it for exact details over your "
            "own recollection, and say which file a detail came from if asked.\n\n" + "\n".join(lines).rstrip()), notes


def prompt_block(root: Path, session: str, cfg: Dict[str, Any]) -> str:
    """What she is told about libraries at the start of a conversation."""
    have = names(root)
    if not have:
        return ("\n\n# Reference libraries\nYou can keep libraries of reference material, apart from your memory. None "
                "has been made yet. When the user asks you to make one from a folder, use the holonomic_library tool "
                "(action 'create').")
    listed = []
    for n in have:
        s = summary(root, n)
        listed.append(f"- {n}: {s['pieces']} piece{'s' if s['pieces'] != 1 else ''} from {s['files']} file{'s' if s['files'] != 1 else ''}"
                      + (f". {s['about']}" if s["about"] else "")
                      + (" (still being built)" if s.get("building", {}).get("running") else ""))
    now = opened(root, session)
    if now:
        state = f"Open in this conversation: {', '.join(now)}."
    else:
        state = "None is open in this conversation, so nothing from them reaches you."
        last = last_used(root, session, float(cfg.get("library_remember_days", 30)))
        if last:
            state += (f" The last conversation that used any had {', '.join(last['names'])} open ({last['when']}). If the user "
                      "seems to be picking that work up again, ask whether to open it.")
    return ("\n\n# Reference libraries\nReference material kept apart from your memory:\n" + "\n".join(listed) + "\n" + state
            + " When the user asks a question that one of these would answer, you may look it up once with the "
            "holonomic_library tool (action 'search', with the library in 'names') and say which library it came from. "
            "Opening a library (action 'open') makes it feed every message of the conversation, so that is the user's "
            "decision: it works only when the user's own message asks for it. If you think one should be open, ask them.")
