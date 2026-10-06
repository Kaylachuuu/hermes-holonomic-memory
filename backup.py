"""Backing up and restoring the whole store: memories, images, dreams, libraries and settings, as one zip file.

A backup can be taken while Hermes is running: each database is copied through SQLite's own backup, which gives
a consistent copy of a store that is being written to.  A restore needs Hermes closed.  It does not delete what
is there: the present store is set aside beside it, so a restore can itself be undone.

    holonomic/...        the data folder
    holonomic.json       the settings
    backup.json          when it was made, and by which version
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PREFIX = "holonomic_"
_LEAVE_OUT = ("-wal", "-shm", ".part", ".tmp")          # SQLite's working files and our own half-written ones


class BackupError(RuntimeError):
    """Something the person can put right; the store itself has not been harmed."""


def plugin_version() -> str:
    """The plugin's version, read from plugin.yaml beside this file.  Hermes loads the command line's modules into
    a package of its own making, which has no __version__ to import."""
    try:
        for line in (Path(__file__).resolve().parent / "plugin.yaml").read_text(encoding="utf-8").splitlines():
            if line.startswith("version:"):
                return line.split(":", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return ""


def default_folder(cfg: Optional[Dict[str, Any]] = None) -> Path:
    given = str((cfg or {}).get("backup_dir") or "").strip()
    if given:
        return Path(os.path.expandvars(os.path.expanduser(given)))
    documents = Path.home() / "Documents"
    return (documents if documents.is_dir() else Path.home()) / "holonomic-backups"


def _stamp() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def _copy_database(source: Path, target: Path) -> None:
    """A consistent copy of a database that may be in use, checked before it is trusted."""
    src = sqlite3.connect(str(source), timeout=30.0)      # a plain path: a file URI trips on Windows drives and odd characters
    try:
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)
            verdict = dst.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            dst.close()
    finally:
        src.close()
    if verdict != "ok":
        raise BackupError(f"The copy of {source.name} did not pass its check ({verdict}). No backup was written.")


def backups(folder: Path) -> List[dict]:
    """The backups in a folder, newest first."""
    try:
        found = sorted((p for p in Path(folder).glob(PREFIX + "*.zip") if p.is_file()), key=lambda p: p.name, reverse=True)
    except OSError:
        return []
    return [{"file": str(p), "name": p.name, "bytes": p.stat().st_size} for p in found]


def backup(hermes_home: Path, folder: Path, *, label: str = "", keep: int = 10, version: str = "") -> dict:
    home = Path(hermes_home)
    data = home / "holonomic"
    if not (data / "holonomic.db").exists():
        raise BackupError(f"There is no memory store at {data} to back up.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if data.resolve() in [folder.resolve(), *folder.resolve().parents]:
        raise BackupError("The backup folder is inside the data folder. Choose another place with --to.")
    tag = "".join(c if c.isalnum() or c in "._-" else "-" for c in label.strip())[:40]
    final = folder / f"{PREFIX}{_stamp()}{'_' + tag if tag else ''}.zip"
    part = final.with_suffix(".zip.part")
    count = size = 0
    try:
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            for path in sorted(data.rglob("*")):
                if not path.is_file() or path.name.endswith(_LEAVE_OUT):
                    continue
                name = "holonomic/" + path.relative_to(data).as_posix()
                if path.suffix == ".db":
                    fd, tmp = tempfile.mkstemp(dir=str(folder), prefix=".copy-", suffix=".tmp")
                    os.close(fd)
                    try:
                        os.unlink(tmp)
                        _copy_database(path, Path(tmp))
                        z.write(tmp, name)
                    finally:
                        if os.path.exists(tmp):
                            os.unlink(tmp)
                else:
                    try:
                        z.write(path, name)
                    except FileNotFoundError:             # removed while we were at it (a half-written file tidied up)
                        continue
                count += 1
                size += path.stat().st_size if path.exists() else 0
            config = home / "holonomic.json"
            if config.exists():
                z.write(config, "holonomic.json")
                count += 1
            z.writestr("backup.json", json.dumps({"made": time.time(), "made_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                  "plugin_version": version, "files": count, "label": label}, indent=1))
        with zipfile.ZipFile(part) as z:                  # read it back before calling it a backup
            broken = z.testzip()
            names = set(z.namelist())
        if broken or "holonomic/holonomic.db" not in names or len(names) != count + 1:
            raise BackupError("The backup did not read back correctly. Nothing was kept; the store itself is untouched.")
        os.replace(part, final)
    except BaseException:
        if part.exists():
            part.unlink()
        raise
    removed = []
    if keep and keep > 0:
        for old in backups(folder)[keep:]:
            try:
                os.unlink(old["file"])
                removed.append(old["name"])
            except OSError:
                pass
    return {"file": str(final), "files": count, "bytes": final.stat().st_size, "store_bytes": size, "removed": removed}


def inspect(file: Path) -> dict:
    """What a backup is, without unpacking it.  Raises BackupError for a file that is not one of ours."""
    file = Path(file)
    if not file.is_file():
        raise BackupError(f"There is no backup at {file}.")
    try:
        with zipfile.ZipFile(file) as z:
            names = z.namelist()
            info = json.loads(z.read("backup.json")) if "backup.json" in names else {}
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        raise BackupError(f"{file.name} cannot be read as a backup ({exc}).") from exc
    bad = [n for n in names if n not in ("holonomic.json", "backup.json") and not n.startswith("holonomic/")
           or ".." in n.split("/") or n.startswith("/") or ":" in n]
    if "holonomic/holonomic.db" not in names or bad:
        raise BackupError(f"{file.name} is not a backup made by this plugin. Nothing was changed.")
    return {"file": str(file), "files": len(names) - 1, "made_at": info.get("made_at", ""), "plugin_version": info.get("plugin_version", ""),
            "label": info.get("label", "")}


def pick(folder: Path, which: str = "") -> Path:
    """The backup meant by a name, a path, or nothing (the newest)."""
    if which:
        for candidate in (Path(os.path.expandvars(os.path.expanduser(which))), Path(folder) / which):
            if candidate.is_file():
                return candidate
        raise BackupError(f"There is no backup at {which}.")
    have = backups(folder)
    if not have:
        raise BackupError(f"There are no backups in {folder}.")
    return Path(have[0]["file"])


def restore(hermes_home: Path, file: Path) -> dict:
    """Put a backup in place of the present store, which is set aside, not deleted."""
    home = Path(hermes_home)
    info = inspect(file)
    data, config, stamp = home / "holonomic", home / "holonomic.json", _stamp()
    staging = home / f"holonomic.restoring_{stamp}"
    aside = home / f"holonomic.before-restore_{stamp}"
    home.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(file) as z:
            z.extractall(staging)
        if not (staging / "holonomic" / "holonomic.db").exists():
            raise BackupError("The backup did not unpack correctly. Nothing was changed.")
        check = sqlite3.connect(staging / "holonomic" / "holonomic.db")
        try:
            verdict = check.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            check.close()
        if verdict != "ok":
            raise BackupError(f"The memory store in that backup is damaged ({verdict}). Nothing was changed.")
        moved, kept_config = False, None
        try:
            if data.exists():
                os.rename(data, aside)                    # fails on Windows while Hermes has the store open
                moved = True
            os.rename(staging / "holonomic", data)
        except OSError as exc:
            if moved and not data.exists():
                os.rename(aside, data)                    # put it back as it was
            raise BackupError("The memory store is in use (Hermes is probably running). Close Hermes and try again. "
                              f"Nothing was changed. ({exc.__class__.__name__})") from exc
        if (staging / "holonomic.json").exists():
            if config.exists():
                kept_config = Path(str(config) + f".before-restore_{stamp}")
                shutil.copy2(config, kept_config)
            os.replace(staging / "holonomic.json", config)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dict(info, set_aside=str(aside) if moved else "", config_set_aside=str(kept_config) if kept_config else "", into=str(data))
