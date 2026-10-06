import json, sys, time, zipfile

import pytest

from test_provider import HAVE_HERMES, make
from conftest import track
from holonomic import HolonomicMemory, HashEmbedder
from holonomic import backup as bk


def store(tmp_path):
    home = tmp_path / "home"
    m = track(HolonomicMemory(home / "holonomic", HashEmbedder()))
    m.remember("My name is Kayla and I have two cats", kind="said_user", session="s1")
    (home / "holonomic" / "images").mkdir()
    (home / "holonomic" / "images" / "cat.jpg").write_bytes(b"\xff\xd8 not really a picture " * 50)
    (home / "holonomic.json").write_text(json.dumps({"embed_model": "nomic-embed-text"}))
    return home, m


def test_a_backup_is_taken_while_the_store_is_open_and_put_back(tmp_path):
    home, m = store(tmp_path)
    out = tmp_path / "backups"
    done = bk.backup(home, out, label="before update!", version="9.9.9")          # the store is open: this is the usual case
    assert done["file"].endswith("_before-update-.zip") and done["files"] >= 4 and done["removed"] == []
    with zipfile.ZipFile(done["file"]) as z:
        names = z.namelist()
    assert {"holonomic/holonomic.db", "holonomic/images/cat.jpg", "holonomic.json", "backup.json"} <= set(names)
    assert not [n for n in names if n.endswith(("-wal", "-shm", ".part", ".tmp"))] and not list(out.glob("*.part")) + list(out.glob(".copy-*"))
    info = bk.inspect(done["file"])
    assert info["plugin_version"] == "9.9.9" and info["label"] == "before update!" and info["files"] == done["files"]
    # things change, then the backup is put back
    m.remember("Something said after the backup was made", kind="said_user", session="s2")
    m.close()
    (home / "holonomic.json").write_text(json.dumps({"embed_model": "changed"}))
    back = bk.restore(home, bk.pick(out))
    again = track(HolonomicMemory(home / "holonomic", HashEmbedder()))
    assert [r["text"] for r in again.recent(10)] == ["My name is Kayla and I have two cats"]
    assert again.recall("Kayla cats", k=3, min_score=0.0)[0].text.startswith("My name is Kayla")
    assert (home / "holonomic" / "images" / "cat.jpg").read_bytes().startswith(b"\xff\xd8")
    assert json.loads((home / "holonomic.json").read_text())["embed_model"] == "nomic-embed-text"
    # what was there is set aside whole, not deleted
    aside = track(HolonomicMemory(back["set_aside"], HashEmbedder()))
    assert len(aside.recent(10)) == 2 and json.loads(open(back["config_set_aside"]).read())["embed_model"] == "changed"
    assert not list(home.glob("holonomic.restoring_*"))


def test_old_backups_are_removed_and_strange_files_refused(tmp_path):
    home, m = store(tmp_path)
    out = tmp_path / "backups"
    for i in range(4):
        (out / f"holonomic_2026-01-0{i + 1}_000000.zip").parent.mkdir(exist_ok=True)
        (out / f"holonomic_2026-01-0{i + 1}_000000.zip").write_bytes(b"old")
    done = bk.backup(home, out, keep=3)
    assert done["removed"] == ["holonomic_2026-01-02_000000.zip", "holonomic_2026-01-01_000000.zip"]
    assert len(bk.backups(out)) == 3 and bk.backups(out)[0]["file"] == done["file"] and bk.backups(tmp_path / "none") == []
    assert len(bk.backups(out)) == 3 and bk.backup(home, out, keep=0)["removed"] == []
    for bad in (out / "holonomic_2026-01-04_000000.zip", tmp_path / "missing.zip"):
        with pytest.raises(bk.BackupError):
            bk.inspect(bad)
    evil = out / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("holonomic/holonomic.db", "x"); z.writestr("../outside.txt", "x")
    other = out / "other.zip"
    with zipfile.ZipFile(other, "w") as z:
        z.writestr("readme.txt", "x")
    for bad in (evil, other):
        with pytest.raises(bk.BackupError):
            bk.restore(home, bad)
    assert not (tmp_path / "outside.txt").exists() and len(m.recent(10)) == 1                # the store was not touched
    with pytest.raises(bk.BackupError):
        bk.backup(tmp_path / "nohome", out)
    with pytest.raises(bk.BackupError):
        bk.backup(home, home / "holonomic" / "backups")
    with pytest.raises(bk.BackupError):
        bk.pick(tmp_path / "none")
    assert bk.pick(out, "evil.zip") == evil and bk.default_folder({"backup_dir": str(out)}) == out


def test_a_damaged_backup_changes_nothing(tmp_path):
    home, m = store(tmp_path)
    out = tmp_path / "backups"; out.mkdir()
    bad = out / "holonomic_2026-01-01_000000.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("holonomic/holonomic.db", "this is not a database at all, " * 200)
    with pytest.raises(Exception):
        bk.restore(home, bad)
    assert len(m.recent(10)) == 1 and not list(home.glob("holonomic.restoring_*")) and not list(home.glob("holonomic.before-restore_*"))


def test_libraries_are_backed_up_with_the_rest(tmp_path):
    from holonomic import library as lib
    home, m = store(tmp_path)
    src = tmp_path / "material"; src.mkdir()
    (src / "boot.md").write_text("# Boot sector\n\nThe boot sector ends with the signature 0xAA55.\n")
    root = lib.root_of(m)
    lib.create(root, "x86", str(src))
    lib.build(root, "x86", HashEmbedder(), {})
    done = bk.backup(home, tmp_path / "backups")                                  # with the library's store open as well
    lib.close_all(root); m.close()
    bk.restore(home, done["file"])
    again = track(HolonomicMemory(home / "holonomic", HashEmbedder()))
    root = lib.root_of(again)
    try:
        assert lib.names(root) == ["x86"]
        assert "0xAA55" in lib.search(root, ["x86"], "boot sector signature", HashEmbedder(), {}, floor=0.1)[0]["text"]
    finally:
        lib.close_all(root)


def test_cli_backup_and_restore(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, io, types
    p = make(tmp_path)
    p.sync_turn("My name is Kayla and I build memory systems for fun", "Nice to meet you, Kayla.", session_id="s1")
    home, out = tmp_path / "home", tmp_path / "backups"
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
    try:
        import holonomic.cli as cli
        parser = argparse.ArgumentParser(); cli.register_cli(parser)

        def run(*argv):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                args = parser.parse_args(list(argv)); args.func(args)
            return buf.getvalue()
        assert "There are no backups" in run("backup", "--list", "--to", str(out))
        text = run("backup", "--to", str(out), "--label", "first")                # while the provider has the store open
        assert "Backup written:" in text and "_first.zip" in text and "hermes holonomic restore" in text
        assert "_first.zip" in run("restore", "--list", "--from", str(out))
        p.sync_turn("And I have a cat called Theo", "Lovely.", session_id="s1")
        p.shutdown()
        text = run("restore", "--from", str(out))
        assert "This would put back" in text and "add --yes" in text and "set aside beside it, not deleted" in text
        text = run("restore", "--from", str(out), "--yes")
        assert "Restored from" in text and "What was there before is kept at" in text
        assert "There is no backup at nothing.zip" in run("restore", "nothing.zip", "--from", str(out), "--yes")
        q = make(tmp_path)
        try:
            assert q._engine.stats()["memories"] == 2
        finally:
            q.shutdown()
    finally:
        sys.modules.pop("hermes_constants", None)
