import json, sys, time, zipfile

import pytest

from test_provider import HAVE_HERMES, make
from holonomic import HashEmbedder
from holonomic import library as lib

BOOT = """# Boot sector

The boot sector is the first 512 bytes of the disk and ends with the signature 0xAA55.
The BIOS loads it at address 0x7C00 and jumps to it in real mode.

## BIOS parameter block

The BIOS parameter block starts at offset 11. Bytes per sector is a word at offset 11,
sectors per cluster a byte at offset 13, and the number of reserved sectors a word at offset 14.

```
mov ax, 0x07C0
mov ds, ax
```

# Reading sectors

INT 13h with AH=02h reads sectors using cylinder, head and sector numbers.
INT 13h with AH=42h is the extended read and takes a disk address packet in DS:SI.
"""
FAT = "FAT12 keeps cluster numbers in 12 bits, packed three bytes to two entries.\n\nThe root directory has a fixed number of 32-byte entries.\n"
CFG = {"dim": 512, "plate_capacity": 64.0}


def folder(tmp_path):
    src = tmp_path / "material"
    (src / "fs").mkdir(parents=True)
    (src / ".git").mkdir()
    (src / "boot.md").write_text(BOOT, encoding="utf-8")
    (src / "fs" / "fat12.txt").write_text(FAT, encoding="utf-8")
    (src / "loader.asm").write_text("; second stage\nstart:\n    mov si, msg\n    call print\n    jmp $\n", encoding="utf-8")
    (src / "image.bin").write_bytes(b"\x00\x01\x02" * 400)
    (src / "strange.dat").write_bytes(b"MZ\x00\x00" * 300)
    (src / "empty.txt").write_text("", encoding="utf-8")
    (src / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (src / "page.html").write_text("<html><head><title>x</title><script>var a=1;</script></head><body><h1>Video modes</h1>"
                                   "<p>Mode 13h is 320 by 200 pixels in 256 colours.</p></body></html>", encoding="utf-8")
    return src


def built(tmp_path, name="x86"):
    root = tmp_path / "store" / "libraries"
    lib.create(root, name, str(folder(tmp_path)), "Booting an x86 machine")
    report = lib.build(root, name, HashEmbedder(), CFG)
    return root, report


def test_a_file_is_cut_at_headings_and_paragraphs():
    cut = lib.pieces_of(BOOT, "boot.md", 300)
    heads = [h for h, _ in cut]
    assert heads[0] == "Boot sector" and "Boot sector > BIOS parameter block" in heads and heads[-1] == "Reading sectors"
    assert all(len(p) <= 300 for _, p in cut) and not any(p.startswith("#") for _, p in cut)
    code = next(p for h, p in cut if "mov ax, 0x07C0" in p)
    assert "```\nmov ax, 0x07C0\nmov ds, ax\n```" in code                       # a block of code is not cut in two
    # underlined headings, and a file with no headings at all
    assert [h for h, _ in lib.pieces_of("Title\n=====\n\nSome words here.\n\nPart\n----\n\nMore words.\n", "a.rst")] == ["Title", "Title > Part"]
    assert lib.pieces_of("; code\nstart:\n    jmp $\n", "a.asm") == [("start", "; code\nstart:\n    jmp $")]
    long = lib.pieces_of("\n".join(f"line {i} of a very long listing" for i in range(200)), "a.asm", 400)
    assert len(long) > 5 and all(len(p) <= 400 for _, p in long) and long[0][1].startswith("line 0 ")
    assert lib.pieces_of("   \n\n", "a.txt") == []


def test_names_and_folders_are_checked(tmp_path):
    root = tmp_path / "store" / "libraries"
    assert lib.clean_name(" x86 Assembly ") == "x86-assembly" and lib.clean_name("FAT_fs") == "fat_fs"
    for bad in ("", "   ", "../..", "x" * 60):
        with pytest.raises(lib.LibraryError):
            lib.clean_name(bad)
    with pytest.raises(lib.LibraryError):
        lib.create(root, "x86", str(tmp_path / "nowhere"))
    with pytest.raises(lib.LibraryError):
        lib.create(root, "x86", "")
    src = folder(tmp_path)
    lib.create(root, "x86", str(src))
    with pytest.raises(lib.LibraryError):
        lib.create(root, "X86", str(src))                                       # there already
    with pytest.raises(lib.LibraryError):
        lib.build(root, "other", HashEmbedder(), CFG)
    assert lib.names(root) == ["x86"] and lib.names(tmp_path / "none") == []
    lib.close_all(root)


def test_a_library_is_built_from_a_folder_and_looked_up(tmp_path):
    root, report = built(tmp_path)
    try:
        assert report["added"] == 4 and report["pieces"] >= 6 and not report["stopped"]
        left = {s["file"]: s["why"] for s in report["skipped"]}
        assert left == {"image.bin": "not a text file", "strange.dat": "not a text file", "empty.txt": "empty"}
        s = lib.summary(root, "x86")
        assert s["files"] == 4 and s["pieces"] == report["pieces"] and s["about"] == "Booting an x86 machine" and s["built"] != "not yet"
        found = lib.search(root, ["x86"], "INT 13h AH=42h extended read", HashEmbedder(), CFG, floor=0.1)
        assert found and found[0]["source"] == "boot.md > Reading sectors" and "disk address packet" in found[0]["text"]
        assert not found[0]["text"].startswith("boot.md")                       # the heading that leads a piece is not repeated
        found = lib.search(root, ["x86"], "how many bits is a FAT12 cluster number", HashEmbedder(), CFG, floor=0.1)
        assert found[0]["file"] == "fs/fat12.txt" and found[0]["library"] == "x86"
        found = lib.search(root, ["x86"], "mode 13h pixels colours", HashEmbedder(), CFG, floor=0.1)
        assert found[0]["source"] == "page.html > Video modes" and "var a=1" not in found[0]["text"]
        assert lib.search(root, ["x86"], "", HashEmbedder(), CFG) == [] and lib.search(root, [], "boot", HashEmbedder(), CFG) == []
        assert lib.search(root, ["gone"], "boot sector", HashEmbedder(), CFG) == []
    finally:
        lib.close_all(root)


def test_updating_takes_in_what_changed_and_only_that(tmp_path):
    root, first = built(tmp_path)
    try:
        src = tmp_path / "material"
        again = lib.build(root, "x86", HashEmbedder(), CFG)
        assert again["unchanged"] == 4 and again["added"] == again["changed"] == again["removed"] == again["pieces"] == 0
        (src / "fs" / "fat12.txt").write_text(FAT + "\nFAT16 keeps cluster numbers in 16 bits.\n", encoding="utf-8")
        (src / "loader.asm").unlink()
        (src / "notes.txt").write_text("We use NASM syntax, and stage two loads at 0x7E00.\n", encoding="utf-8")
        third = lib.build(root, "x86", HashEmbedder(), CFG)
        assert (third["added"], third["changed"], third["removed"], third["unchanged"]) == (1, 1, 1, 2)
        assert lib.summary(root, "x86")["files"] == 4
        assert lib.search(root, ["x86"], "FAT16 cluster numbers 16 bits", HashEmbedder(), CFG, floor=0.1)[0]["file"] == "fs/fat12.txt"
        assert lib.search(root, ["x86"], "NASM syntax stage two", HashEmbedder(), CFG, floor=0.1)[0]["file"] == "notes.txt"
        assert not [f for f in lib.search(root, ["x86"], "second stage mov si msg call print", HashEmbedder(), CFG, floor=0.1)
                    if f["file"] == "loader.asm"]                                  # what was taken out is gone
        store = lib.store(root, "x86", HashEmbedder(), CFG)
        assert store.stats()["memories"] == lib.summary(root, "x86")["pieces"]
        # a folder that has gone away is said, not guessed around
        info = lib.info(root, "x86"); info["folder"] = str(tmp_path / "moved")
        lib._save_info(root, "x86", info)
        with pytest.raises(lib.LibraryError):
            lib.build(root, "x86", HashEmbedder(), CFG)
    finally:
        lib.close_all(root)


def test_docx_is_read_and_a_folder_that_is_too_big_is_refused(tmp_path):
    src = tmp_path / "docs"; src.mkdir()
    with zipfile.ZipFile(src / "spec.docx", "w") as z:
        z.writestr("word/document.xml", "<w:document><w:body><w:p><w:r><w:t>The partition table holds four entries.</w:t></w:r></w:p>"
                                        "<w:p><w:r><w:t>Each entry is 16 bytes &amp; starts at offset 446.</w:t></w:r></w:p></w:body></w:document>")
    assert "four entries.\n\nEach entry is 16 bytes & starts" in lib.read_file(src / "spec.docx", 10**7)
    for i in range(6):
        (src / f"n{i}.txt").write_text(f"note number {i} about things\n", encoding="utf-8")
    root = tmp_path / "store" / "libraries"
    lib.create(root, "docs", str(src))
    with pytest.raises(lib.LibraryError):
        lib.build(root, "docs", HashEmbedder(), dict(CFG, library_max_files=5))
    assert lib.build(root, "docs", HashEmbedder(), CFG)["added"] == 7
    lib.close_all(root)
    assert lib.delete(root, "docs") and lib.names(root) == [] and (src / "spec.docx").exists()      # the material is untouched


def test_a_library_is_open_in_one_conversation_only(tmp_path):
    root, _ = built(tmp_path)
    try:
        assert lib.opened(root, "monday") == [] and lib.last_used(root, "monday") is None
        assert lib.context_block(root, "monday", "how does INT 13h read sectors", HashEmbedder(), CFG) == ("", [])
        with pytest.raises(lib.LibraryError):
            lib.open_for(root, "monday", ["nothing-here"])
        assert lib.open_for(root, "monday", ["X86"]) == ["x86"] and lib.open_for(root, "monday", ["x86"]) == ["x86"]
        block, notes = lib.context_block(root, "monday", "how does INT 13h read sectors with AH=42h", HashEmbedder(), dict(CFG, library_min_score=0.1))
        assert block.startswith("## Reference material (from the library open in this conversation: x86)")
        assert "[x86: boot.md > Reading sectors]" in block and "disk address packet" in block and "not something you remember" in block
        assert notes[0] == "reference libraries open in this conversation: x86" and notes[1].startswith("pieces given: x86: boot.md > Reading sectors")
        # the next conversation has nothing open, and is told what the last one used
        assert lib.opened(root, "tuesday") == []
        assert lib.context_block(root, "tuesday", "how does INT 13h read sectors", HashEmbedder(), CFG) == ("", [])
        assert lib.last_used(root, "tuesday")["names"] == ["x86"] and lib.last_used(root, "monday") is None
        assert lib.last_used(root, "tuesday", days=0) is None
        told = lib.prompt_block(root, "tuesday", CFG)
        assert "- x86: " in told and "Booting an x86 machine" in told and "None is open in this conversation" in told
        assert "had x86 open" in told and "ask whether to open it" in told and "works only when the user's own message asks" in told
        assert "- x86: " in told and " pieces from 4 files" in told
        assert "Open in this conversation: x86." in lib.prompt_block(root, "monday", CFG)
        # the same conversation under a new id keeps it; closing ends it
        lib.carry_over(root, "monday-2", "monday")
        assert lib.opened(root, "monday-2") == ["x86"] and lib.close_for(root, "monday-2") == [] and lib.opened(root, "monday") == ["x86"]
        assert lib.close_for(root, "monday", ["x86"]) == [] and lib.opened(root, "monday") == []
        # a budget is kept, and a message nothing matches says so
        lib.open_for(root, "wednesday", ["x86"])
        block, _ = lib.context_block(root, "wednesday", "boot sector BIOS parameter block offset signature reserved sectors",
                                     HashEmbedder(), dict(CFG, library_min_score=0.05, library_score_band=1.0, library_context_chars=400, library_k=6))
        assert len(block) < 400 + 420
        block, notes = lib.context_block(root, "wednesday", "what shall we have for dinner tonight", HashEmbedder(), dict(CFG, library_min_score=0.6))
        assert "Nothing in it matched this message" in block and notes[-1] == "nothing in them matched this message"
        assert "None has been made yet" in lib.prompt_block(tmp_path / "empty", "s", CFG)
    finally:
        lib.close_all(root)


def test_a_build_can_run_in_the_background(tmp_path):
    root = tmp_path / "store" / "libraries"
    lib.create(root, "x86", str(folder(tmp_path)))
    try:
        lib.build_in_background(root, "x86", HashEmbedder(), CFG)
        for _ in range(200):
            state = lib.build_state(root, "x86")
            if not state["running"]:
                break
            time.sleep(0.05)
        assert not state["running"] and state["error"] == "" and state["report"]["added"] == 4 and state["done"] == state["total"] == 7
        assert lib.summary(root, "x86")["building"]["running"] is False
    finally:
        lib.close_all(root)


def lib_tool(p, **args):
    return json.loads(p.handle_tool_call("holonomic_library", args))


def test_she_makes_opens_and_uses_a_library_through_her_tool(tmp_path):
    if not HAVE_HERMES: return
    src = folder(tmp_path)
    p = make(tmp_path, session_id="monday")
    try:
        assert [t["name"] for t in p.get_tool_schemas()] == ["holonomic_memory", "holonomic_library"]
        assert lib_tool(p, action="list") == {"libraries": [], "open_in_this_conversation": []}
        assert "None has been made yet" in p.system_prompt_block()
        assert "error" in lib_tool(p, action="create", name="x86") and "error" in lib_tool(p, action="create", name="x86", folder=str(tmp_path / "no"))
        made = lib_tool(p, action="create", name="x86", folder=str(src), about="Booting an x86 machine")
        assert made["library"] == "x86" and made["building"]
        for _ in range(200):
            status = lib_tool(p, action="status", name="x86")
            if not status["building"]["running"]:
                break
            time.sleep(0.05)
        assert status["files"] == 4 and status["last_build"]["added"] == 4 and status["left_out_count"] == 3 and not status["open_in_this_conversation"]
        # made, but not open: nothing reaches her, and she cannot look anything up
        p.sync_turn("I had a lovely walk by the lake this morning", "That sounds peaceful.", session_id="monday")
        assert "Reference material" not in p.prefetch("how does INT 13h read sectors with AH=42h", session_id="monday")
        assert "Say which library" in lib_tool(p, action="search", query="INT 13h")["error"]
        assert "error" in lib_tool(p, action="open", names=["nope"]) and "error" in lib_tool(p, action="open")
        # Asked a question the library answers, she may look it up once; she may not open it on her own.
        p.prefetch("how does INT 13h read sectors with AH=42h", session_id="monday")
        refused = lib_tool(p, action="open", names=["x86"])
        assert refused["opened"] is False and refused["not_opened"] == ["x86"] and refused["open_in_this_conversation"] == []
        assert "ask the user" in refused["what_you_can_do"]
        once = lib_tool(p, action="search", query="INT 13h AH=42h extended read", names=["x86"])
        assert once["results"][0]["source"] == "boot.md > Reading sectors" and "A single lookup: x86 is not open" in once["note"]
        assert lib_tool(p, action="list")["open_in_this_conversation"] == []
        assert "Reference material" not in p.prefetch("and how many sectors can it read at once", session_id="monday")
        assert refused["error"].startswith("NOT OPENED: x86.") and "Do not tell the user it is open" in refused["error"]
        # She asks, the user says yes: now it opens.  A yes to nothing in particular opens nothing.
        p.prefetch("yes please", session_id="monday")
        assert lib_tool(p, action="open", names=["x86"])["opened"] is True
        assert lib_tool(p, action="close")["open_in_this_conversation"] == []
        p.prefetch("yes please", session_id="monday")
        assert lib_tool(p, action="open", names=["x86"])["opened"] is False                  # that yes was already used
        p.prefetch("no, leave it closed", session_id="monday")
        assert lib_tool(p, action="open", names=["x86"])["opened"] is False
        # She asks before trying at all, and the user says yes.
        p.sync_turn("I'd like to get back to the bootloader", "Gladly. Would you like me to open the `x86` library for it?", session_id="monday")
        p.prefetch("Yes, please.", session_id="monday")
        assert lib_tool(p, action="open", names=["x86"])["opened"] is True
        assert lib_tool(p, action="close")["open_in_this_conversation"] == []
        p.sync_turn("thanks", "You're welcome. What shall we look at?", session_id="monday")
        p.prefetch("Yes, please.", session_id="monday")
        assert lib_tool(p, action="open", names=["x86"])["opened"] is False                  # that reply offered nothing
        # The user asks in their own words.
        p.prefetch("Let's work on the bootloader. Use the x86 library for this conversation.", session_id="monday")
        assert lib_tool(p, action="open", names=["x86"])["open_in_this_conversation"] == ["x86"]
        p._cfg["library_min_score"] = 0.1
        given = p.prefetch("how does INT 13h read sectors with AH=42h", session_id="monday")
        assert "## Reference material (from the library open in this conversation: x86)" in given and "disk address packet" in given
        found = lib_tool(p, action="search", query="FAT12 cluster number bits")
        assert found["searched"] == ["x86"] and found["results"][0]["source"] == "fs/fat12.txt" and "note" not in found
        assert "error" in lib_tool(p, action="search", query="x", names=["other"])
        context = (tmp_path / "home" / "holonomic" / "last_context.txt").read_text(encoding="utf-8")
        assert "reference libraries open in this conversation: x86" in context
        # reference material is not her memory: none of it was stored there
        assert not [r for r in p._engine.recent(50) if "disk address packet" in r["text"]]
        assert not [r for r in p._engine.recent(50) if r["kind"] == "reference"]
        # another conversation: nothing open, nothing given
        assert "Reference material" not in p.prefetch("how does INT 13h read sectors with AH=42h", session_id="tuesday")
        p.on_session_switch("tuesday", reset=True)
        assert "None is open in this conversation" in p.system_prompt_block() and "had x86 open" in p.system_prompt_block()
        assert "Say which library" in lib_tool(p, action="search", query="INT 13h")["error"]
        # the first conversation compressed and renamed keeps its library
        p.on_session_switch("monday")
        p.on_session_switch("monday-b", parent_session_id="monday")
        assert lib_tool(p, action="list")["open_in_this_conversation"] == ["x86"]
        assert lib_tool(p, action="close")["open_in_this_conversation"] == []
        assert lib_tool(p, action="update", name="x86")["building"]
        for _ in range(200):
            if not lib_tool(p, action="status", name="x86")["building"]["running"]:
                break
            time.sleep(0.05)
    finally:
        p.shutdown()


def test_cli_library_commands(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, io, types
    src = folder(tmp_path)
    p = make(tmp_path)
    p.sync_turn("My name is Kayla and I build memory systems for fun", "Nice to meet you, Kayla.", session_id="s1")
    p.shutdown()
    home = tmp_path / "home"
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
    try:
        import holonomic.cli as cli, holonomic.embed as embed
        real = embed.OllamaEmbedder
        embed.OllamaEmbedder = lambda *a, **k: HashEmbedder()
        parser = argparse.ArgumentParser(); cli.register_cli(parser)

        def run(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                args = parser.parse_args(["library", *argv]); args.func(args)
            return out.getvalue()
        assert "No libraries yet" in run()
        text = run("create", "x86", str(src), "--about", "Booting an x86 machine")
        assert "Made library 'x86'" in text and "4 file(s) added" in text and "left out: image.bin  (not a text file)" in text
        assert "There is already a library called 'x86'" in run("create", "x86", str(src))
        assert "x86: " in run("list") and "Booting an x86 machine" in run("list")
        assert "boot.md" in run("show", "x86") and "fs/fat12.txt" in run("show", "x86")
        assert "boot.md > Reading sectors" in run("search", "x86", "INT 13h AH=42h extended read")
        assert "4 unchanged" in run("update", "x86")
        assert "Add --yes" in run("delete", "x86") and "Removed library 'x86'" in run("delete", "x86", "--yes")
        assert "No libraries yet" in run("list") and "There is no library" in run("update", "x86") and (src / "boot.md").exists()
    finally:
        embed.OllamaEmbedder = real
        sys.modules.pop("hermes_constants", None)


def test_a_build_that_loses_its_embedding_server_keeps_what_it_had(tmp_path):
    root = tmp_path / "store" / "libraries"
    lib.create(root, "x86", str(folder(tmp_path)))

    class Flaky(HashEmbedder):
        calls = 0

        def embed(self, texts, kind="document"):
            Flaky.calls += 1
            if Flaky.calls > 4 and kind == "document" and "FAT12" in texts[0]:
                raise RuntimeError("connection refused")
            return super().embed(texts, kind)
    try:
        with pytest.raises(lib.LibraryError):
            lib.build(root, "x86", Flaky(), CFG)
        kept = lib.summary(root, "x86")
        assert kept["files"] >= 1 and kept["built"] == "not yet"
        lib.close_all(root)
        report = lib.build(root, "x86", HashEmbedder(), CFG)                    # carrying on reads only what is missing
        assert report["unchanged"] == kept["files"] and report["added"] == 4 - kept["files"] and lib.summary(root, "x86")["files"] == 4
    finally:
        lib.close_all(root)


def test_only_the_users_own_words_open_a_library(tmp_path):
    root = tmp_path / "libraries"
    asked = lambda message, name="x86", session="s": lib.user_asked(root, session, message, name)
    assert asked("Use the x86 library for this conversation") and asked("can you open the X86 data store?")
    assert asked("Please use the fat-filesystems and x86 libraries", "fat-filesystems") and asked("use the fat filesystems library", "fat-filesystems")
    assert asked("load all of my libraries") and asked("I want the reference material from x86 for this")
    # a question the library would answer is not a request to open it; nor is naming it
    assert not asked("What address does the x86 boot sector load its second stage at?")
    assert not asked("I was writing x86 assembly last night") and not asked("which libraries do you have?")
    assert not asked("don't open the x86 library yet") and not asked("") and not asked("yes")
    assert not asked("use the marigold library")                                             # another library
    lib.note_refused(root, "s", ["x86"])
    assert asked("yes") and asked("Sure, go ahead") and asked("yes please") and asked("ok")
    assert not asked("no") and not asked("not now") and not asked("yes", "marigold") and not asked("yes", session="other")
    assert not asked("yes, " + "and another thing entirely " * 5)                            # a long message that happens to open with yes
    lib.clear_refused(root, "s")
    assert not asked("yes")
    # She asks before trying, as she should, and the user says yes: that yes is an answer to her question.
    offer = "I'd love to dive back in. Would you like me to open the `x86` library? That'll give me access to the code."
    assert lib.user_asked(root, "s", "Yes, please.", "x86", offer) and lib.user_asked(root, "s", "sure", "x86", offer)
    assert not lib.user_asked(root, "s", "Yes, please.", "marigold", offer)                  # she asked about another one
    assert not lib.user_asked(root, "s", "no thanks", "x86", offer) and not lib.user_asked(root, "s", "What does x86 mean?", "x86", offer)
    assert not lib.user_asked(root, "s", "yes", "x86", "x86 assembly is fun, isn't it? Shall we carry on?")      # no offer to open in that


def test_old_programs_in_a_source_tree_are_not_read_as_text(tmp_path):
    """A folder of assembly source had its DOS executables beside it.  A small .COM program need not contain a
    single zero byte."""
    import random
    rnd = random.Random(7)
    com = bytes(rnd.choice([0xB4, 0x09, 0xBA, 0x0D, 0x01, 0xCD, 0x21, 0xC3, 0xEB, 0xFE, 0x8A, 0x04, 0x3C, 0x24, 0x74, 0xE8, 0x90, 0x1F, 0x07]) for _ in range(52))
    assert b"\x00" not in com and lib.looks_binary(com)
    assert lib.looks_binary(b"MZ" + b"This program cannot be run in DOS mode.") and lib.looks_binary(b"\x7fELF\x02\x01\x01")
    assert lib.looks_binary(bytes(range(1, 256)) * 4)
    assert not lib.looks_binary(b"; boot sector\r\nstart:\r\n\tmov ax, 0x07C0\r\n\tmov ds, ax\r\n\x1a")      # DOS text, with its end-of-file mark
    assert not lib.looks_binary("Größe: 512 Bytes, naïve café".encode("utf-8")) and not lib.looks_binary("日本語のメモ".encode("utf-8"))
    assert not lib.looks_binary("Gr\xf6\xdfe: 512 Bytes \xc9\xcd\xcd\xbb box".encode("latin-1") + b" plain words follow here" * 4)
    assert not lib.looks_binary(b"")
    # Old DOS text is not a program: a boxed notice, a BASIC program that draws with the box characters, a short
    # file with a symbol or two in it.  A stricter check threw 39 of a project's own files out.
    box = ("\u2554" + "\u2550" * 40 + "\u2557\r\n\u2551  VERSA  Copyright (C) Kayla             \u2551\r\n\u255a" + "\u2550" * 40 + "\u255d\r\n").encode("cp437")
    assert not lib.looks_binary(box) and "\u2554\u2550\u2550" in lib.decode(box) and "Copyright" in lib.decode(box)
    bas = ("\r\n".join(f'{i * 10} PRINT "' + "\u2588\u2592\u2591\u2500\u2502" * 6 + '"' for i in range(40))).encode("cp437")
    assert not lib.looks_binary(bas)
    assert not lib.looks_binary(b"Press \x10 to go on, \x03 to stop.\r\nThat is all.\r\n\x1a")
    assert lib.decode("na\xefve caf\xe9".encode("latin-1")) == "na\u00efve caf\u00e9" and lib.decode("\u65e5\u672c".encode("utf-8")) == "\u65e5\u672c"
    assert lib.looks_binary(bytes(rnd.randrange(1, 256) for _ in range(3000)))                # no zero byte, and no sense in it either
    assert lib.looks_binary(bytes(rnd.choice(range(128, 256)) for _ in range(900)))           # one long run of bytes with no lines
    src = tmp_path / "os"; src.mkdir()
    (src / "COPYRIGH.T").write_bytes(box)
    (src / "SPECS.DOC").write_text("V2OS memory proposal\r\n\r\nThe kernel keeps a bitmap of free pages.\r\n")       # a DOS .DOC is a text file
    (src / "BOOT.SCR").write_text("load system16\r\nrun shell\r\n")                                             # and this .SCR a script
    (src / "REPORT.DOC").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(600))                             # a Word document is not
    (src / "PADDED.TXT").write_bytes(b"Notes on the loader.\r\nIt reads the kernel from the disk.\r\n\x1a" + bytes(90) + b"\xff\x00junk")
    (src / "BLOB.DAT").write_bytes(b"\x1a" + bytes(range(1, 255)) * 3)
    (src / "boot.asm").write_text("; boot sector\nstart:\n    mov si, msg\n    call print\n    jmp $\n")
    (src / "CHOICE.COM").write_bytes(com)
    (src / "SHELL.EXE").write_bytes(b"MZ" + bytes(200))
    (src / "CONFIG.SYS").write_text("FILES=30\r\nBUFFERS=20\r\n")
    root = tmp_path / "store" / "libraries"
    lib.create(root, "my-os", str(src))
    try:
        report = lib.build(root, "my-os", HashEmbedder(), CFG)
        assert report["added"] == 6 and {s["file"]: s["why"] for s in report["skipped"]} == {
            "BLOB.DAT": "not a text file", "CHOICE.COM": "not a text file", "REPORT.DOC": "not a text file", "SHELL.EXE": "not a text file"}
        assert sorted(lib.info(root, "my-os")["files"]) == ["BOOT.SCR", "CONFIG.SYS", "COPYRIGH.T", "PADDED.TXT", "SPECS.DOC", "boot.asm"]
        assert "junk" not in lib.read_file(src / "PADDED.TXT", 10**7) and lib.read_file(src / "PADDED.TXT", 10**7).endswith("disk.\r\n")
        assert [g["file"] for g in lib.info(root, "my-os")["left_out"]] == ["BLOB.DAT", "CHOICE.COM", "REPORT.DOC", "SHELL.EXE"]       # kept for looking at later
        assert "\u2551  VERSA  Copyright" in lib.search(root, ["my-os"], "VERSA copyright Kayla", HashEmbedder(), CFG, floor=0.05)[0]["text"]
    finally:
        lib.close_all(root)


ASM = """;- VXOS boot sector
[BITS 16]
        Jmp   Start
OEM_ID          db "VERSA   "

Start:
        Cli
        Mov   AX, 0x07C0
""" + "\n".join(f"        Mov   AX, {i}        ;step {i} of setting up the stack and segments" for i in range(12)) + """

Load_FAT:
        ;Save starting cluster of boot image
        Mov   DX, WORD [DI + 0x001A]
""" + "\n".join(f"        Add   CX, {i}        ;compute size of FAT, part {i}" for i in range(12)) + """
.Loop:
        Call  ReadSectors
        Jmp   .Loop

;----PROCEDURE DisplayMessage--------------------------------------------------
;- Prints the ASCIZ string at DS:SI using the BIOS teletype service          -
;------------------------------------------------------------------------------
DisplayMessage:
        Lodsb
        Or    AL, AL
        Jz    .Done
        Mov   AH, 0x0E
        Int   0x10
.Done:
        Ret
""" + "\n".join(f"        Nop                  ;padding line {i} so the routine is long enough" for i in range(8)) + """

;----PROCEDURE ReadSectors-----------------------------------------------------
;- Reads CX sectors from disk starting at AX into memory location ES:BX      -
ReadSectors:
        Mov   DI, 0x0005          ;Five retries for error
        Int   0x13
        Ret
msgA: db 1
msgB: db 2
msgC: db 3
"""


def test_source_is_cut_at_its_routines_and_named_for_them(tmp_path):
    """A search of real assembly found the right file and could not say which routine: a piece was only
    'somewhere in BOOT.ASM'."""
    cut = lib.pieces_of(ASM, "BOOT.ASM", 1100)
    names = [h for h, _ in cut]
    assert names == ["Start", "Load_FAT", "DisplayMessage", "ReadSectors"]
    by = dict(cut)
    assert by["DisplayMessage"].startswith(";----PROCEDURE DisplayMessage") and "Prints the ASCIZ string" in by["DisplayMessage"]     # its comment goes with it
    assert ".Loop:" in by["Load_FAT"] and "Call  ReadSectors" in by["Load_FAT"]                # a local label does not start a routine
    assert "msgA: db 1" in by["ReadSectors"] and "msgC: db 3" in by["ReadSectors"]             # one-line labels do not each become a piece
    assert "PROCEDURE ReadSectors" not in by["DisplayMessage"]
    pas = "program x;\n" + "var a: integer;\n" * 30 + "procedure SetBoot(drive: byte);\nbegin\n" + "  writeln(1);\n" * 30 + "end;\n\nfunction GetBoot: byte;\nbegin\n" + "  GetBoot := 1;\n" * 30 + "end;\n"
    assert [h for h, _ in lib.pieces_of(pas, "SETBOOTF.PAS")] == ["", "SetBoot", "GetBoot"]
    c = "#include <stdio.h>\n" + "int x;\n" * 60 + "\nint main(int argc, char **argv)\n{\n" + "    if (x) {\n        x++;\n    }\n" * 12 + "}\n\nstatic void load_boot(void)\n{\n" + "    x--;\n" * 60 + "}\n"
    assert [h for h, _ in lib.pieces_of(c, "BOOTMGR.C")] == ["", "main", "load_boot"]
    assert lib.pieces_of("; just a comment\n", "X.INC") == [("", "; just a comment")]
    # What follows a routine is not always part of it: a stretch of commented-out code, a Pascal program's own body.
    dead = ASM.replace(";----PROCEDURE ReadSectors", ";Read root directory into memory (7C00:0200)\n"
                       + "".join(f";        Mov   BX, 0x020{i}          ;old way {i}\n;        Call  ReadSectors\n" for i in range(8))
                       + "\n;----PROCEDURE ReadSectors")
    cut = lib.pieces_of(dead, "VERSALDR.ASM", 1100)
    assert [h for h, _ in cut] == ["Start", "Load_FAT", "DisplayMessage", "commented-out code", "ReadSectors"]
    by = dict(cut)
    assert by["commented-out code"].startswith(";Read root directory into memory") and "old way 7" in by["commented-out code"]
    assert "old way" not in by["DisplayMessage"] and by["ReadSectors"].startswith(";----PROCEDURE ReadSectors")
    assert "commented-out code" not in [h for h, _ in lib.pieces_of(ASM, "BOOT.ASM")]             # an ordinary comment is not dead code
    prog = "program boothdd;\n" + "var a: integer;\n" * 30 + "function upstr(s: string): string;\nbegin\n" + "  upstr := s;\n" * 30 + "end;\n\nbegin\n" + "  writeln('Reading original boot sector...');\n" * 12 + "end.\n"
    cut = lib.pieces_of(prog, "BOOTHDD.PAS")
    assert [h for h, _ in cut] == ["", "upstr", "main program"] and "Reading original boot sector" in dict(cut)["main program"]
    assert "Reading original" not in dict(cut)["upstr"]
    # in a library: found by the routine's name, and cited by it
    src = tmp_path / "os"; (src / "OLD").mkdir(parents=True)
    (src / "BOOT.ASM").write_text(ASM)
    (src / "OLD" / "BOOT.BAK").write_text(ASM.replace("Five retries", "5 retries"))
    (src / "OLD" / "BOOT.ASM").write_text(ASM)
    root = tmp_path / "store" / "libraries"
    lib.create(root, "versa", str(src))
    try:
        lib.build(root, "versa", HashEmbedder(), CFG)
        found = lib.search(root, ["versa"], "ReadSectors reads sectors from disk into memory", HashEmbedder(), CFG, floor=0.1, k=4)
        assert found[0]["section"] == "ReadSectors" and found[0]["source"].endswith("BOOT.ASM > ReadSectors")
        # the copy kept in another folder is the same passage: given once, with where else it is
        assert len([f for f in found if f["section"] == "ReadSectors"]) == 1 and len(found[0]["also_in"]) >= 1
        assert len({f["section"] for f in found}) == len(found)                                 # so there is room for other routines
        lib.open_for(root, "s", ["versa"])
        block, _ = lib.context_block(root, "s", "ReadSectors reads sectors from disk into memory", HashEmbedder(), dict(CFG, library_min_score=0.1))
        assert "(the same passage is also in: " in block
        # reading everything again, for when the way files are cut has changed
        before = lib.summary(root, "versa")["pieces"]
        again = lib.build(root, "versa", HashEmbedder(), CFG, fresh=True)
        assert again["added"] == 3 and again["unchanged"] == 0 and lib.summary(root, "versa")["pieces"] == before
        assert [h for h, _ in lib.pieces_of(ASM, "BOOT.BAK")] == ["Start", "Load_FAT", "DisplayMessage", "ReadSectors"]   # an old copy is still assembly
        assert [h for h, _ in lib.pieces_of("Notes on the boot sector.\n\nIt loads the kernel.\n", "NOTES.BAK")] == [""]
        assert not lib._reads_as_assembly("We call it a day and then test the loop again.\n" * 9)
        assert lib.store(root, "versa", HashEmbedder(), CFG).stats()["memories"] == before
    finally:
        lib.close_all(root)


# ----------------------------------------------------------------- her notebook

def notebook_of(tmp_path):
    from holonomic import notebook as nb
    root, _ = built(tmp_path)
    store = lib.store(root, "x86", HashEmbedder(), CFG)
    return nb, root, store, lib.info(root, "x86")["files"]


def test_a_note_is_bound_to_its_passages_and_comes_back_with_them(tmp_path):
    """A library is a shelf; the notebook is what she has learned by using it.  A note is bound on plates of its own
    to the passages it is about, so each brings the other up though they are nothing alike."""
    nb, root, store, files = notebook_of(tmp_path)
    reading = next(i for i in files["boot.md"]["ids"] if store.get(i)["meta"]["section"] == "Reading sectors")
    before = {i: store.get(i)["text"] for f in files.values() for i in f["ids"]}
    note = nb.add(store, files, CFG, text="This example failed on our target because the drive number was not passed in DL; set it first.",
                  kind="lesson", about=["boot.md > Reading sectors"])
    guide = nb.add(store, files, CFG, text="For disk geometry problems read the BIOS parameter block and the FAT12 notes together.",
                   kind="guidance", about=["BOOT.MD > bios parameter block", "fat12.txt"])
    assert [a["id"] for a in note["about"]] == [reading] and note["provenance"] == "inference" and note["version"] == 1
    assert [a["file"] for a in guide["about"]] == ["boot.md", "fs/fat12.txt"]
    # the reference material is untouched, and is still all a search of the library gives
    assert {i: store.get(i)["text"] for f in files.values() for i in f["ids"]} == before
    assert lib.summary(root, "x86")["pieces"] == len(before)
    assert all(h.kind == lib.PIECE for h in store.recall("drive number DL", k=10, min_score=0.0))
    # the plates carry it both ways
    c = nb.check(store)
    assert c == {"notes": 2, "bindings": 3, "passage_to_note": 3, "note_to_passage": 3, "unbound": 0, "stale": 0}
    assert [n["id"] for n in nb.for_passages(store, [reading], CFG)[reading]] == [note["id"]]
    assert nb.for_passages(store, [files["loader.asm"]["ids"][0]], CFG) == {}
    # a passage that comes up brings its note, whatever the note's own words
    found = lib.search(root, ["x86"], "INT 13h AH=02h reads sectors cylinder head", HashEmbedder(), CFG, floor=0.0)
    hit = next(f for f in found if f["id"] == reading)
    assert [n["id"] for n in hit["notes"]] == [note["id"]] and all(not f["notes"] for f in found if f["id"] not in (reading, *[a["id"] for a in guide["about"]]))
    # and a note that answers the message itself comes up with where it points
    own = lib.note_matches(root, ["x86"], "disk geometry problems", HashEmbedder(), CFG, floor=0.0, k=1)
    assert [n["id"] for n in own] == [guide["id"]]
    line = nb.line(own[0], sources=True)
    assert "where to look, your own inference, not checked" in line and "boot.md > Boot sector > BIOS parameter block; fs/fat12.txt" in line
    # what cannot be found says what there is
    for bad, why in ((["nothing.asm"], "No file called"), (["boot.md > Nonsense"], "has no section"), (["999"], "no passage numbered")):
        with pytest.raises(lib.LibraryError, match=why):
            nb.add(store, files, CFG, text="A note about something that is not in the library at all.", about=bad)
    with pytest.raises(lib.LibraryError, match="already says this"):
        nb.add(store, files, CFG, text="This example failed on our target because the drive number was not passed in DL, set it first!", kind="lesson")
    assert len(nb.list_notes(store)) == 2


def test_a_guess_is_never_worded_like_a_result(tmp_path):
    """Where a note comes from is carried into recall, and decided here: a model asked for provenance calls its own
    guess confirmed."""
    nb, root, store, files = notebook_of(tmp_path)
    said = ["Yes, the loader has to set DL before the call, I checked the Bochs log myself."]
    text = "The loader must set DL to the boot drive before calling INT 13h."
    with pytest.raises(lib.LibraryError, match="NOT SAVED as tested"):
        nb.add(store, files, CFG, text=text, provenance="tested", evidence="it works")
    with pytest.raises(lib.LibraryError, match="NOT SAVED as the user's"):
        nb.add(store, files, CFG, text=text, provenance="user", quote="The user confirmed that DL must be set", messages=said)
    with pytest.raises(lib.LibraryError, match="NOT SAVED as the user's"):
        nb.add(store, files, CFG, text=text, provenance="user", quote="", messages=said)
    assert nb.list_notes(store) == []
    guess = nb.add(store, files, CFG, text=text, about=["loader.asm"])
    assert nb.where_from(guess) == "your own inference, not checked"
    tested = nb.confirm(store, guess["id"], provenance="tested", evidence="Assembled with NASM and booted in Bochs: the sector read succeeded.")
    assert tested["id"] == guess["id"] and nb.where_from(tested) == "you tested this: Assembled with NASM and booted in Bochs: the sector read succeeded."
    hers = nb.confirm(store, guess["id"], provenance="user", quote="the loader has to set DL before the call", messages=said)
    assert nb.where_from(hers) == 'the user told you: "the loader has to set DL before the call"' and hers["evidence"] == ""
    with pytest.raises(lib.LibraryError):
        nb.confirm(store, guess["id"], provenance="inference")
    # a better version takes the note's place; changed words are her inference again
    better = nb.revise(store, files, CFG, guess["id"], text="The loader must set DL to the boot drive, which the BIOS leaves in DL at entry, before INT 13h.")
    assert better["revises"] == guess["id"] and better["version"] == 2 and better["provenance"] == "inference"
    assert [a["file"] for a in better["about"]] == ["loader.asm"]                         # it is still about what it was about
    old = nb.get_note(store, guess["id"])
    assert old["retired"] and old["replaced_by"] == better["id"] and [n["id"] for n in nb.list_notes(store)] == [better["id"]]
    assert [n["id"] for n in nb.list_notes(store, retired=True)] == [guess["id"], better["id"]]
    passage = files["loader.asm"]["ids"][0]
    assert [n["id"] for n in nb.for_passages(store, [passage], CFG)[passage]] == [better["id"]]      # the retired one does not come back
    assert nb.matching(store, text, CFG, floor=0.0)[0]["id"] == better["id"]
    # the same words with more passages keep their standing
    again = nb.revise(store, files, CFG, better["id"], about=["loader.asm", "boot.md > Reading sectors"], provenance="tested",
                      evidence="Booted in Bochs with DL preserved from the BIOS; the read returned carry clear.")
    wider = nb.revise(store, files, CFG, again["id"], kind="connection")
    assert wider["provenance"] == "tested" and wider["type"] == "connection" and len(wider["about"]) == 2
    gone = nb.retire(store, wider["id"], "the loader was rewritten")
    assert gone["retired"] and gone["retired_because"] == "the loader was rewritten" and nb.list_notes(store) == []
    assert not nb.restore(store, wider["id"])["retired"] and len(nb.list_notes(store)) == 1
    # the user's ranks above a test, a test above a guess
    a = nb.add(store, files, CFG, text="A guess about how the print routine walks the string.", about=["loader.asm"])
    b = nb.add(store, files, CFG, text="The print routine stops at a zero byte, as the user explained.", about=["loader.asm"], provenance="user", by_command=True)
    order = [n["id"] for n in nb.for_passages(store, [passage], dict(CFG, library_notes_per_passage=3))[passage]]
    assert order == [b["id"], wider["id"], a["id"]] and nb.where_from(b) == "from the user"


def test_notes_follow_their_passages_when_the_folder_is_read_again(tmp_path):
    nb, root, store, files = notebook_of(tmp_path)
    src = tmp_path / "material"
    kept = nb.add(store, files, CFG, text="FAT12 entries are packed, so read two bytes and mask or shift by the cluster's parity.", about=["fat12.txt"])
    edited = nb.add(store, files, CFG, text="The extended read needs a disk address packet; our loader does not build one yet.",
                    kind="connection", about=["boot.md > Reading sectors"])
    lost = nb.add(store, files, CFG, text="Mode 13h is the mode the splash screen should use.", kind="connection", about=["page.html"])
    old = {n["id"]: [a["id"] for a in n["about"]] for n in (kept, edited, lost)}
    # adding a file changes nothing for the notes
    (src / "notes.md").write_text("# A20\n\nThe A20 line must be enabled before using memory above one megabyte.\n", encoding="utf-8")
    time.sleep(0.01)
    report = lib.build(root, "x86", HashEmbedder(), CFG)
    assert report["added"] == 1 and report["notes"] == {"notes": 3, "rebound": 0, "changed": 0, "gone": 0}
    assert {n["id"]: [a["id"] for a in n["about"]] for n in nb.list_notes(store)} == old
    # a file edited, a file removed
    (src / "boot.md").write_text(BOOT.replace("takes a disk address packet in DS:SI", "takes a 16-byte disk address packet in DS:SI"), encoding="utf-8")
    (src / "page.html").unlink()
    time.sleep(0.01)
    report = lib.build(root, "x86", HashEmbedder(), CFG)
    assert report["changed"] == 1 and report["removed"] == 1 and report["notes"] == {"notes": 3, "rebound": 0, "changed": 1, "gone": 1}
    files = lib.info(root, "x86")["files"]
    now = {n["id"]: n for n in nb.list_notes(store)}
    assert [a["id"] for a in now[kept["id"]]["about"]] == old[kept["id"]] and not now[kept["id"]]["about"][0]["state"]
    moved = now[edited["id"]]["about"][0]
    assert moved["state"] == "changed" and moved["id"] != old[edited["id"]][0] and moved["id"] in files["boot.md"]["ids"]
    assert "16-byte" in store.get(moved["id"])["text"] and "the passage it was written about has since changed" in nb.label(now[edited["id"]])
    assert [n["id"] for n in nb.for_passages(store, [moved["id"]], CFG)[moved["id"]]] == [edited["id"]]       # bound to the new passage
    orphan = now[lost["id"]]
    assert orphan["about"][0]["state"] == "gone" and orphan["about"][0]["id"] is None and "no longer in the library" in nb.label(orphan)
    assert nb.matching(store, "splash screen mode 13h", CFG, floor=0.0)[0]["id"] == lost["id"]              # kept, and found by what it says
    assert nb.check(store)["unbound"] == 1 and nb.check(store)["stale"] == 2
    # everything read again: every passage is new, and every note that can be is bound again without a word
    report = lib.build(root, "x86", HashEmbedder(), CFG, fresh=True)
    assert report["notes"] == {"notes": 3, "rebound": 2, "changed": 0, "gone": 0}
    files = lib.info(root, "x86")["files"]
    now = {n["id"]: n for n in nb.list_notes(store)}
    assert now[kept["id"]]["about"][0]["id"] == files["fs/fat12.txt"]["ids"][0] != old[kept["id"]][0]
    c = nb.check(store)
    assert c["bindings"] == 2 and c["passage_to_note"] == 2 and c["note_to_passage"] == 2
    # the page comes back: its note finds it again
    (src / "page.html").write_text("<html><body><h1>Video modes</h1><p>Mode 13h is 320 by 200 pixels in 256 colours.</p></body></html>", encoding="utf-8")
    assert lib.build(root, "x86", HashEmbedder(), CFG)["notes"]["rebound"] == 1
    assert nb.get_note(store, lost["id"])["about"][0]["state"] == ""


def test_she_keeps_a_notebook_through_her_tool(tmp_path):
    if not HAVE_HERMES: return
    from holonomic import notebook as nb
    src = folder(tmp_path)
    p = make(tmp_path)
    root = lib.root_of(p._engine)
    lib.create(root, "x86", str(src), "Booting an x86 machine")
    lib.build(root, "x86", p._engine.embedder, p._cfg)
    block = p.system_prompt_block()
    assert "Each library has a notebook that is yours" in block and "never call a guess a result" in block
    # closed: she may read notes, not write them
    p.prefetch("How does the boot sector end?", session_id="s1")
    refused = lib_tool(p, action="note", name="x86", text="The boot signature is the last two bytes of the sector.", passages=["boot.md > Boot sector"])
    assert "NOT SAVED" in refused["error"] and "not open" in refused["error"]
    assert lib_tool(p, action="notes", name="x86")["count"] == 0
    p.prefetch("Please open the x86 library for this.", session_id="s1")
    assert lib_tool(p, action="open", names=["x86"])["opened"] is True
    found = lib_tool(p, action="search", query="INT 13h AH=02h reads sectors")
    passage = next(r for r in found["results"] if r["source"] == "boot.md > Reading sectors")
    mine = p._engine.remember("We are writing a loader that reads the kernel with INT 13h", kind="said_user", session="s1")[0]
    saved = lib_tool(p, action="note", text="Our loader reads the kernel with the AH=02h call described here, so it needs CHS numbers.",
                     type="connection", passages=[str(passage["passage"])], memories=[mine, 99999])
    assert saved["saved"] and saved["about"] == ["boot.md > Reading sectors"] and saved["memories"] == [mine]
    assert saved["where_it_comes_from"] == "your own inference, not checked"
    loose = lib_tool(p, action="note", text="Remember to check which BIOS calls the target machine really supports.", type="question")
    assert loose["saved"] and "not about any passage" in loose["warning"]
    # she cannot make a guess the user's word, or a result
    p.prefetch("Right, and CHS is fine for a floppy, we do not need the extended read.", session_id="s1")
    claim = dict(action="confirm_note", note_id=saved["note_id"], provenance="user")
    assert "NOT SAVED as the user's" in lib_tool(p, quote="The user agreed that the loader is correct", **claim)["error"]
    assert "NOT SAVED as tested" in lib_tool(p, action="confirm_note", note_id=saved["note_id"], provenance="tested", evidence="works")["error"]
    ok = lib_tool(p, quote="CHS is fine for a floppy", **claim)
    assert ok["where_it_comes_from"] == 'the user told you: "CHS is fine for a floppy"'
    # the note comes back with its passage, in the message's context and in a lookup, labelled as hers
    context = p.prefetch("What does INT 13h with AH=02h need to read sectors?", session_id="s1")
    assert "(passage %d)" % passage["passage"] in context and "Your notes on this passage (yours, not the reference material):" in context
    assert f"(note {saved['note_id']}, how it bears on the project, the user told you: \"CHS is fine for a floppy\")" in context
    assert "It came out of what you remember: #%d" % mine in context
    again = lib_tool(p, action="search", query="INT 13h AH=02h reads sectors")
    with_note = next(r for r in again["results"] if r["source"] == "boot.md > Reading sectors")
    assert "Our loader reads the kernel" in with_note["your_notes_on_this_passage"][0]
    # a note that answers the message itself
    context = p.prefetch("Which BIOS calls does the target machine really support?", session_id="s1")
    assert "### Your notes that bear on this message" in context and f"(note {loose['note_id']}, open question, your own inference, not checked)" in context
    # revised, not doubled
    dup = lib_tool(p, action="note", text="Our loader reads the kernel with the AH=02h call described here so it needs CHS numbers", type="connection")
    assert "already says this" in dup["error"] and f"note_id {saved['note_id']}" in dup["error"]
    new = lib_tool(p, action="revise_note", note_id=saved["note_id"], text="Our loader reads the kernel with AH=02h, so it needs CHS numbers worked out from the LBA.")
    assert new["replaced_note"] == saved["note_id"] and new["where_it_comes_from"] == "your own inference, not checked"
    listed = lib_tool(p, action="notes")
    assert [n["note_id"] for n in listed["notes"]] == [loose["note_id"], new["note_id"]]
    assert lib_tool(p, action="retire_note", note_id=loose["note_id"], reason="answered")["retired"] is True
    # none of it is in her own memory, and another conversation sees none of it
    assert not [h for h in p._engine.recall("CHS numbers worked out from the LBA", k=10, min_score=0.0) if "CHS numbers worked out" in h.text]
    assert "Your notes" not in (p.prefetch("What does INT 13h with AH=02h need to read sectors?", session_id="s2") or "")
    assert len(nb.list_notes(lib.store(root, "x86", p._engine.embedder, p._cfg))) == 1
    p.shutdown()


def test_cli_library_notes(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, io, types
    src = folder(tmp_path)
    p = make(tmp_path)
    home = tmp_path / "home"
    p.shutdown()
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
    try:
        import holonomic.cli as cli, holonomic.embed as embed
        keep = embed.OllamaEmbedder
        embed.OllamaEmbedder = lambda *x, **k: HashEmbedder()
        parser = argparse.ArgumentParser(); cli.register_cli(parser)

        def run(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                args = parser.parse_args(list(argv)); args.func(args)
            return out.getvalue()
        run("library", "create", "x86", str(src))
        assert "0 note(s) in the notebook of 'x86'." in run("library", "notes", "x86")
        text = run("library", "notes", "x86", "add", "INT 13h AH=02h wants the drive number in DL.", "--on", "boot.md > Reading sectors")
        assert "lesson learned; from the user" in text and "about: boot.md > Reading sectors" in text and "Saved, as yours." in text
        assert "No file called" in run("library", "notes", "x86", "add", "A note about a file that is not there at all.", "--on", "zzz.asm")
        listing = run("library", "notes", "x86")
        note_id = int(listing.split("[")[1].split("]")[0])
        assert "1 note(s)" in listing
        assert "the note from its passage 1 of 1; the passage from its note 1 of 1" in run("library", "notes", "x86", "check")
        assert f"her note {note_id} (lesson learned; from the user)" in run("library", "search", "x86", "INT 13h AH=02h reads sectors")
        fixed = run("library", "notes", "x86", "correct", str(note_id), "INT 13h AH=02h wants the drive in DL and the count in AL.", "--type", "lesson")
        assert f"Note {note_id} is retired and this takes its place" in fixed and "version 2" in fixed
        new_id = int(fixed.split("[")[1].split("]")[0])
        assert "Marked as confirmed by you." in run("library", "notes", "x86", "confirm", str(new_id))
        assert "RETIRED (superseded)" in run("library", "notes", "x86", "retire", str(new_id), "--reason", "superseded")
        assert "0 note(s)" in run("library", "notes", "x86") and "2 note(s)" in run("library", "notes", "x86", "--all")
        assert "RETIRED" not in run("library", "notes", "x86", "restore", str(new_id))
        (src / "boot.md").write_text(BOOT.replace("reads sectors using", "reads up to 127 sectors using"), encoding="utf-8")
        time.sleep(0.01)
        updated = run("library", "update", "x86")
        assert "Her notebook: 1 note(s). 0 binding(s) moved to the same passage read again, 1 to a passage whose text changed" in updated
        assert "[the passage has changed since]" in run("library", "notes", "x86", "show", str(new_id))
        assert "[the passage has changed since]" not in run("library", "notes", "x86", "confirm", str(new_id))      # confirmed as it now stands
        assert "and the 1 note(s) she has written in its notebook" in run("library", "delete", "x86")
        assert "Usage:" in run("library", "notes", "x86", "frobnicate")
    finally:
        embed.OllamaEmbedder = keep
        sys.modules.pop("hermes_constants", None)


LOADER = """# Loader

The second stage loader of Versa OS is entered from the boot sector with the drive in DL.

## Reading the kernel

The loader reads the kernel image with INT 13h AH=02h, sixteen sectors at a time, converting each LBA to
cylinder, head and sector for a 1.44 MB floppy before every call.

## Entering protected mode

The loader enables the A20 line, loads the GDT and sets the PE bit of CR0 before the far jump.
"""


def two_libraries(tmp_path):
    from holonomic import notebook as nb
    root, _ = built(tmp_path)
    src = tmp_path / "versa-src"
    src.mkdir()
    (src / "loader.md").write_text(LOADER, encoding="utf-8")
    (src / "todo.txt").write_text("Things still to do in the kernel: a file system driver and a shell.\n", encoding="utf-8")
    lib.create(root, "versa", str(src), "The project's own source")
    lib.build(root, "versa", HashEmbedder(), CFG)
    shelf = lib.Shelf(root, HashEmbedder(), CFG)
    x86, versa = shelf.get("x86")[0], shelf.get("versa")[0]
    return nb, root, shelf, x86, versa, src


def test_a_note_points_into_another_library_and_is_found_from_both_sides(tmp_path):
    """Plates cannot tie two stores together.  A note keeps a pointer, and the library pointed into keeps a marker
    on its own plates, so the connection is found whichever passage comes up first."""
    nb, root, shelf, x86, versa, src = two_libraries(tmp_path)
    files = lib.info(root, "x86")["files"]
    emb = HashEmbedder()
    kernel = next(i for i in lib.info(root, "versa")["files"]["loader.md"]["ids"] if versa.get(i)["meta"]["section"].endswith("Reading the kernel"))
    reading = next(i for i in files["boot.md"]["ids"] if x86.get(i)["meta"]["section"] == "Reading sectors")
    assert nb.split(["boot.md > Reading sectors", "versa: loader.md > Reading the kernel", "[x86: fat12.txt]", "C: not a library"], "x86",
                    ["x86", "versa"]) == (["boot.md > Reading sectors", "fat12.txt", "C: not a library"], {"versa": ["loader.md > Reading the kernel"]})
    with pytest.raises(lib.LibraryError, match="has no section"):
        nb.add(x86, files, CFG, text="A note that points at a section that is not in the other library.", about=["versa: loader.md > Nowhere"],
               shelf=shelf, owner="x86")
    assert nb.list_notes(x86) == [] and not nb.has_markers(versa)                       # refused before anything was written
    note = nb.add(x86, files, CFG, kind="connection", shelf=shelf, owner="x86",
                  text="The CHS read described here is the call our loader uses for the kernel, so the geometry must be the floppy's.",
                  about=["boot.md > Reading sectors", "versa: loader.md > Reading the kernel"])
    # one owning notebook; its own passage on its own plates, the other as a pointer
    assert [a["id"] for a in note["about"]] == [reading] and len(note["refs"]) == 1 and nb.list_notes(versa) == []
    ref = note["refs"][0]
    assert ref["library"] == "versa" and ref["uid"] == shelf.uid("versa") and ref["digest"] == nb.digest(versa.get(kernel)["text"])
    assert nb.pointers(note, shelf) == [{"library": "versa", "source": "loader.md > Loader > Reading the kernel", "id": kernel, "state": ""}]
    assert nb.incoming(versa, [kernel]) == {kernel: [{"owner": "x86", "owner_uid": shelf.uid("x86"), "note": note["id"], "stub": ref["stub"]}]}
    c = nb.check(x86, shelf)
    assert c["pointers"] == 1 and c["pointers_stale"] == 0 and c["pointers_found_from_far_side"] == 1 and c["bindings"] == 1
    assert "It ties this to, in another library: versa: loader.md > Loader > Reading the kernel" in nb.line(note, shelf=shelf)
    assert "(that library is not open here)" in nb.line(note, shelf=shelf, open_in=["x86"])
    assert all(h.kind == lib.PIECE for h in versa.recall("a note refers to this passage", k=10, min_score=0.0))      # the marker is not reference material
    # FROM THE OWNING SIDE, both libraries read: the x86 passage comes up, its note brings the loader passage in
    # with it, in place of the weakest piece and not in addition
    q = "the extended read takes a disk address packet in DS:SI"
    wide = dict(CFG, library_score_band=9.0)                                            # so that two pieces are given without it
    plain = lib.search(root, ["x86", "versa"], q, emb, wide, k=2, floor=0.0, notes=False)
    found = lib.search(root, ["x86", "versa"], q, emb, wide, k=2, floor=0.0)
    assert len(found) == len(plain) == 2 and found[0]["id"] == reading and [n["id"] for n in found[0]["notes"]] == [note["id"]]
    assert plain[1]["id"] != kernel and found[1]["library"] == "versa" and found[1]["id"] == kernel
    assert found[1]["brought_by"] == {"library": "x86", "note": note["id"], "with": "x86: boot.md > Reading sectors"}
    one = lib.search(root, ["x86", "versa"], q, emb, CFG, k=1, floor=0.0)
    assert [f["id"] for f in one] == [reading]                                          # no room: nothing is pushed out for it
    assert not any(f.get("brought_by") for f in lib.search(root, ["x86", "versa"], q, emb, dict(CFG, library_follow_notes=0), k=2, floor=0.0))
    # only x86 read: the pointer names the passage and no more
    alone = lib.search(root, ["x86"], q, emb, CFG, k=3, floor=0.0)
    assert all(f["library"] == "x86" for f in alone) and not any(f.get("brought_by") for f in alone)
    # FROM THE FAR SIDE: the loader passage comes up in versa; its plates bring the marker, the marker the note
    q2 = "The loader reads the kernel image sixteen sectors at a time converting each LBA"
    far = next(f for f in lib.search(root, ["versa", "x86"], q2, emb, CFG, k=4, floor=0.0) if f["id"] == kernel)
    assert [(n["library"], n["id"]) for n in far["notes"]] == [("x86", note["id"])] and far["noted_in"] == []
    # versa alone: that there is a note, and where; not what it says
    far = next(f for f in lib.search(root, ["versa"], q2, emb, CFG, k=4, floor=0.0) if f["id"] == kernel)
    assert far["notes"] == [] and far["noted_in"] == ["x86"]
    # a revision keeps the pointer, and the marker names the new note
    better = nb.revise(x86, files, CFG, note["id"], shelf=shelf, owner="x86",
                       text="The CHS read described here is the call our loader uses for the kernel: geometry is 80 cylinders, 2 heads, 18 sectors.")
    assert better["refs"] == note["refs"] and nb.incoming(versa, [kernel])[kernel][0]["note"] == better["id"]
    # retired: nothing in the other library leads to it; restored: it does again
    nb.retire(x86, better["id"], "testing", shelf=shelf)
    assert nb.incoming(versa, [kernel]) == {}
    nb.restore(x86, better["id"], shelf=shelf)
    assert nb.incoming(versa, [kernel])[kernel][0]["note"] == better["id"]
    # the far library is read again: the pointer follows its passage, or says it is stale
    (src / "loader.md").write_text(LOADER.replace("sixteen sectors at a time", "eighteen sectors at a time"), encoding="utf-8")
    time.sleep(0.01)
    report = lib.build(root, "versa", emb, CFG)
    assert report["notes"]["notes"] == 0 and report["notes"]["pointed_at"] == {"markers": 1, "rebound": 0, "changed": 1, "gone": 0}
    now = nb.pointers(nb.get_note(x86, better["id"]), shelf)[0]
    assert now["state"] == "changed" and now["id"] != kernel and "eighteen sectors" in versa.get(now["id"])["text"]
    assert "[the passage has changed since]" in nb.line(nb.get_note(x86, better["id"]), shelf=shelf)
    assert nb.check(x86, shelf)["pointers_stale"] == 1
    assert lib.build(root, "versa", emb, CFG, fresh=True)["notes"]["pointed_at"]["rebound"] == 1          # every passage new: found again
    assert nb.pointers(nb.get_note(x86, better["id"]), shelf)[0]["id"] in lib.info(root, "versa")["files"]["loader.md"]["ids"]
    (src / "loader.md").unlink()
    time.sleep(0.01)
    assert lib.build(root, "versa", emb, CFG)["notes"]["pointed_at"]["gone"] == 1
    assert nb.pointers(nb.get_note(x86, better["id"]), shelf)[0]["state"] == "gone"
    # the far library is removed, and another made under its name: the pointer does not take it for the old one
    old_uid = shelf.uid("versa")
    lib.delete(root, "versa")
    assert nb.pointers(nb.get_note(x86, better["id"]), shelf)[0]["state"] == "no library"
    time.sleep(0.01)
    lib.create(root, "versa", str(src))
    assert shelf.uid("versa") != old_uid and nb.pointers(nb.get_note(x86, better["id"]), shelf)[0]["state"] == "no library"


def test_a_marker_whose_note_is_gone_leads_nowhere(tmp_path):
    nb, root, shelf, x86, versa, src = two_libraries(tmp_path)
    emb = HashEmbedder()
    note = nb.add(versa, lib.info(root, "versa")["files"], CFG, kind="connection", shelf=shelf, owner="versa",
                  text="Our loader depends on the BIOS loading the boot sector at 0x7C00 as described in the reference.",
                  about=["loader.md > Loader", "x86: boot.md > Boot sector"])
    boot = nb.pointers(note, shelf)[0]["id"]
    q = "The boot sector is the first 512 bytes of the disk and ends with the signature 0xAA55"
    assert next(f for f in lib.search(root, ["x86"], q, emb, CFG, floor=0.0) if f["id"] == boot)["noted_in"] == ["versa"]
    lib.delete(root, "versa")                                                       # the owning library is removed outright
    assert nb.incoming(x86, [boot])[boot]
    assert next(f for f in lib.search(root, ["x86"], q, emb, CFG, floor=0.0) if f["id"] == boot)["noted_in"] == []
    assert nb.incoming(x86, [boot]) == {}                                           # and the marker has been retired


def test_she_ties_two_libraries_together_through_her_tool(tmp_path):
    if not HAVE_HERMES: return
    src = folder(tmp_path)
    other = tmp_path / "versa-src"
    other.mkdir()
    (other / "loader.md").write_text(LOADER, encoding="utf-8")
    p = make(tmp_path)
    root = lib.root_of(p._engine)
    for name, where in (("x86", src), ("versa", other)):
        lib.create(root, name, str(where))
        lib.build(root, name, p._engine.embedder, p._cfg)
    assert "'other-library: FILE > SECTION'" in p.system_prompt_block()
    p.prefetch("Use the x86 library and the versa library for this.", session_id="s1")
    assert lib_tool(p, action="open", names=["x86", "versa"])["opened"] is True
    assert "Say which library" in lib_tool(p, action="note", text="A note with two libraries open and neither of them named.")["error"]
    saved = lib_tool(p, action="note", name="x86", type="connection",
                     text="The CHS read described here is the call the Versa loader makes for the kernel image.",
                     passages=["boot.md > Reading sectors", "versa: loader.md > Reading the kernel"])
    assert saved["saved"] and saved["about"] == ["boot.md > Reading sectors"] and "warning" not in saved
    assert saved["points_to_in_other_libraries"] == ["versa: loader.md > Loader > Reading the kernel"]
    # the x86 passage comes up: the note, and the loader passage it brings in
    context = p.prefetch("INT 13h with AH=02h reads sectors using cylinder, head and sector numbers", session_id="s1")
    assert f"(note {saved['note_id']}, how it bears on the project" in context
    assert "It ties this to, in another library: versa: loader.md > Loader > Reading the kernel" in context
    assert f"(brought in by your note {saved['note_id']}, which ties it to x86: boot.md > Reading sectors)" in context
    assert "sixteen sectors at a time" in context
    # the loader passage comes up: the note from the x86 notebook comes with it
    context = p.prefetch("The loader reads the kernel image sixteen sectors at a time converting each LBA", session_id="s1")
    assert f"- (from your notebook in x86) (note {saved['note_id']}" in context
    # a conversation with only versa open is told there is a note, and nothing of what it says
    p.prefetch("Open the versa library please.", session_id="s2")
    assert json.loads(p.handle_tool_call("holonomic_library", {"action": "open", "names": ["versa"]}, session_id="s2"))["opened"] is True
    context = p.prefetch("The loader reads the kernel image sixteen sectors at a time converting each LBA", session_id="s2")
    assert "A note of yours in the notebook of 'x86' refers to this passage; x86 is not open in this conversation." in context
    assert "the call the Versa loader makes" not in context
    found = json.loads(p.handle_tool_call("holonomic_library", {"action": "search", "query": "The loader reads the kernel image sixteen sectors at a time"},
                                          session_id="s2"))
    hit = next(r for r in found["results"] if r["source"].endswith("Reading the kernel"))
    assert hit["also_noted"] == ["A note of yours in the notebook of 'x86' refers to this passage; x86 was not searched."]
    p.shutdown()


# ----------------------------------------------------------------- leaving files out, and pictures

def test_a_folder_says_what_to_leave_out(tmp_path):
    """A source tree holds things that are text and are not reference material: a tool's own source, a data file."""
    assert lib.ignored("CH01/HEXCONV/HEXCONVU.PAS", ["*.pas"]) and lib.ignored("ch13/EX13_1.IN", ["EX13_1.IN"])
    assert lib.ignored("CH01/HEXCONV/x.asm", ["CH01/"]) and lib.ignored("a/CH01/x.asm", ["ch01/"]) and not lib.ignored("CH010/x.asm", ["CH01/"])
    assert lib.ignored("include/conv.a6", ["INCLUDE/*.A6"]) and not lib.ignored("include/conv.a", ["INCLUDE/*.A6"])
    assert not lib.ignored("src/pas.asm", ["*.pas"]) and not lib.ignored("CH01.asm", ["CH01/"])
    src = folder(tmp_path)
    (src / "tool").mkdir()
    (src / "tool" / "gui.pas").write_text("program gui;\nbegin\n  writeln('a window with buttons');\nend.\n", encoding="utf-8")
    (src / "big.in").write_text("data " * 400, encoding="utf-8")
    root = tmp_path / "store" / "libraries"
    lib.create(root, "x86", str(src))
    first = lib.build(root, "x86", HashEmbedder(), CFG)
    assert first["ignored"] == 0 and {"tool/gui.pas", "big.in"} <= set(lib.info(root, "x86")["files"])
    (src / lib.IGNORE_FILE).write_text("# the lab tool, and its data\ntool/\n*.in\n", encoding="utf-8")
    again = lib.build(root, "x86", HashEmbedder(), CFG)
    data = lib.info(root, "x86")
    assert again["ignored"] == 2 and again["removed"] == 2 and sorted(data["ignored"]) == ["big.in", "tool/gui.pas"]
    assert not {"tool/gui.pas", "big.in", lib.IGNORE_FILE} & set(data["files"]) and "boot.md" in data["files"]
    assert lib.summary(root, "x86")["ignored"] == 2
    assert not [f for f in lib.search(root, ["x86"], "a window with buttons", HashEmbedder(), CFG, floor=0.0) if f["file"] == "tool/gui.pas"]
    (src / lib.IGNORE_FILE).write_text("*.in\n", encoding="utf-8")                  # and what is let back in is read again
    assert lib.build(root, "x86", HashEmbedder(), CFG)["added"] == 1 and "tool/gui.pas" in lib.info(root, "x86")["files"]


def figure_folder(tmp_path):
    from test_images import picture
    src = folder(tmp_path)
    (src / "figures").mkdir()
    (src / "figures" / "bus.png").write_bytes(picture(320, 200, colour=(250, 240, 10)))
    (src / "figures" / "flags.gif").write_bytes(picture(300, 60, colour=(10, 10, 10), fmt="GIF"))
    (src / "cover.jpg").write_bytes(picture(200, 300, colour=(90, 10, 200)))
    (tmp_path / "elsewhere.png").write_bytes(picture(64, 64, colour=(1, 2, 3)))
    (src / "bus.md").write_text(
        "# The system bus\n\nThe CPU reaches memory over the address bus and the data bus.\n\n"
        "![Figure 3.5: Eight-bit CPU and memory joined by an address bus and a data bus](figures/bus.png)\n\n"
        "The figure shows a box labelled CPU on the left and a column of memory cells on the right.\n\n"
        "## Flags\n\nThe flags register holds the carry and zero flags among others.\n\n![](figures/flags.gif)\n\n"
        "A picture that is not there: ![gone](figures/none.png). One from outside: ![out](../elsewhere.png). "
        "One on the web: ![web](http://example.com/x.png).\n", encoding="utf-8")
    return src


def test_a_library_keeps_its_own_pictures(tmp_path):
    """Diagrams belong with the material they are part of, apart from the images she has been shown.  A passage
    that refers to one says so and gives its file, so she can show the user the figure itself or look at it."""
    from holonomic import images
    if not images._pil():
        return
    src = figure_folder(tmp_path)
    root = tmp_path / "store" / "libraries"
    lib.create(root, "x86", str(src))
    report = lib.build(root, "x86", HashEmbedder(), CFG)
    assert report["figures"] == 2 and report["pictures"] == 1 and not [s for s in report["skipped"] if "figure" in s["why"]]
    store = lib.store(root, "x86", HashEmbedder(), CFG)
    held = lib.figures(root, "x86", HashEmbedder(), CFG)
    assert [(g["name"], g["from"]) for g in held] == [("bus.png", "bus.md"), ("flags.gif", "bus.md"), ("cover.jpg", "cover.jpg")]
    bus, flags, cover = held
    assert bus["caption"].startswith("Figure 3.5: Eight-bit CPU") and flags["caption"] == "flags" and cover["caption"] == "cover"
    # kept inside the library's own folder, and the one a chat window cannot show has a copy it can
    for g in held:
        assert str(root / "x86" / "images") in g["file"] and open(g["file"], "rb").read()
    assert flags["file"].endswith(".view.jpg") and flags["original"].endswith(".gif") and bus["file"].endswith(".png")
    assert [g["figure"] for g in lib.figures(root, "x86", HashEmbedder(), CFG, "address bus")] == [bus["figure"]]
    # the passage names the figure where the reference stood, and carries it
    found = lib.search(root, ["x86"], "the CPU reaches memory over the address bus and the data bus", HashEmbedder(),
                       dict(CFG, library_score_band=9.0), k=12, floor=0.0)
    hit = next(f for f in found if f"[figure {bus['figure']}: Figure 3.5" in f["text"])
    assert [g["figure"] for g in hit["figures"]] == [bus["figure"]] and "figures/bus.png" not in hit["text"]
    other = next(f for f in found if f"[figure {flags['figure']}]" in f["text"])
    assert "[figure, its file is missing: gone]" in other["text"] and "![out](../elsewhere.png)" in other["text"] and "![web](http://example.com/x.png)" in other["text"]
    assert all(not f["figures"] for f in found if "[figure " not in f["text"])
    lib.open_for(root, "s1", ["x86"])
    block, _ = lib.context_block(root, "s1", "the CPU reaches memory over the address bus and the data bus", HashEmbedder(), CFG)
    assert f"Figure {bus['figure']} (Figure 3.5: Eight-bit CPU" in block and f"is kept at: {bus['file']}" in block
    assert "write MEDIA: followed by its file path" in block and "action 'look'" in block
    plain, _ = lib.context_block(root, "s1", "FAT12 keeps cluster numbers in 12 bits, packed three bytes to two entries", HashEmbedder(), dict(CFG, library_k=1))
    assert "MEDIA" not in plain                                                        # no figure given: nothing said about figures
    # she can look at it herself
    asked = []

    def see(step, system, prompt, jpeg, schema, max_tokens):
        asked.append((prompt, len(jpeg)))
        return json.dumps({"answer": "A box labelled CPU joined to memory by two yellow bands."})
    assert lib.look_at(root, "x86", HashEmbedder(), CFG, bus["figure"], "What joins the CPU to memory?", see=see).startswith("A box labelled CPU")
    assert "What joins the CPU to memory?" in asked[0][0] and asked[0][1] > 100
    with pytest.raises(lib.LibraryError, match="no figure 999"):
        lib.look_at(root, "x86", HashEmbedder(), CFG, 999, "anything", see=see)
    # read again, every passage is new and the pictures are the same pictures
    again = lib.build(root, "x86", HashEmbedder(), CFG, fresh=True)
    assert again["figures"] == 2 and again["pictures"] == 0 and len(lib.figures(root, "x86", HashEmbedder(), CFG)) == 3
    assert next(f for f in lib.search(root, ["x86"], "the CPU reaches memory over the address bus", HashEmbedder(), CFG, k=6, floor=0.0)
                if "[figure " in f["text"] and "Figure 3.5" in f["text"])["figures"][0]["figure"] == bus["figure"]
    assert lib.build(root, "x86", HashEmbedder(), CFG)["pictures"] == 0                 # nothing changed: nothing taken in twice
    # pictures can be left out like anything else
    (src / lib.IGNORE_FILE).write_text("cover.jpg\n", encoding="utf-8")
    (src / "more.png").write_bytes(__import__("test_images").picture(50, 50, colour=(7, 7, 7)))
    r = lib.build(root, "x86", HashEmbedder(), CFG)
    assert r["ignored"] == 1 and r["pictures"] == 1


def test_a_librarys_pictures_are_not_hers(tmp_path):
    if not HAVE_HERMES: return
    from holonomic import images
    if not images._pil():
        return
    src = figure_folder(tmp_path)
    p = make(tmp_path)
    root = lib.root_of(p._engine)
    lib.create(root, "x86", str(src))
    lib.build(root, "x86", p._engine.embedder, p._cfg)
    assert images.count_images(p._engine) == 0 and images.list_images(p._engine) == [] and images.pending(p._engine)["images"] == 0
    assert "FIGURES." in __import__("holonomic.provider", fromlist=["LIBRARY_TOOL"]).LIBRARY_TOOL["description"]
    p.prefetch("Use the x86 library for this.", session_id="s1")
    assert lib_tool(p, action="open", names=["x86"])["opened"] is True
    held = lib_tool(p, action="figures")
    assert held["count"] == 3 and held["figures"][0]["caption"].startswith("Figure 3.5")
    assert [g["from"] for g in lib_tool(p, action="figures", query="flags")["figures"]] == ["bus.md"]
    found = lib_tool(p, action="search", query="the CPU reaches memory over the address bus and the data bus", limit=6)
    hit = next(r for r in found["results"] if r.get("figures"))
    assert hit["figures"][0]["file"].startswith(str(root / "x86" / "images")) and "[figure " in hit["text"]
    assert "look needs 'figure'" in lib_tool(p, action="look")["error"] and "no figure 999" in lib_tool(p, action="look", figure=999)["error"]
    keep = images._seer
    images._seer = lambda ic, report: (lambda step, system, prompt, jpeg, schema, max_tokens: json.dumps({"answer": "Two bands, labelled Address and Data."}))
    try:
        seen = lib_tool(p, action="look", figure=hit["figures"][0]["figure"], question="What are the bands labelled?")
    finally:
        images._seer = keep
    assert seen["what_you_saw"] == "Two bands, labelled Address and Data." and seen["file"] == hit["figures"][0]["file"]
    assert "your own reading" in seen["note"]
    context = p.prefetch("How does the CPU reach memory over the address bus and the data bus?", session_id="s1")
    assert "is kept at: " + hit["figures"][0]["file"] in context
    assert images.count_images(p._engine) == 0                                         # still none of hers
    p.shutdown()


def test_an_assembly_include_file_ending_in_a_is_read_and_a_code_archive_is_not(tmp_path):
    from holonomic import library as L
    inc = tmp_path / "STDLIB.A"
    inc.write_bytes(b"; the standard library's include file\r\nextrn sl_putc:far\r\nputc macro\r\n call sl_putc\r\n endm\r\n" * 4)
    assert "sl_putc" in L.read_file(inc, 10_000_000)
    ar = tmp_path / "libthing.a"
    ar.write_bytes(b"!<arch>\nthing.o/        0           0     0     644     120       `\n" + b"code " * 60)
    try:
        L.read_file(ar, 10_000_000)
        assert False, "a code archive was read as text"
    except L.LibraryError as e:
        assert "not a text file" in str(e)
