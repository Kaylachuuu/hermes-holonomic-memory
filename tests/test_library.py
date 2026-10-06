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
        # She asks, the user says yes: now it opens.  A yes to nothing in particular opens nothing.
        p.prefetch("yes please", session_id="monday")
        assert lib_tool(p, action="open", names=["x86"])["opened"] is True
        assert lib_tool(p, action="close")["open_in_this_conversation"] == []
        p.prefetch("yes please", session_id="monday")
        assert lib_tool(p, action="open", names=["x86"])["opened"] is False                  # that yes was already used
        p.prefetch("no, leave it closed", session_id="monday")
        assert lib_tool(p, action="open", names=["x86"])["opened"] is False
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
        assert p._engine.stats()["memories"] == 2
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
    src = tmp_path / "os"; src.mkdir()
    (src / "boot.asm").write_text("; boot sector\nstart:\n    mov si, msg\n    call print\n    jmp $\n")
    (src / "CHOICE.COM").write_bytes(com)
    (src / "SHELL.EXE").write_bytes(b"MZ" + bytes(200))
    (src / "CONFIG.SYS").write_text("FILES=30\r\nBUFFERS=20\r\n")
    root = tmp_path / "store" / "libraries"
    lib.create(root, "my-os", str(src))
    try:
        report = lib.build(root, "my-os", HashEmbedder(), CFG)
        assert report["added"] == 2 and {s["file"]: s["why"] for s in report["skipped"]} == {"CHOICE.COM": "not a text file", "SHELL.EXE": "not a text file"}
        assert sorted(lib.info(root, "my-os")["files"]) == ["CONFIG.SYS", "boot.asm"]
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
