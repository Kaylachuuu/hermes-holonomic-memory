"""Provider tests.  They need the Hermes source for the real MemoryProvider base class:
set HERMES_SRC, or keep a checkout next to this folder, or run inside Hermes' environment."""
import importlib.util, json, os, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for cand in (os.environ.get("HERMES_SRC"), ROOT.parent / "hermes-agent"):
    if cand and (Path(cand) / "agent" / "memory_provider.py").exists():
        sys.path.insert(0, str(cand))
        break
try:
    from agent.memory_provider import MemoryProvider
    HAVE_HERMES = True
except Exception:
    HAVE_HERMES = False

from holonomic import HolonomicMemory, HashEmbedder


def make(tmp_path, **kwargs):
    from holonomic.provider import HolonomicMemoryProvider
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "holonomic.json").write_text(json.dumps({"embedder": "hash", "min_score": 0.3}))
    p = HolonomicMemoryProvider()
    p.initialize(kwargs.pop("session_id", "s1"), hermes_home=str(home), platform="cli", **kwargs)
    return p


def tool(p, **args):
    return json.loads(p.handle_tool_call("holonomic_memory", args))


def test_loads_the_way_hermes_loads_it():
    if not HAVE_HERMES: return
    assert "register_memory_provider" in (ROOT / "__init__.py").read_text()
    name = "hermes_user_plugins.holonomic__source_test"
    import types
    sys.modules.setdefault("hermes_user_plugins", types.ModuleType("hermes_user_plugins")).__path__ = []
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    got = []
    mod.register(type("Ctx", (), {"register_memory_provider": lambda self, prov: got.append(prov)})())
    assert isinstance(got[0], MemoryProvider) and got[0].name == "holonomic" and got[0].is_available()
    assert got[0].get_tool_schemas()[0]["name"] == "holonomic_memory"
    assert {f["key"] for f in got[0].get_config_schema()} == {"ollama_host", "embed_model"}


def test_turns_are_stored_and_recalled_in_a_later_session(tmp_path):
    if not HAVE_HERMES: return
    p = make(tmp_path)
    p.sync_turn("My sister Mara moved to Lisbon last spring", "That sounds like a big change for Mara and for you.", session_id="s1")
    p.sync_turn("Shellfish gives her a dangerous allergic reaction", "Understood, I will keep the allergy in mind.", session_id="s1")
    p.sync_turn("ok", "Sure.", session_id="s1")                       # trivial prompt: not stored
    assert json.loads(p.handle_tool_call("holonomic_memory", {"action": "stats"}))["memories"] == 4
    # same session: those turns are still in the context window, so nothing is injected
    assert p.prefetch("where did Mara move to, was it Lisbon?", session_id="s1") == "" and p.recall_status() is None
    # a later session recalls them, including the linked allergy turn
    block = p.prefetch("where did Mara move to, was it Lisbon?", session_id="s2")
    assert "Mara moved to Lisbon" in block and "Shellfish" in block and "linked" in block and "[#1]" in block
    assert p.recall_status().count >= 2
    assert p.prefetch("thanks!", session_id="s2") == ""
    # after compression the same session may recall its own older turns again
    p.on_pre_compress([])
    assert "Lisbon" in p.prefetch("where did Mara move to, was it Lisbon?", session_id="s1")
    assert "Holonomic Memory" in p.system_prompt_block() and "4 memories" in p.system_prompt_block()
    p.shutdown()


def test_tool_actions(tmp_path):
    if not HAVE_HERMES: return
    p = make(tmp_path)
    r = tool(p, action="remember", content="Kayla runs Gemma on a V100 in the hermes-ollama VM", about=["Kayla"], importance=3)
    mid = r["stored"][0]
    assert "Kayla" in r["filed_under"]
    assert tool(p, action="recall", query="which GPU does Gemma run on, the V100?")["results"][0]["id"] == mid
    assert [h["id"] for h in tool(p, action="related", entity="kayla")["results"]] == [mid]
    assert tool(p, action="feedback", memory_id=mid, rating="helpful")["trust"] == 0.9
    assert tool(p, action="feedback", memory_id=mid, rating="wrong")["trust"] == 0.7
    assert tool(p, action="forget", memory_id=mid) == {"forgotten": True}
    assert tool(p, action="recall", query="which GPU does Gemma run on, the V100?")["count"] == 0
    for bad in ({"action": "recall"}, {"action": "nope"}, {"action": "forget", "memory_id": 999}, {"action": "related"}):
        assert "error" in tool(p, **bad)
    assert "error" in json.loads(p.handle_tool_call("other_tool", {}))
    p.shutdown()


def test_builtin_memory_writes_are_mirrored(tmp_path):
    if not HAVE_HERMES: return
    p = make(tmp_path)
    p.on_memory_write("add", "user", "Prefers concise answers without filler")
    p.on_memory_write("add", "user", "Prefers concise answers without filler")          # no duplicate
    assert tool(p, action="stats")["memories"] == 1
    p.on_memory_write("replace", "user", "Prefers concise answers with examples",
                      {"previous_content": "Prefers concise answers without filler"})
    hits = tool(p, action="recall", query="concise answers preference")["results"]
    assert [h["text"] for h in hits] == ["Prefers concise answers with examples"] and hits[0]["kind"] == "core"
    p.on_memory_write("remove", "user", "", {"previous_content": "Prefers concise answers with examples"})
    assert tool(p, action="stats")["memories"] == 0
    p.shutdown()


def test_subagents_read_but_do_not_write(tmp_path):
    if not HAVE_HERMES: return
    main = make(tmp_path)
    main.sync_turn("The deploy script lives in the tools folder of the infra repo", "Noted, tools folder of the infra repo.", session_id="s1")
    sub = make(tmp_path, session_id="child", agent_context="subagent")
    assert sub._engine is main._engine                                 # one shared engine per data directory
    sub.sync_turn("Subagent chatter that should not be remembered at all", "Indeed it should not be stored.", session_id="child")
    assert tool(main, action="stats")["memories"] == 2
    assert "deploy script" in sub.prefetch("where does the deploy script live in the infra repo?", session_id="child")
    sub.shutdown()
    assert tool(main, action="stats")["memories"] == 2                 # still open for the main agent
    main.shutdown()


def test_failed_writes_are_retried(tmp_path):
    if not HAVE_HERMES: return
    p = make(tmp_path)
    real = p._engine.remember
    calls = {"n": 0}
    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("embedding server down")
        return real(*a, **k)
    p._engine.remember = flaky
    p.sync_turn("First message about the violin lesson on Tuesday", "Got it, violin lesson on Tuesday.", session_id="s1")
    assert len(p._backlog) == 1 and tool(p, action="stats")["memories"] == 0
    p.sync_turn("Second message about the cello recital in spring", "Got it, cello recital in spring.", session_id="s1")
    assert not p._backlog and tool(p, action="stats")["memories"] == 4
    p._engine.remember = real
    p.shutdown()


def test_unreachable_embedding_server_degrades_quietly(tmp_path):
    if not HAVE_HERMES: return
    from holonomic.provider import HolonomicMemoryProvider
    home = tmp_path / "home"; home.mkdir()
    (home / "holonomic.json").write_text(json.dumps({"ollama_host": "http://127.0.0.1:9", "embed_timeout": 1}))
    p = HolonomicMemoryProvider()
    p.initialize("s1", hermes_home=str(home), platform="cli")
    assert p._engine is None and "unreachable" in p.system_prompt_block()
    assert p.prefetch("what do you remember about the server?") == ""
    p.sync_turn("Something worth keeping about the server rack", "Acknowledged, noted about the server rack.", session_id="s1")
    assert len(p._backlog) == 1 and "error" in tool(p, action="stats")
    p.shutdown()


def test_config_roundtrip(tmp_path):
    if not HAVE_HERMES: return
    from holonomic.provider import HolonomicMemoryProvider, load_config
    home = tmp_path / "home"; home.mkdir()
    (home / "holonomic.json").write_text(json.dumps({"recall_k": 9}))
    HolonomicMemoryProvider().save_config({"ollama_host": "http://10.0.0.5:11434", "embed_model": ""}, str(home))
    cfg = load_config(home)
    assert cfg["ollama_host"] == "http://10.0.0.5:11434" and cfg["recall_k"] == 9 and cfg["embed_model"] == "nomic-embed-text"


def test_two_processes_do_not_overwrite_each_others_plates(tmp_path):
    a = HolonomicMemory(tmp_path / "m", HashEmbedder())
    b = HolonomicMemory(tmp_path / "m", HashEmbedder())               # stands in for a second process
    a.remember("Alpha observatory opened on the mountain ridge", session="a")
    ta = a.remember("Its telescope mirror was polished for two years", session="a")[0]
    b.remember("Bravo bakery started selling cardamom buns downtown", session="b")
    tb = b.remember("The queue reaches around the corner every Saturday", session="b")[0]
    a.remember("Charlie the cat sleeps on the warm router", session="c")
    for eng in (a, b):
        assert eng.stats()["memories"] == 5
        assert ta in {h.id for h in eng.recall("Alpha observatory mountain ridge opened", k=4) if h.assoc > 0.3}
        assert tb in {h.id for h in eng.recall("Bravo bakery cardamom buns downtown", k=4) if h.assoc > 0.3}
    a.close(); b.close()


def test_key_extraction():
    if not HAVE_HERMES: return
    from holonomic.provider import extract_keys, clean_for_storage
    keys = extract_keys("Yesterday Mara Oliveira flew to Lisbon. The flight used `TAP 204` and Mara loved it.")
    assert "Mara Oliveira" in keys and "Lisbon" in keys and "TAP 204" in keys and "The" not in keys and "Yesterday" not in keys
    assert extract_keys("i think it is fine. so do it.") == []
    cleaned = clean_for_storage("Here:\n```python\nprint(1)\n```\ndone <memory-context>secret</memory-context>", 100)
    assert "print" not in cleaned and "secret" not in cleaned and "[code omitted]" in cleaned


def test_package_loads_and_registers_without_numpy():
    """Hermes only lists providers whose package imports; numpy is installed after selection."""
    if not HAVE_HERMES: return
    import subprocess
    hermes = next(p for p in sys.path if (Path(p) / "agent" / "memory_provider.py").exists())
    code = f"""
import importlib.abc, importlib.util, sys, types
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("numpy", "threadpoolctl"):
            raise ImportError("blocked: " + name)
sys.meta_path.insert(0, Block())
sys.path.insert(0, {hermes!r})
sys.modules["hermes_user_plugins"] = types.ModuleType("hermes_user_plugins"); sys.modules["hermes_user_plugins"].__path__ = []
name = "hermes_user_plugins.holonomic__source_x"
spec = importlib.util.spec_from_file_location(name, {str(ROOT / "__init__.py")!r}, submodule_search_locations=[{str(ROOT)!r}])
mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
got = []
mod.register(type("Ctx", (), {{"register_memory_provider": lambda self, p: got.append(p)}})())
p = got[0]
assert p.name == "holonomic" and p.is_available() is False and "numpy" in p.unavailable_reason()
assert len(p.get_config_schema()) == 2 and p.get_tool_schemas()
assert "numpy" not in sys.modules
print("ok")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stderr[-800:]


def test_top_folder_only_holds_modules_that_are_safe_to_execute():
    """Hermes executes every top-level .py when it loads the plugin, before __init__.py."""
    assert sorted(p.name for p in ROOT.glob("*.py")) == ["__init__.py", "cli.py", "embed.py", "engine.py", "images.py", "paint.py", "provider.py", "reflect.py", "sleep.py", "vsa.py"]


def test_failed_first_open_releases_the_database_file(tmp_path):
    from holonomic import OllamaEmbedder, EmbeddingError
    with pytest.raises(EmbeddingError):
        HolonomicMemory(tmp_path / "m", OllamaEmbedder(host="http://127.0.0.1:9", timeout=1))
    (tmp_path / "m" / "holonomic.db").unlink()          # would raise on Windows if the handle had leaked


def test_status_config_works_uninitialised():
    if not HAVE_HERMES: return
    from holonomic.provider import HolonomicMemoryProvider
    assert set(HolonomicMemoryProvider().get_status_config({})) == {"ollama_host", "embed_model", "recall_k", "min_score"}


def test_hermes_real_plugin_loader_loads_the_folder():
    if not HAVE_HERMES: return
    try:
        from plugins import plugin_loader
    except Exception:
        return
    import logging
    mod = plugin_loader.load_plugin_module("hermes_user_plugins.holonomic__source_loader", ROOT,
                                           parents=("plugins", "plugins.memory"), logger=logging.getLogger("test"),
                                           synthetic_namespace="hermes_user_plugins")
    got = []
    mod.register(type("Ctx", (), {"register_memory_provider": lambda self, prov: got.append(prov)})())
    assert got and got[0].name == "holonomic"


def test_cli_commands(tmp_path, capsys=None):
    if not HAVE_HERMES: return
    import argparse, contextlib, io, types
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
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for argv in (["stats"], ["list", "-n", "5"], ["recall", "what is my name Kayla", "-k", "3"], []):
                args = parser.parse_args(argv); args.func(args)
        text = out.getvalue()
        assert "memories: 2" in text and "[#1]" in text and "My name is Kayla" in text and "content words: name, kayla" in text and "Usage:" in text
    finally:
        embed.OllamaEmbedder = real
        sys.modules.pop("hermes_constants", None)


def test_sentences_questions_and_denials(tmp_path):
    if not HAVE_HERMES: return
    from holonomic.provider import strip_memory_denials, is_question
    assert strip_memory_denials("I do not have access to your personal identity or your name.") == ""
    assert strip_memory_denials("I don't have a persistent memory of our past conversations. Each session is independent, "
                                "so I only know this chat. Your OS project sounds great.") == "Your OS project sounds great."
    assert strip_memory_denials("I don't know your name.") == ""
    assert strip_memory_denials("I can't open that port. No config was loaded.") == "I can't open that port. No config was loaded."
    p = make(tmp_path)
    p.sync_turn("Okay! My name is Kayla. I'm a 45 year old woman, mother of 5 children. What should we build first?",
                "It's a pleasure to meet you, Kayla.", session_id="s1")
    p.sync_turn("What is my name?", "I do not have access to your personal identity or your name.", session_id="s1")
    p.sync_turn("What is my name?", "It is not something we have discussed yet.", session_id="s1")   # short reply to a bare question
    rows = p._engine.recent(10)
    kinds = {r["text"]: r["kind"] for r in rows}
    assert kinds["Okay! My name is Kayla."] == "said_user" and kinds["What is my name?"] == "asked_user"
    assert kinds["What should we build first?"] == "asked_user" and len(rows) == 6       # neither reply was stored
    block = p.prefetch("What is my name?", session_id="s2")
    assert "My name is Kayla" in block and "What is my name?" not in block and "asked" not in block
    hits = tool(p, action="recall", query="What is my name?")["results"]
    assert hits[0]["text"] == "Okay! My name is Kayla." and all(h["kind"] != "asked_user" for h in hits)
    p.shutdown()


def test_a_question_finds_its_answer_through_shared_words(tmp_path):
    """From a real session: "what is my name" vs "Hello! My name is Kayla..." scored 0.24 as vectors."""
    if not HAVE_HERMES: return
    from holonomic.provider import select_for_injection
    p = make(tmp_path)
    p.sync_turn("Hello! My name is Kayla, I'm excited to start working with you!", "Hello, Kayla. I'm Hermes, an assistant built to help.", session_id="s1")
    p.sync_turn("The project I'm most looking forward to is an operating system written in assembly.",
                "That is a significant undertaking. Should we start by mapping out the architecture?", session_id="s1")
    plain = p._engine.recall("what is my name", k=5, min_score=0.0)
    boosted = p._engine.recall("what is my name", k=5, min_score=0.0, lexical=0.2)
    assert boosted[0].text.startswith("Hello! My name is Kayla") and boosted[0].lexical == pytest.approx(0.2)
    assert boosted[0].direct == pytest.approx(next(h for h in plain if h.id == boosted[0].id).direct + 0.2, abs=1e-3)
    assert "My name is Kayla" in p.prefetch("What is my name?", session_id="s2")
    # the band keeps a clear winner from dragging weak matches in with it
    class H:  # minimal stand-in
        def __init__(self, s): self.score = s
    band = {"min_score": 0.2, "score_band": 0.25}
    kept = select_for_injection([H(0.61), H(0.50), H(0.28), H(0.21)], band)
    assert [h.score for h in kept] == [0.61, 0.50]
    kept = select_for_injection([H(0.28), H(0.24), H(0.03)], band)
    assert [h.score for h in kept] == [0.28, 0.24]
    p.shutdown()


def test_old_store_gains_word_matching_on_open(tmp_path):
    import sqlite3
    m = HolonomicMemory(tmp_path / "m", HashEmbedder())
    m.remember("The telescope mirror was polished for two years", session="a")
    m.close()
    db = sqlite3.connect(tmp_path / "m" / "holonomic.db")
    db.executescript("DROP TRIGGER memories_fts_ai; DROP TRIGGER memories_fts_ad; DROP TRIGGER memories_fts_au; DROP TABLE memories_fts;")
    db.close()
    m = HolonomicMemory(tmp_path / "m", HashEmbedder())
    assert m.recall("telescope", k=1, min_score=0.0, lexical=0.2)[0].lexical == pytest.approx(0.2)
    mid = m.remember("A second note about the telescope dome", session="a")[0]
    assert m.forget(mid) and all(h.id != mid for h in m.recall("dome", k=5, min_score=0.0, lexical=0.2))
    m.close()


def test_cli_show_and_forget(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, io, types
    p = make(tmp_path)
    p.sync_turn("My name is Kayla and I build memory systems for fun", "Nice to meet you, Kayla, memory systems are a fine hobby.", session_id="s1")
    fact = p._engine.remember("Kayla builds memory systems.", kind="fact", session="reflection", chain=False, links=[1], meta={"sources": [1]})[0]
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
                args = parser.parse_args(list(argv)); args.func(args)
            return out.getvalue()
        shown = run("show", str(fact), "999")
        assert "Kayla builds memory systems." in shown and "drawn from: #1" in shown and "linked to [#1]" in shown and "[#999] no such memory" in shown
        preview = run("forget", str(fact))
        assert "Nothing removed" in preview and "Kayla builds memory systems." in preview and "Kayla builds" in run("list")
        assert f"Forgot #{fact}" in run("forget", str(fact), "--yes")
        assert "Kayla builds memory systems." not in run("list") and "no such memory" in run("forget", str(fact), "--yes")
    finally:
        embed.OllamaEmbedder = real
        sys.modules.pop("hermes_constants", None)
