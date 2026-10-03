"""`hermes holonomic ...` commands: look inside the memory store from a terminal.

    hermes holonomic stats
    hermes holonomic list [-n 20]
    hermes holonomic recall "what is my name" [-k 10]

Only stdlib imports at module level: Hermes imports this file while building its
command line, before any provider dependency is needed.
"""

from __future__ import annotations

import sys
import time


def _open():
    from hermes_constants import get_hermes_home
    from .embed import OllamaEmbedder
    from .engine import HolonomicMemory
    from .provider import load_config
    home = get_hermes_home()
    cfg = load_config(home)
    data_dir = home / "holonomic"
    if not (data_dir / "holonomic.db").exists():
        print(f"No memory store yet at {data_dir}. It is created on the first conversation.")
        return None, cfg
    embedder = OllamaEmbedder(cfg["embed_model"], cfg["ollama_host"], timeout=float(cfg["embed_timeout"]))
    return HolonomicMemory(data_dir, embedder, dim=int(cfg["dim"]), plate_capacity=float(cfg["plate_capacity"])), cfg


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 3] + "..."


def holonomic_command(args) -> None:
    try:
        sys.stdout.reconfigure(errors="replace")       # Windows consoles choke on some characters
    except Exception:
        pass
    action = getattr(args, "holonomic_action", None)
    if action not in ("stats", "list", "recall"):
        print('Usage: hermes holonomic stats | list [-n N] | recall "query" [-k N]')
        return
    engine, cfg = _open()
    if engine is None:
        return
    try:
        if action == "stats":
            for key, value in engine.stats().items():
                print(f"  {key}: {value}")
            for key in ("min_score", "score_band", "lexical_weight", "assistant_weight", "recall_k"):
                print(f"  {key} (config): {cfg[key]}")
        elif action == "list":
            rows = engine.recent(args.n)
            print(f"{len(rows)} most recent memories (newest first):")
            for r in rows:
                print(f"  [#{r['id']}] {_when(r['created_at'])} {r['kind']:<14} {len(r['text']):>5} chars  {_clip(r['text'], args.width)}")
        else:
            t0 = time.perf_counter()
            from .engine import content_terms
            from .provider import QUESTION_KIND, recall_options, select_for_injection
            opts = recall_options(cfg)
            # What the agent would get (questions skipped), then the skipped questions for reference.
            hits = engine.recall(args.query, k=max(args.k, int(cfg["recall_k"]) * 3), min_score=0.0, **opts)
            ms = (time.perf_counter() - t0) * 1000
            floor = [h for h in hits if h.score >= float(cfg["min_score"]) or h.assoc >= 0.6]
            chosen = {h.id for h in select_for_injection(floor, cfg)[:int(cfg["recall_k"])]}
            print(f'Query: "{args.query}"  ({ms:.0f} ms; floor {cfg["min_score"]}, band {cfg["score_band"]}; '
                  f'content words: {", ".join(content_terms(args.query)) or "none"})')
            print("   id  score  vector  words  linked  injected  kind            text")
            for h in hits[:args.k]:
                print(f"  {h.id:>3}  {h.score:>5.2f}  {h.as_query:>6.2f} {h.lexical:>6.2f}  {h.assoc:>6.2f}  "
                      f"{'yes' if h.id in chosen else 'no':<8}  {h.kind:<14}  {_clip(h.text, args.width)}")
            opts["skip_kinds"] = ()
            for h in [h for h in engine.recall(args.query, k=50, min_score=0.0, **opts) if h.kind == QUESTION_KIND][:3]:
                print(f"  {h.id:>3}  {h.score:>5.2f}  {h.as_query:>6.2f} {h.lexical:>6.2f}  {h.assoc:>6.2f}  {'never':<8}  "
                      f"{h.kind:<14}  {_clip(h.text, args.width)}")
            if not hits:
                print("  (nothing stored matches)")
    finally:
        engine.close()


def register_cli(subparser) -> None:
    subs = subparser.add_subparsers(dest="holonomic_action")
    subs.add_parser("stats", help="Size of the memory store")
    lst = subs.add_parser("list", help="Show recent memories")
    lst.add_argument("-n", type=int, default=20, help="How many (default 20)")
    lst.add_argument("--width", type=int, default=100, help="Characters of text to show")
    rec = subs.add_parser("recall", help="Run a recall and show the scores")
    rec.add_argument("query")
    rec.add_argument("-k", type=int, default=10, help="How many results (default 10)")
    rec.add_argument("--width", type=int, default=100, help="Characters of text to show")
    subparser.set_defaults(func=holonomic_command)
