"""`hermes holonomic ...` commands: look inside the memory store from a terminal.

    hermes holonomic stats
    hermes holonomic list [-n 20]
    hermes holonomic recall "what is my name" [-k 10]
    hermes holonomic reflect status | on [--model NAME] [--host URL] | off | now [--dry-run]
    hermes holonomic profile [--history]
    hermes holonomic profile --set user "Kayla is ..."      (who: user, self or us)

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
    if action not in ("stats", "list", "recall", "reflect", "profile"):
        print('Usage: hermes holonomic stats | list [-n N] | recall "query" [-k N] | '
              'reflect status|on|off|now | profile [--history]')
        return
    if action == "reflect" and args.reflect_action in ("on", "off"):
        _reflect_toggle(args)
        return
    engine, cfg = _open()
    if engine is None:
        return
    try:
        if action == "reflect":
            _reflect(engine, cfg, args)
        elif action == "profile":
            if args.set:
                who, text = args.set
                if who not in ("user", "self", "us") or len(text.strip()) < 3:
                    print('Usage: hermes holonomic profile --set user|self|us "text"')
                    return
                engine.set_profile(who, text)
                print(f"Profile '{who}' set. Earlier versions are kept (see --history). "
                      "Reflection will build on this text from now on.")
                return
            from hermes_constants import get_hermes_home
            from .reflect import read_foundation
            soul = read_foundation(get_hermes_home()).strip()
            print(f"Foundation (SOUL.md, written by you, never changed by this plugin): "
                  f"{str(len(soul)) + ' characters' if soul else 'not found'}")
            for who, title in (("user", "About the user"), ("self", "About herself"), ("us", "About the two of you")):
                history = engine.profile_history(who, 10 if args.history else 1)
                print(f"{title}:" if history else f"{title}: (none yet)")
                for entry in history:
                    print(f"  [{_when(entry['created_at'])}] {entry['text']}")
        elif action == "stats":
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


def _reflect_toggle(args) -> None:
    from hermes_constants import get_hermes_home
    from .provider import load_config, write_config
    home = get_hermes_home()
    values = {"reflect_enabled": args.reflect_action == "on"}
    if args.reflect_action == "on":
        if args.model:
            values["reflect_model"] = args.model
        if args.host:
            values["reflect_host"] = args.host.rstrip("/")
        if not (args.model or load_config(home).get("reflect_model")):
            print("Reflection needs a model. Run: hermes holonomic reflect on --model NAME   (a name from `ollama list`)")
            return
    write_config(home, values)
    cfg = load_config(home)
    print(f"Reflection is {'ON' if cfg['reflect_enabled'] else 'OFF'}"
          + (f" (model {cfg['reflect_model']})" if cfg["reflect_model"] else "")
          + ". Restart Hermes for a running session to pick this up.")


def _reflect(engine, cfg, args) -> None:
    from .provider import extract_keys
    from .reflect import ReflectionError, pending, reflect_config, reflect_once
    rc = reflect_config(cfg)
    if args.reflect_action == "now":
        try:
            t0 = time.perf_counter()
            from hermes_constants import get_hermes_home
            from .reflect import read_foundation
            report = reflect_once(engine, cfg, dry_run=args.dry_run, key_fn=extract_keys,
                                  foundation=read_foundation(get_hermes_home()))
        except ReflectionError as exc:
            print(f"Reflection failed: {exc}")
            return
        print(f"Read {report['read']} memories in {time.perf_counter() - t0:.0f} s"
              + (" (dry run: nothing stored)" if args.dry_run else ""))
        for kind, items in (report.get("proposed") or {}).items():
            for item in items:
                print(f"  {kind:<9} {item['text']}   <- {', '.join('#' + str(s) for s in item['sources'])}")
        for who, text in (report.get("profiles") or {}).items():
            print(f"  profile ({who}): {text or '(empty)'}")
        if not args.dry_run:
            print(f"Stored {len(report['stored'])} new, reinforced {len(report['reinforced'])} existing, "
                  f"profiles updated: {', '.join(report['profiles_updated']) or 'none'}")
        return
    last_run = engine.kv_get("reflect:last_run")
    print(f"  enabled: {rc['reflect_enabled']}")
    print(f"  model: {rc['reflect_model'] or '(not set)'} at {rc['reflect_host']}")
    print(f"  runs when: {rc['reflect_min_new']} new memories and {rc['reflect_idle_seconds']} s of quiet")
    print(f"  memories waiting: {pending(engine)}")
    print(f"  last run: {_when(float(last_run)) if last_run else 'never'}")


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
    ref = subs.add_parser("reflect", help="Reflection: turn on or off, check, or run once now")
    ref.add_argument("reflect_action", choices=["status", "on", "off", "now"], nargs="?", default="status")
    ref.add_argument("--model", help="Ollama model that does the reflecting (with 'on')")
    ref.add_argument("--host", help="Ollama server for that model, if different from the embedding server (with 'on')")
    ref.add_argument("--dry-run", action="store_true", help="With 'now': show what would be stored, store nothing")
    prof = subs.add_parser("profile", help="Show the profiles written by reflection")
    prof.add_argument("--history", action="store_true", help="Show earlier versions too")
    prof.add_argument("--set", nargs=2, metavar=("WHO", "TEXT"), help="Write a profile yourself: user, self or us")
    subparser.set_defaults(func=holonomic_command)
