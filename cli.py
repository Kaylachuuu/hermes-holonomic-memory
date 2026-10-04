"""`hermes holonomic ...` commands: look inside the memory store from a terminal.

    hermes holonomic stats
    hermes holonomic list [-n 20]
    hermes holonomic recall "what is my name" [-k 10]
    hermes holonomic show 37 40                 full text of memories, and what each is linked to
    hermes holonomic forget 52 [--yes]          remove memories for good (shows them first; --yes to confirm)
    hermes holonomic reflect status | on [--model NAME] [--host URL] | off | now [--dry-run] [--depth N] [--think]
    hermes holonomic sleep status | on | off | now [--dry-run] [--only reflect,consolidate,fade,dream]
    hermes holonomic dreams [-n 5]
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
    if action not in ("stats", "list", "recall", "reflect", "profile", "show", "forget", "sleep", "dreams"):
        print('Usage: hermes holonomic stats | list [-n N] | recall "query" [-k N] [--deep] | show ID... | forget ID... [--yes] | '
              'reflect status|on|off|now | sleep status|on|off|now | dreams | profile [--history]')
        return
    if action == "reflect" and args.reflect_action in ("on", "off"):
        _reflect_toggle(args)
        return
    if action == "sleep" and args.sleep_action in ("on", "off"):
        from hermes_constants import get_hermes_home
        from .provider import load_config, write_config
        home = get_hermes_home()
        if args.sleep_action == "on" and not load_config(home).get("reflect_model"):
            print("Sleep uses the reflection model. Set one first: hermes holonomic reflect on --model NAME")
            return
        write_config(home, {"sleep_enabled": args.sleep_action == "on"})
        print(f"Unattended sleep is {'ON' if args.sleep_action == 'on' else 'OFF'}. Restart Hermes for a running session to pick this up.")
        return
    engine, cfg = _open()
    if engine is None:
        return
    try:
        if action == "sleep":
            _sleep(engine, cfg, args)
        elif action == "dreams":
            from .sleep import dreams
            found = dreams(engine, args.n)
            print(f"{len(found)} most recent dream(s), newest first:" if found else "No dreams yet.")
            for d in found:
                print(f"\n[#{d['id']}] {_when(d['created_at'])}\n  {d['text']}")
                if d.get("thoughts"):
                    print(f"  What she made of it: {d['thoughts']}")
                for c in d.get("connections", []):
                    print(f"  Connection she noticed: {c}")
        elif action in ("show", "forget"):
            found = [(mid, engine.get(mid)) for mid in args.ids]
            for mid, mem in found:
                if mem is None:
                    print(f"[#{mid}] no such memory (or already forgotten)")
                    continue
                meta = mem.get("meta") or {}
                print(f"[#{mid}] {mem['kind']}, {_when(mem['created_at'])}, trust {mem['trust']:.2f}, strength {mem['strength']:.2f}"
                      + (f", SUPERSEDED by #{meta['superseded_by']}" if meta.get("superseded_by") else ""))
                print(f"    {mem['text']}")
                if meta.get("sources"):
                    print(f"    drawn from: {', '.join('#' + str(s) for s in meta['sources'])}")
                if action == "show":
                    for h in engine.associates(mid, k=6):
                        print(f"    linked to [#{h.id}] ({h.kind}, {h.assoc:.2f}) {_clip(h.text, 90)}")
            if action == "forget":
                real = [mid for mid, mem in found if mem is not None]
                if not real:
                    return
                if not args.yes:
                    print(f"Nothing removed. To remove {'this memory' if len(real) == 1 else 'these ' + str(len(real)) + ' memories'} "
                          f"for good, run the same command with --yes.")
                    return
                for mid in real:
                    engine.forget(mid)
                print(f"Forgot {', '.join('#' + str(m) for m in real)}. This cannot be undone. "
                      "Profiles are not changed by this; see `hermes holonomic profile`.")
        elif action == "reflect":
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
            if args.deep:
                opts.update(min_strength=0.0, reach=2)
            hits = engine.recall(args.query, k=max(args.k, int(cfg["recall_k"]) * 3), min_score=0.0, **opts)
            ms = (time.perf_counter() - t0) * 1000
            floor = [h for h in hits if h.score >= float(cfg["min_score"]) or h.assoc >= 0.6]
            chosen = {h.id for h in select_for_injection(floor, cfg)[:int(cfg["recall_k"])]}
            print(f'Query: "{args.query}"  ({ms:.0f} ms; floor {cfg["min_score"]}, band {cfg["score_band"]}; '
                  f'content words: {", ".join(content_terms(args.query)) or "none"})')
            print("   id  score  vector  words  linked  strength  injected  kind            text")
            for h in hits[:args.k]:
                faded = h.strength < float(cfg["fade_threshold"])
                print(f"  {h.id:>3}  {h.score:>5.2f}  {h.as_query:>6.2f} {h.lexical:>6.2f}  {h.assoc:>6.2f}  {h.strength:>8.2f}  "
                      f"{'faded' if faded else 'yes' if h.id in chosen else 'no':<8}  {h.kind:<14}  {_clip(h.text, args.width)}")
            opts["skip_kinds"] = ()
            for h in [h for h in engine.recall(args.query, k=50, min_score=0.0, **opts) if h.kind == QUESTION_KIND][:3]:
                print(f"  {h.id:>3}  {h.score:>5.2f}  {h.as_query:>6.2f} {h.lexical:>6.2f}  {h.assoc:>6.2f}  {h.strength:>8.2f}  {'never':<8}  "
                      f"{h.kind:<14}  {_clip(h.text, args.width)}")
            if not hits:
                print("  (nothing stored matches)")
    finally:
        engine.close()


def _print_calls(report) -> None:
    for c in report["calls"]:
        print(f"  step {c['step']}: {c.get('seconds', 0):.0f} s, prompt {c.get('prompt_tokens')} tokens, "
              f"reply {c.get('reply_tokens')} tokens, finished: {c.get('done_reason')}" + (f"  [{c['failed']}]" if c.get("failed") else ""))


def _sleep(engine, cfg, args) -> None:
    from hermes_constants import get_hermes_home
    from .provider import extract_keys
    from .reflect import ReflectionError, pending, read_foundation
    from .sleep import sleep_config, sleep_once, unfinished_business
    sc = sleep_config(cfg)
    if args.sleep_action != "now":
        last = engine.kv_get("sleep:last_run")
        print(f"  unattended sleep: {'on' if sc['sleep_enabled'] else 'off'} "
              f"(after {sc['sleep_idle_seconds'] // 60} quiet minutes, at most every {sc['sleep_min_hours']} hours)")
        print(f"  steps: reflect on, consolidate {'on' if sc['consolidate_enabled'] else 'off'}, "
              f"fade {'on' if sc['fade_enabled'] else 'off'} (half-life {sc['fade_half_life_days']} days), "
              f"dream {'on' if sc['dream_enabled'] else 'off'}")
        print(f"  dream model: {sc['dream_model'] or cfg.get('reflect_model') or '(not set)'}")
        print(f"  waiting: {pending(engine)} memories to reflect on, {len(unfinished_business(engine, cfg))} conversation(s) to summarise")
        print(f"  last sleep: {_when(float(last)) if last else 'never'}")
        return
    steps = [s.strip() for s in args.only.split(",")] if args.only else None
    t0 = time.perf_counter()
    try:
        report = sleep_once(engine, cfg, dry_run=args.dry_run, key_fn=extract_keys,
                            foundation=read_foundation(get_hermes_home()), steps=steps)
    except ReflectionError as exc:
        print(f"Sleep failed: {exc}")
        return
    print(f"Slept for {time.perf_counter() - t0:.0f} s" + (" (dry run: nothing stored)" if args.dry_run else ""))
    _print_calls(report)
    for r in report["reflections"]:
        n = sum(len(v) for v in (r.get("proposed") or {}).values())
        print(f"  REFLECT   read {r['read']} memories, {n} item(s)" + ("" if args.dry_run else f", stored {len(r['stored'])}"))
    for e in report["episodes"]:
        print(f"  EPISODE   ({e['memories']} memories, {_when(e['when'])}) {e['summary']}")
    if report["fade"]:
        f = report["fade"]
        print(f"  FADE      {f['eligible']} summarised memories x {f['factor']}" + ("" if args.dry_run else f", {f['changed']} lowered"))
    d = report["dream"]
    if d and d.get("skipped"):
        print(f"  DREAM     skipped: {d['skipped']}")
    for n, d in enumerate(report.get("dreams") or [], 1):
        faded = sum(1 for f in d["fragments"] if f["faded"])
        print(f"  DREAM {n}   from {len(d['fragments'])} fragments ({sum(1 for f in d['fragments'] if f['age'] == 'OLDER')} older, {faded} faded)")
        print(f"            {d['text']}")
        print(f"  ON WAKING {d['thoughts'] or '(nothing)'}")
        for c in d["connections"]:
            print(f"  CONNECTION {c['text']}   <- {', '.join('#' + str(s) for s in c['sources'])}")
    for err in report["errors"]:
        print(f"  ERROR     {err}")


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
        if args.depth:
            values["reflect_depth"] = args.depth
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
        if args.think or args.no_think:
            cfg = dict(cfg, reflect_think=bool(args.think))
        if args.depth:
            cfg = dict(cfg, reflect_depth=args.depth)
        try:
            t0 = time.perf_counter()
            from hermes_constants import get_hermes_home
            from .reflect import read_foundation
            report = reflect_once(engine, cfg, dry_run=args.dry_run, key_fn=extract_keys,
                                  foundation=read_foundation(get_hermes_home()))
        except ReflectionError as exc:
            from .reflect import LAST_CALL
            print(f"Reflection failed: {exc}")
            if LAST_CALL:
                print(f"  last model call: {LAST_CALL['seconds']:.0f} s, prompt {LAST_CALL['prompt_tokens']} tokens, "
                      f"reply {LAST_CALL['reply_tokens']} tokens, finished: {LAST_CALL['done_reason']}")
            return
        print(f"Read {report['read']} memories in {time.perf_counter() - t0:.0f} s at depth {report['depth']}"
              + (" (dry run: nothing stored)" if args.dry_run else ""))
        for c in report["calls"]:
            print(f"  step {c['step']}: {c.get('seconds', 0):.0f} s, prompt {c.get('prompt_tokens')} tokens, "
                  f"reply {c.get('reply_tokens')} tokens, thinking {'on' if c.get('think') else 'off'}, "
                  f"finished: {c.get('done_reason')}" + (f"  [{c['failed']}]" if c.get("failed") else ""))
        for c in report.get("checked") or []:
            if c["verdict"] == "drop":
                print(f"  CHECK dropped ({c['kind']}): {c['was']}")
            elif c["verdict"] == "unquote":
                print(f"  UNQUOTED ({c['kind']}), not the user's own words: {c['was']}\n             ->  {c['now']}")
            else:
                print(f"  CHECK rewrote ({c['kind']}): {c['was']}\n             ->  {c['now']}")
        for kind, items in (report.get("proposed") or {}).items():
            for item in items:
                print(f"  {kind:<9} {item['text']}   <- {', '.join('#' + str(s) for s in item['sources'])}")
        for text in report.get("cut_off") or []:
            print(f"  cut off   {text}   (stopped mid-sentence; not stored)")
        for text in report.get("dropped_assistant_only") or []:
            print(f"  dropped   {text}   (rested only on the assistant's own words)")
        for item in report.get("superseded") or []:
            print(f"  REPLACES  [#{item['fact']}] {item['was']}\n        ->  {item['replacement']}   <- "
                  f"{', '.join('#' + str(s) for s in item['sources'])}")
        for who, text in (report.get("profiles") or {}).items():
            print(f"  profile ({who}): {text or '(empty)'}")
        if not args.dry_run:
            print(f"Stored {len(report['stored'])} new, reinforced {len(report['reinforced'])} existing, "
                  f"profiles updated: {', '.join(report['profiles_updated']) or 'none'}")
        return
    last_run = engine.kv_get("reflect:last_run")
    print(f"  enabled: {rc['reflect_enabled']}")
    print(f"  model: {rc['reflect_model'] or '(not set)'} at {rc['reflect_host']}")
    print(f"  depth: {rc['reflect_depth']} (1 facts only, 2 everything unchecked, 3 everything checked); "
          f"thinking {'on' if rc['reflect_think'] else 'off'}")
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
    rec.add_argument("--deep", action="store_true", help="Include faded memories and follow links further")
    slp = subs.add_parser("sleep", help="The long-idle cycle: reflect, consolidate, fade, dream")
    slp.add_argument("sleep_action", choices=["status", "on", "off", "now"], nargs="?", default="status")
    slp.add_argument("--dry-run", action="store_true", help="With 'now': show what would happen, store nothing")
    slp.add_argument("--only", help="With 'now': comma-separated steps to run (reflect,consolidate,fade,dream)")
    drm = subs.add_parser("dreams", help="Show recent dreams")
    drm.add_argument("-n", type=int, default=5, help="How many (default 5)")
    show = subs.add_parser("show", help="Full text of memories and what each is linked to")
    show.add_argument("ids", type=int, nargs="+", help="Memory ids, as shown in brackets")
    fg = subs.add_parser("forget", help="Remove memories for good")
    fg.add_argument("ids", type=int, nargs="+", help="Memory ids, as shown in brackets")
    fg.add_argument("--yes", action="store_true", help="Actually remove them (without this, they are only shown)")
    ref = subs.add_parser("reflect", help="Reflection: turn on or off, check, or run once now")
    ref.add_argument("reflect_action", choices=["status", "on", "off", "now"], nargs="?", default="status")
    ref.add_argument("--model", help="Ollama model that does the reflecting (with 'on')")
    ref.add_argument("--host", help="Ollama server for that model, if different from the embedding server (with 'on')")
    ref.add_argument("--depth", type=int, choices=[1, 2, 3],
                     help="1 facts and user profile only; 2 everything, unchecked; 3 everything, each item checked (default)")
    ref.add_argument("--dry-run", action="store_true", help="With 'now': show what would be stored, store nothing")
    ref.add_argument("--think", action="store_true", help="With 'now': let the model reason first (slow; can run away on small models)")
    ref.add_argument("--no-think", action="store_true", help="With 'now': answer without reasoning first (the default)")
    prof = subs.add_parser("profile", help="Show the profiles written by reflection")
    prof.add_argument("--history", action="store_true", help="Show earlier versions too")
    prof.add_argument("--set", nargs=2, metavar=("WHO", "TEXT"), help="Write a profile yourself: user, self or us")
    subparser.set_defaults(func=holonomic_command)
