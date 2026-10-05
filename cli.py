"""`hermes holonomic ...` commands: look inside the memory store from a terminal.

    hermes holonomic stats
    hermes holonomic list [-n 20]
    hermes holonomic recall "what is my name" [-k 10]
    hermes holonomic show 37 40                 full text of memories, and what each is linked to
    hermes holonomic forget 52 [--yes]          remove memories for good (shows them first; --yes to confirm)
    hermes holonomic reflect status | on [--model NAME] [--host URL] | off | now [--dry-run] [--depth N] [--think]
    hermes holonomic sleep status | on | off | now [--dry-run] [--only reflect,consolidate,fade,dream]
    hermes holonomic dreams [-n 5]
    hermes holonomic dreams images [off|words|pictures|from_images] [--api comfyui|a1111|openai] [--host URL] [--model NAME]
                                   [--size 768x512] [--count 3] [--people yes|no] [--strength 0.75] [--style "..."]
    hermes holonomic dreamtalk [--apply]          find stored conversation that is talk about a dream, and label it
    hermes holonomic relabel 63 64 --as said      correct a label: 'said' (ordinary conversation) or 'dream' (dream talk)
    hermes holonomic images                       image memory: status, and what is waiting to be described
    hermes holonomic images on [--model NAME] [--host URL] [--sections idle|now|off]  |  off
    hermes holonomic images list [-n 20] | show ID | find cat | labels
    hermes holonomic images add photo.jpg [--say "This is my cat"]     keep an image and describe it now
    hermes holonomic images process               describe everything that is waiting
    hermes holonomic images people ID yes|no      say whether an image has real people in it (dreams draw from those only if allowed)
    hermes holonomic images dream ID... yes|no|default     allow or forbid dream pictures drawn from particular images
    hermes holonomic images redo ID [--fix "That is a couch, not a lap"] [--say "new words for when it was shown"]
    hermes holonomic images look ID "what colour is the car?" [--section N]
    hermes holonomic images forget ID [--yes] [--keep-files]
    hermes holonomic tidy [--apply]               find stored messages that are Hermes' notes about attachments, and remove them
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
    if action not in ("stats", "list", "recall", "reflect", "profile", "show", "forget", "sleep", "dreams", "dreamtalk", "relabel", "images", "tidy"):
        print('Usage: hermes holonomic stats | list [-n N] | recall "query" [-k N] [--deep] | show ID... | forget ID... [--yes] | '
              'reflect status|on|off|now | sleep status|on|off|now | dreams | images | profile [--history]')
        return
    if action == "images" and args.images_action in ("on", "off"):
        _images_toggle(args)
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
        elif action == "images":
            _images_cmd(engine, cfg, args)
        elif action == "tidy":
            from .images import strip_image_markers
            from .sleep import RAW_KINDS
            found = []
            for s in engine.sessions(RAW_KINDS):
                noted = False                            # inside Hermes' footer about attached files
                for m in engine.session_memories(s["session"], kinds=("said_user", "asked_user"), limit=100000):
                    text = m["text"]
                    if text.startswith("--- Attached Context ---") or text.startswith("--- Context Warnings ---"):
                        noted = True
                    elif noted and not (text.startswith("📎") or text.startswith("It is available on disk at")
                                        or text.startswith("Use your tools to work with it")):
                        noted = False
                    if noted or not strip_image_markers(text):
                        found.append(m)
            print(f"{len(found)} stored message(s) are Hermes' notes about attachments, not something you said" + (":" if found else "."))
            for m in found:
                print(f"  [#{m['id']}] {_clip(m['text'], args.width)}")
                if args.apply:
                    engine.forget(m["id"])
            if found:
                print("Removed." if args.apply else "Nothing changed. Run again with --apply to remove them.")
        elif action == "relabel":
            to_dream = {"said_user": "dreamtalk_user", "said_assistant": "dreamtalk_assistant"}
            to_said = {v: k for k, v in to_dream.items()}
            table = to_dream if args.to == "dream" else to_said
            for mid in args.ids:
                mem = engine.get(mid)
                if mem is None:
                    print(f"[#{mid}] no such memory")
                elif mem["kind"] not in table:
                    print(f"[#{mid}] is '{mem['kind']}'; nothing to change")
                else:
                    engine.set_kind(mid, table[mem["kind"]])
                    print(f"[#{mid}] {mem['kind']} -> {table[mem['kind']]}   {_clip(mem['text'], 80)}")
        elif action == "dreamtalk":
            from .provider import asks_about_dreams, is_dream_talk, mark_dream_talk
            from .sleep import RAW_KINDS
            found = []
            for s in engine.sessions(RAW_KINDS):
                asked = False                           # was the user's last message a question about her dreams?
                for m in engine.session_memories(s["session"], kinds=RAW_KINDS, limit=5000):
                    if m["kind"] in ("said_user", "asked_user"):
                        asked = asks_about_dreams(engine, m["text"])
                    if m["kind"] == "said_assistant" and asked:
                        found.append(dict(m, force=True))
                    elif m["kind"] in ("said_user", "said_assistant") and is_dream_talk(
                            engine, m["id"], m["text"], "user" if m["kind"] == "said_user" else "assistant", cfg):
                        found.append(m)
            print(f"{len(found)} piece(s) of conversation look like talk about a dream"
                  + (":" if found else ". Nothing to label."))
            for m in found:
                print(f"  [#{m['id']}] {m['kind']:<14} {_clip(m['text'], args.width)}")
                if args.apply:
                    mark_dream_talk(engine, m["id"], m["text"], m["kind"], cfg, force=bool(m.get("force")))
            if found:
                print("Labelled as dream talk." if args.apply else "Nothing changed. Run again with --apply to label them.")
        elif action == "dreams" and args.dreams_action == "images":
            _dream_images(cfg, args)
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
                for p in d.get("pictures", []):
                    print(f"  Picture: {p['scene']}\n    {p['file']}")
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
                    mem = engine.get(mid) or {}
                    if mem.get("kind") == "image" and (mem.get("meta") or {}).get("image_id"):
                        from . import images as _img          # what an image showed: forget the image with it
                        _img.forget_image(engine, int(mem["meta"]["image_id"]))
                        print(f"  #{mid} described image #{mem['meta']['image_id']}, which is forgotten too. Its file is still on "
                              f"disk; `hermes holonomic images forget` removes files.")
                        continue
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
            from . import images as _img
            for key, value in dict(engine.stats(), images=_img.count_images(engine)).items():
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


def _dream_images(cfg, args) -> None:
    from hermes_constants import get_hermes_home
    from .paint import APIS
    from .provider import load_config, write_config
    from .sleep import DREAM_IMAGE_MODES, sleep_config
    home = get_hermes_home()
    values = {}
    if args.mode:
        if args.mode not in DREAM_IMAGE_MODES:
            print(f"Choose one of: {', '.join(DREAM_IMAGE_MODES)}")
            return
        values["dream_images"] = args.mode
    if args.api:
        values["dream_image_api"] = args.api
    if args.host:
        values["dream_image_host"] = args.host.rstrip("/")
    if args.model is not None:
        values["dream_image_model"] = args.model
    if args.count:
        values["dream_image_count"] = args.count
    if args.people:
        values["dream_image_use_people"] = args.people == "yes"
    if args.attempts:
        values["dream_image_candidates"] = max(1, min(args.attempts, 8))
    if args.swap:
        values["dream_image_swap"] = args.swap == "on"
    if args.think:
        values["dream_image_choose_think"] = args.think == "on"
    if args.encoder:
        values["dream_image_text_encoder_on"] = "cpu" if args.encoder == "cpu" else ""
    if args.strength is not None:
        values["dream_image_strength"] = min(max(args.strength, 0.1), 1.0)
    if args.style is not None:
        values["dream_image_style"] = args.style
    if args.size:
        try:
            w, h = (int(v) for v in args.size.lower().split("x"))
            values.update(dream_image_width=w, dream_image_height=h)
        except ValueError:
            print("Give the size as WIDTHxHEIGHT, for example 768x512")
            return
    if values:
        write_config(home, values)
        cfg = load_config(home)
    sc = sleep_config(cfg)
    if args.test:
        from .paint import PaintError, make_painter
        try:
            t0 = time.time()
            data = make_painter(sc)(args.test + (", " + sc["dream_image_style"] if sc["dream_image_style"] else ""), None)
        except PaintError as exc:
            print(f"The image generator did not produce a picture: {exc}")
            return
        out = home / "holonomic-test-picture.png"
        try:
            out.write_bytes(data)
        except OSError:              # Windows refuses to overwrite a picture that a viewer still has open
            out = home / f"holonomic-test-picture-{time.strftime('%H%M%S')}.png"
            out.write_bytes(data)
        print(f"Drew a picture in {time.time() - t0:.0f} s ({len(data) // 1024} KB). Nothing was stored in memory. Open it to look:\n  {out}")
        return
    mode = sc["dream_images"]
    print(f"Images in dreams: {mode}")
    print({"off": "  Images play no part in dreams.",
           "words": "  What she saw in recent images, and in older images they resemble, joins what a dream is made from.",
           "pictures": "  As 'words', and scenes from each dream are drawn from her description of them.",
           "from_images": "  As 'pictures', and a scene that resembles images she has seen is drawn starting from those images."}[mode])
    if mode in ("pictures", "from_images"):
        ready = sc["dream_image_api"] in APIS and sc["dream_image_host"]
        print(f"  image generator: {sc['dream_image_api'] + ' at ' + sc['dream_image_host'] if ready else 'NOT SET (use --api and --host)'}"
              + (f", model {sc['dream_image_model']}" if sc["dream_image_model"] else ""))
        print(f"  {sc['dream_image_count']} picture(s) per dream, {sc['dream_image_width']}x{sc['dream_image_height']}, "
              f"each drawn {sc['dream_image_candidates']} time(s)" + (" and she keeps the best" if int(sc["dream_image_candidates"]) > 1 else "")
              + (", reasoning before she chooses (--think off to stop)" if sc["dream_image_choose_think"] and int(sc["dream_image_candidates"]) > 1 else ""))
        if sc["dream_image_swap"]:
            print("  one graphics card: the language models are unloaded while pictures are drawn and reloaded afterwards")
    if mode == "from_images":
        print(f"  images with real people in them: {'may be drawn from' if sc['dream_image_use_people'] else 'are not drawn from (--people yes to allow)'}")
        print(f"  how far a picture may move from the images it starts from: {sc['dream_image_strength']} (--strength)")
    if mode in ("pictures", "from_images"):
        print(f"  style added to every scene: {sc['dream_image_style'] or '(none)'} (--style)")
    if not cfg.get("image_enabled"):
        print("  Image memory is off, so there are no images to dream of. Turn on with: hermes holonomic images on")
    if values:
        print("Restart Hermes for a running session to pick this up.")


def _images_toggle(args) -> None:
    from hermes_constants import get_hermes_home
    from .images import image_config
    from .provider import load_config, write_config
    home = get_hermes_home()
    values = {"image_enabled": args.images_action == "on"}
    if args.images_action == "on":
        if args.model:
            values["image_model"] = args.model
        if args.host:
            values["image_host"] = args.host.rstrip("/")
        if args.sections:
            values.update({"image_sections": args.sections != "off"},
                          **({"image_sections_when": args.sections} if args.sections != "off" else {}))
        if not image_config(dict(load_config(home), **values))["image_model"]:
            print("Image memory needs a model that can see. Run: hermes holonomic images on --model NAME   (a name from `ollama list`)")
            return
    write_config(home, values)
    ic = image_config(load_config(home))
    print(f"Image memory is {'ON' if ic['image_enabled'] else 'OFF'}"
          + (f" (model {ic['image_model']} at {ic['image_host']}; parts of each image: "
             f"{'not looked at' if not ic['image_sections'] else 'looked at ' + ('straight away' if ic['image_sections_when'] == 'now' else 'when the conversation is quiet')})"
             if ic["image_enabled"] else "")
          + ". Restart Hermes for a running session to pick this up.")


def _print_image(img, width: int = 110, full: bool = False) -> None:
    print(f"image #{img['id']}  {_when(img['created_at'])}  {img['width']}x{img['height']}  {img['bytes'] // 1024} KB"
          + (f"  shown {img['seen']} times" if img["seen"] > 1 else "") + (f"  ({img['origin']})" if img["origin"] else ""))
    print(f"    file: {img['file']}")
    if full and img["original"] != img["file"]:
        print(f"    original: {img['original']}")
    if img["caption"]:
        print(f"    said when shown: {img['caption'] if full else _clip(img['caption'], width)}")
    described = img["description"] or "(not described yet)"
    print(f"    [#{img['memory_id']}] {described if full else _clip(described, width)}" if img["memory_id"] else f"    {described}")
    if img["labels"]:
        print(f"    things in it: {', '.join(img['labels'])}")
    if full:
        print(f"    real people in it: {'yes' if img.get('people') else 'no'}   (wrong? hermes holonomic images people {img['id']} yes|no)")
        said = img.get("dream_from")
        print(f"    dream pictures drawn from it: {'always allowed' if said else 'never' if said is False else 'by the general rule'}"
              f"   (hermes holonomic images dream {img['id']} yes|no|default)")
    if img["sections_waiting"]:
        print(f"    {img['sections_waiting']} of {img['sections_total']} parts not looked at yet")
    for s in img.get("sections", []):
        if s["description"]:
            print(f"    part {s['section']} ({s['place']}) [#{s['memory_id']}]: {s['description']}")
        elif full and s["looked_at"]:
            print(f"    part {s['section']} ({s['place']}): nothing notable")


def _images_cmd(engine, cfg, args) -> None:
    from pathlib import Path
    from . import images as _img
    from .provider import extract_keys
    from .reflect import ReflectionError
    ic = _img.image_config(cfg)
    what, items = args.images_action, list(args.items or [])
    if what == "status":
        waiting = _img.pending(engine)
        print(f"Image memory is {'ON' if ic['image_enabled'] else 'OFF'}.  Model: {ic['image_model'] or '(none set)'} at {ic['image_host']}")
        print(f"  parts of each image: "
              + ("not looked at" if not ic["image_sections"] else f"{ic['image_grid']}x{ic['image_grid']} overlapping, looked at "
                 + ("straight away" if ic["image_sections_when"] == "now" else f"after {ic['image_idle_seconds']} s of quiet")))
        print(f"  images kept: {_img.count_images(engine)}   waiting for a description: {waiting['images']}   "
              f"parts waiting to be looked at: {waiting['sections']}")
        print(f"  files are in: {engine.path / 'images'}")
        if not ic["image_enabled"]:
            print("  Turn on with: hermes holonomic images on --model NAME [--host URL]")
    elif what == "list":
        found = _img.list_images(engine, args.n)
        print(f"{len(found)} most recent image(s), newest first:" if found else "No images yet.")
        for img in found:
            _print_image(img, args.width)
    elif what == "show":
        for raw in items:
            img = _img.get_image(engine, int(raw), sections=True) if raw.isdigit() else None
            _print_image(img, full=True) if img else print(f"image #{raw}: no such image")
    elif what == "labels":
        found = _img.all_labels(engine, 200)
        print("Things noticed in images (and in how many):" if found else "Nothing noticed yet.")
        print("  " + ",  ".join(f"{label} ({n})" for label, n in found)) if found else None
    elif what == "find":
        if not items:
            print("Usage: hermes holonomic images find LABEL      (see `hermes holonomic images labels`)")
            return
        found = _img.find_by_label(engine, " ".join(items))
        print(f"{len(found)} image(s) in which '{' '.join(items)}' was noticed:" if found else f"No image in which '{' '.join(items)}' was noticed.")
        for img in found:
            _print_image(img, args.width)
            print(f"    matched: {', '.join(img['matched'])}" + (f"   noticed in the: {', '.join(img['matched_in'])}" if img["matched_in"] else ""))
    elif what == "add":
        if not items:
            print('Usage: hermes holonomic images add FILE... [--say "what you would say when showing it"]')
            return
        for raw in items:
            try:
                img = _img.add_image(engine, Path(raw).expanduser().read_bytes(), cfg, origin=raw, caption=args.say or "",
                                     session="shown-from-terminal", source="user")
            except (OSError, _img.ImageError) as exc:
                print(f"{raw}: {exc}")
                continue
            if not img["new"]:
                print(f"{raw}: she has seen this exact image before.")
                _print_image(img, args.width)
                continue
            report = {"calls": []}
            try:
                t0 = time.time()
                img = _img.describe(engine, cfg, img["id"], key_fn=extract_keys, report=report) or img
                print(f"{raw}: kept and described in {time.time() - t0:.0f} s.")
            except (_img.ImageError, ReflectionError) as exc:
                print(f"{raw}: kept as image #{img['id']}, but not described: {exc}")
            _print_image(img, full=True)
        waiting = _img.pending(engine)
        if waiting["sections"]:
            print(f"\n{waiting['sections']} part(s) waiting to be looked at. Run: hermes holonomic images process")
    elif what == "process":
        waiting = _img.pending(engine)
        if not (waiting["images"] or waiting["sections"]):
            unclear = _img.process(engine, cfg, key_fn=extract_keys).get("unclear", 0)      # nothing to look at; writing may need checking
            print("Nothing is waiting." + (f" {unclear} piece(s) of writing reported by only one look are now marked as not read for certain."
                                           if unclear else ""))
            return
        print(f"Describing {waiting['images']} image(s) and looking at {waiting['sections']} part(s) with {ic['image_model'] or '(no model)'} "
              f"at {ic['image_host']} ...")
        t0 = time.time()
        report = _img.process(engine, cfg, key_fn=extract_keys)
        _print_calls(report)
        print(f"Described {len(report['described'])} image(s), looked at {report['sections']} part(s) in {time.time() - t0:.0f} s.")
        if report.get("unclear"):
            print(f"  {report['unclear']} piece(s) of writing were reported by only one look and are now marked as not read for certain.")
        for err in report["errors"]:
            print(f"  stopped: {err}")
        left = _img.pending(engine)
        if left["images"] or left["sections"]:
            print(f"  still waiting: {left['images']} image(s), {left['sections']} part(s)")
    elif what == "dream":
        if len(items) < 2 or items[-1] not in ("yes", "no", "default") or not all(raw.isdigit() for raw in items[:-1]):
            print("Usage: hermes holonomic images dream ID... yes|no|default      (may dream pictures be drawn from these images?)")
            return
        for raw in items[:-1]:
            if not _img.set_dream_use(engine, int(raw), {"yes": True, "no": False, "default": None}[items[-1]]):
                print(f"image #{raw}: no such image")
                continue
            print(f"image #{raw}: " + {"yes": "dream pictures may be drawn from it, whatever the general rule about people.",
                                       "no": "dream pictures are never drawn from it. What she saw in it can still appear in a dream's words.",
                                       "default": "back to the general rule (an image with people in it is used only if those are allowed)."}[items[-1]])
    elif what == "people":
        if len(items) != 2 or not items[0].isdigit() or items[1] not in ("yes", "no"):
            print("Usage: hermes holonomic images people ID yes|no      (does the image have real people in it?)")
            return
        if not _img.set_people(engine, int(items[0]), items[1] == "yes"):
            print(f"image #{items[0]}: no such image")
            return
        print(f"image #{items[0]}: real people in it: {items[1]}. "
              + ("Dreams will not draw from it unless images of people are allowed." if items[1] == "yes" else "Dreams may draw from it."))
    elif what == "redo":
        if not items or not all(raw.isdigit() for raw in items):
            print('Usage: hermes holonomic images redo ID... [--fix "what the description got wrong"] [--say "what was said when it was shown"]')
            return
        for raw in items:
            try:
                t0 = time.time()
                img = _img.redescribe(engine, cfg, int(raw), correction=args.fix or "", caption=args.say, key_fn=extract_keys)
            except (_img.ImageError, ReflectionError, OSError) as exc:
                print(f"image #{raw}: could not be described again: {exc}")
                continue
            if not img:
                print(f"image #{raw}: no such image")
                continue
            print(f"image #{raw}: described again in {time.time() - t0:.0f} s.")
            _print_image(img, full=True)
        waiting = _img.pending(engine)
        if waiting["sections"]:
            print(f"\n{waiting['sections']} part(s) waiting to be looked at again. Run: hermes holonomic images process")
    elif what == "look":
        if len(items) < 2 or not items[0].isdigit():
            print('Usage: hermes holonomic images look ID "question" [--section N]')
            return
        try:
            print(_img.look(engine, cfg, int(items[0]), " ".join(items[1:]), section=args.section))
        except (_img.ImageError, ReflectionError) as exc:
            print(f"Could not look: {exc}")
    elif what == "forget":
        found = [_img.get_image(engine, int(raw)) for raw in items if raw.isdigit()]
        found = [img for img in found if img]
        if not found:
            print("No such image. Usage: hermes holonomic images forget ID... [--yes] [--keep-files]")
            return
        for img in found:
            _print_image(img, args.width)
        if not args.yes:
            print(f"Nothing removed. To forget {'this image' if len(found) == 1 else 'these images'} for good"
                  f"{'' if args.keep_files else ', and delete the files'}, run the same command with --yes.")
            return
        for img in found:
            _img.forget_image(engine, img["id"], delete_files=not args.keep_files)
        print(f"Forgot image {', '.join('#' + str(i['id']) for i in found)}"
              + (" (files kept on disk)." if args.keep_files else " and deleted the files. This cannot be undone."))


def _print_calls(report) -> None:
    for c in report["calls"]:
        print(f"  step {c['step']}{' (thinking)' if c.get('think') else ''}: {c.get('seconds', 0):.0f} s, prompt {c.get('prompt_tokens')} tokens, "
              f"reply {c.get('reply_tokens')} tokens, finished: {c.get('done_reason')}" + (f"  [{c['failed']}]" if c.get("failed") else ""))
        if c.get("step") == "choose" and c.get("thinking"):          # what she reasoned before choosing, to judge whether it helps
            print(f"      her reasoning: {c['thinking'][:1200]}{'...' if len(c['thinking']) > 1200 else ''}")


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
        seen = sorted({f["image_id"] for f in d["fragments"] if f.get("image_id")})
        if seen:
            print(f"  IMAGES    drew on what she saw in image {', '.join('#' + str(i) for i in seen)}")
        for p in d.get("pictures") or []:
            print(f"  PICTURE   {p['scene']}" + (f"   <- from image {', '.join('#' + str(i) for i in p['from'])}" if p.get("from") else "")
                  + (f"\n            she chose attempt {p['chosen']} of {p['of']}" + (f": {p['why']}" if p.get("why") else "") if p.get("of") else "")
                  + (f"\n            {p['file']}" if p.get("file") else ""))
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
    td = subs.add_parser("tidy", help="Find stored messages that are Hermes' notes about attachments, and remove them")
    td.add_argument("--apply", action="store_true", help="Remove what is found (without this, it is only listed)")
    td.add_argument("--width", type=int, default=110, help="Characters of text to show")
    rl = subs.add_parser("relabel", help="Correct whether pieces of conversation are labelled as dream talk")
    rl.add_argument("ids", type=int, nargs="+", help="Memory ids, as shown in brackets")
    rl.add_argument("--as", dest="to", choices=["said", "dream"], required=True,
                    help="'said' = ordinary conversation, 'dream' = talk about a dream")
    dt = subs.add_parser("dreamtalk", help="Find stored conversation that is talk about a dream, and label it")
    dt.add_argument("--apply", action="store_true", help="Label what is found (without this, it is only listed)")
    dt.add_argument("--width", type=int, default=110, help="Characters of text to show")
    drm = subs.add_parser("dreams", help="Show recent dreams, or set what images do in dreams")
    drm.add_argument("dreams_action", nargs="?", choices=["show", "images"], default="show")
    drm.add_argument("mode", nargs="?", help="With 'images': off, words, pictures or from_images")
    drm.add_argument("-n", type=int, default=5, help="How many (default 5)")
    drm.add_argument("--api", choices=["comfyui", "a1111", "openai"], help="With 'images': which interface the image generator speaks")
    drm.add_argument("--host", help="With 'images': address of the image generator, e.g. http://10.0.0.21:8188")
    drm.add_argument("--test", metavar="WORDS", help="With 'images': draw one picture from these words now, to check the image generator")
    drm.add_argument("--model", help="With 'images': model or checkpoint name, if the server needs one")
    drm.add_argument("--size", help="With 'images': picture size, e.g. 768x512")
    drm.add_argument("--count", type=int, help="With 'images': pictures per dream")
    drm.add_argument("--people", choices=["yes", "no"], help="With 'images': may images with real people in them be drawn from")
    drm.add_argument("--attempts", type=int, help="With 'images': how many times each picture is drawn; she keeps the one she thinks best")
    drm.add_argument("--think", choices=["on", "off"], help="With 'images': let her reason before choosing between attempts")
    drm.add_argument("--swap", choices=["on", "off"], help="With 'images': for one graphics card: unload the language models while "
                                                           "pictures are drawn and reload them when the dream is over")
    drm.add_argument("--encoder", choices=["gpu", "cpu"], help="With 'images': Z-Image only: where the prompt is read. 'cpu' leaves "
                                                               "the graphics card to the drawing model")
    drm.add_argument("--strength", type=float, help="With 'images': from_images: how far a picture may move from the images it "
                                                     "starts from, 0.1 (barely) to 1.0 (entirely)")
    drm.add_argument("--style", help="With 'images': words added to every scene, e.g. \"dreamlike, soft light\"")
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
    im = subs.add_parser("images", help="Image memory: what she has been shown")
    im.add_argument("images_action", nargs="?", default="status",
                    choices=["status", "on", "off", "list", "show", "find", "labels", "add", "process", "look", "forget", "redo", "people", "dream"])
    im.add_argument("items", nargs="*", help="Image ids, files to add, a label to find, or an id and a question")
    im.add_argument("--model", help="Ollama model that can see (with 'on'); default: the reflection model")
    im.add_argument("--host", help="Ollama server for that model, if it is not the reflection server (with 'on')")
    im.add_argument("--sections", choices=["idle", "now", "off"],
                    help="With 'on': look at the parts of each image when the conversation is quiet (default), straight away, or not at all")
    im.add_argument("--say", help="With 'add': what you would say when showing the image. With 'redo': replaces what was said")
    im.add_argument("--fix", help="With 'redo': something the description got wrong, in your words; it is taken as true")
    im.add_argument("--section", type=int, help="With 'look': a part number, to look closely at one part")
    im.add_argument("-n", type=int, default=20, help="With 'list': how many (default 20)")
    im.add_argument("--width", type=int, default=110, help="Characters of text to show")
    im.add_argument("--yes", action="store_true", help="With 'forget': actually remove")
    im.add_argument("--keep-files", action="store_true", help="With 'forget': leave the image files on disk")
    prof = subs.add_parser("profile", help="Show the profiles written by reflection")
    prof.add_argument("--history", action="store_true", help="Show earlier versions too")
    prof.add_argument("--set", nargs=2, metavar=("WHO", "TEXT"), help="Write a profile yourself: user, self or us")
    subparser.set_defaults(func=holonomic_command)
