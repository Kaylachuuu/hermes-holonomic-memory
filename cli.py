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
    hermes holonomic images signature ID none|bottom-right|...   say where an image is signed or watermarked (kept out of dream pictures)
    hermes holonomic images fingerprints [on|off|redo] [--host URL]   recognise pictures from the pictures themselves (needs the helper server)
    hermes holonomic images similar ID [--all]    images that look like this one
    hermes holonomic images name ID NAME [--what "..."] [--not]   this image shows (or does not show) a particular named thing
    hermes holonomic images names [forget NAME]   the things she knows by name and where she has seen them
    hermes holonomic images dream ID... yes|no|default     allow or forbid dream pictures drawn from particular images
    hermes holonomic images redo ID [--fix "That is a couch, not a lap"] [--say "new words for when it was shown"]
    hermes holonomic images look ID "what colour is the car?" [--section N]
    hermes holonomic images forget ID...          set an image aside: she no longer recalls it; nothing is deleted
    hermes holonomic images removed               images set aside
    hermes holonomic images restore ID...         put one back exactly as it was
    hermes holonomic images delete ID... [--yes]  remove for good an image already set aside, files included
    hermes holonomic faces                        knowing people by their faces: what is allowed, who she knows
    hermes holonomic faces learn none|me|named|often     whose faces she may learn (none until you say otherwise)
    hermes holonomic faces ask on|off             may she ask who someone is who keeps appearing (with 'often')
    hermes holonomic faces unasked on|off         is she told who is in an image without being asked
    hermes holonomic faces image ID off|on        do not look for faces in one image (a street full of strangers)
    hermes holonomic faces show ID | name ID NAME [--face N] [--me] | not ID NAME | people | often | dream NAME yes|no | forget NAME | scan
    hermes holonomic context                      what memory gave her for the most recent message
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


def _library_cmd(engine, cfg, args) -> None:
    from . import library as lib
    root, what, items = lib.root_of(engine), args.library_action, list(args.items or [])

    def show_report(r: dict) -> None:
        print(f"Library '{r['name']}': {r['added']} file(s) added, {r['changed']} changed, {r['removed']} removed, "
              f"{r['unchanged']} unchanged; {r['pieces']} new piece(s). It now holds {r['in_library']['pieces']} piece(s) "
              f"from {r['in_library']['files']} file(s).")
        for skipped in r["skipped"][:40]:
            print(f"  left out: {skipped['file']}  ({skipped['why']})")
        if len(r["skipped"]) > 40:
            print(f"  ... and {len(r['skipped']) - 40} more left out")

    def building(name: str, fresh: bool = False) -> None:
        started, last = time.time(), [0.0]

        def progress(r: dict) -> None:
            if time.time() - last[0] >= 2.0:
                last[0] = time.time()
                print(f"  {r['done']}/{r['total']} files, {r['pieces']} pieces, {time.time() - started:.0f} s", flush=True)
        show_report(lib.build(root, name, engine.embedder, cfg, progress=progress, fresh=fresh))
    try:
        if what == "list":
            have = lib.names(root)
            if not have:
                print('No libraries yet. Make one:  hermes holonomic library create NAME "C:\\path\\to\\folder"')
                return
            for n in have:
                s = lib.summary(root, n)
                print(f"{n}: {s['pieces']} piece{'s' if s['pieces'] != 1 else ''} from {s['files']} file{'s' if s['files'] != 1 else ''}, "
                      f"built {s['built']}\n    from {s['folder']}" + (f"\n    {s['about']}" if s["about"] else ""))
            print("A library feeds a conversation only once you have asked her to open it there. Without that, she may "
                  "still look a single answer up in one.")
        elif what == "create":
            if len(items) != 2:
                print('Usage: hermes holonomic library create NAME "FOLDER" [--about "what it is for"]')
                return
            made = lib.create(root, items[0], items[1], args.about or "")
            print(f"Made library '{made['name']}' from {made['folder']}. Reading the files in ...")
            building(made["name"])
        elif what == "update":
            if len(items) != 1:
                print("Usage: hermes holonomic library update NAME")
                return
            building(lib.clean_name(items[0]))
        elif what == "rebuild":
            if len(items) != 1:
                print("Usage: hermes holonomic library rebuild NAME")
                return
            name = lib.clean_name(items[0])
            if lib.info(root, name) is None:
                print(f"There is no library called '{name}'.")
                return
            print(f"Reading every file of '{name}' again ...")
            building(name, fresh=True)
        elif what == "show":
            if len(items) != 1:
                print("Usage: hermes holonomic library show NAME")
                return
            name = lib.clean_name(items[0])
            data = lib.info(root, name)
            if data is None:
                print(f"There is no library called '{name}'.")
                return
            s = lib.summary(root, name)
            print(f"{name}: {s['pieces']} pieces from {s['files']} files, built {s['built']}\n  from {s['folder']}"
                  + (f"\n  {s['about']}" if s["about"] else ""))
            if args.left_out:
                gone = data.get("left_out") or []
                print(f"  {len(gone)} file(s) left out:" if gone else "  No file was left out.")
                for g in gone:
                    print(f"    {g['file']}  ({g['why']})")
                return
            for rel, f in sorted((data.get("files") or {}).items()):
                print(f"  {len(f.get('ids') or []):5d}  {rel}")
            if data.get("left_out"):
                print(f"  {len(data['left_out'])} file(s) were left out: hermes holonomic library show {name} --left-out")
        elif what == "search":
            if len(items) < 2:
                print('Usage: hermes holonomic library search NAME "what to look up"')
                return
            name = lib.clean_name(items[0])
            if lib.info(root, name) is None:
                print(f"There is no library called '{name}'.")
                return
            found = lib.search(root, [name], " ".join(items[1:]), engine.embedder, cfg, k=args.n, floor=0.15)
            if not found:
                print("Nothing matched.")
            for f in found:
                print(f"[{f['score']:.2f}{' linked' if f['linked'] else ''}] {f['source']}"
                      + (f"\n    also in: {', '.join(f['also_in'])}" if f.get("also_in") else "") + f"\n    {_clip(f['text'], args.width)}")
        elif what == "delete":
            if len(items) != 1:
                print("Usage: hermes holonomic library delete NAME --yes")
                return
            name = lib.clean_name(items[0])
            data = lib.info(root, name)
            if data is None:
                print(f"There is no library called '{name}'.")
            elif not args.yes:
                s = lib.summary(root, name)
                print(f"This would remove library '{name}' ({s['pieces']} pieces from {s['files']} files). The folder it was "
                      f"built from, {s['folder']}, is not touched. Add --yes to do it.")
            else:
                lib.delete(root, name)
                print(f"Removed library '{name}'. Its folder of material is untouched.")
    except lib.LibraryError as exc:
        print(exc)
    finally:
        lib.close_all(root)


def _plates_cmd(engine, args) -> None:
    """Read-only: whether the write log accounts for the plates, and what the plates do at recall."""
    from . import audit
    st = audit.log_status(engine)
    print(f"Plates: {st['plates']}, of which {st['plates_logged']} have every write in the write log "
          f"({st['bindings']} bindings logged"
          + (f", since {_when(st['log_started'])}" if st["log_started"] else "") + f"). Encoding #{st['encoding']}"
          + (f" of {st['encodings']}." if st["encodings"] > 1 else "."))
    v = audit.verify(engine)
    if v["checked"] or v["cannot"]:
        print(f"Rebuilt from the log alone and compared with what is stored: {v['agree']} of {v['checked']} plate(s) agree "
              f"(largest difference {v['worst']:.1e} of the plate; {v['tolerance']:.0e} is allowed).")
        for d in v["disagree"]:
            print(f"  DISAGREES: plate {d['plate']} ({d['bindings']} bindings) differs by {d['difference']:.2e}")
        for c in v["cannot"]:
            print(f"  not checked: plate {c['plate']} ({c['bindings']} bindings): {c['forgotten']} involve a forgotten memory"
                  + (f", {c['other_encoding']} were written under another encoding" if c["other_encoding"] else ""))
    else:
        print("No plate has been written since the log began, so there is nothing to check yet.")
    if v["unlogged"]:
        print(f"{v['unlogged']} plate(s) were written before the log began and cannot be rebuilt from it.")
    if args.plates_action != "check":
        print("For what the plates do at recall: hermes holonomic plates check")
        return
    d = audit.diagnose(engine, sample=args.n)
    if not d["cues"]:
        print(d.get("note", "Nothing to measure."))
        return
    pl, fg, rs, rc = d["plates"], d["forgotten"], d["resonance"], d["recovery"]
    print(f"\nMeasured with {d['cues']} of {d['live']} memories as cues.")
    print(f"What the plates carry: {pl['members_out_of_recall']} of {pl['members']} memberships belong to memories recall can no "
          f"longer return (retired or marked wrong), on {pl['with_out_of_recall']} of {pl['count']} plates.")
    print(f"  forgotten memories: {fg['total']}; their plates are on record for {fg['on_record']}"
          + (f" (for {fg['plates_unknown']}, forgotten before records were kept, they are not)." if fg["plates_unknown"] else "."))
    for w in pl["worst"]:
        if w["out_of_recall"] or w["forgotten"]:
            print(f"    plate {w['plate']}: {w['members']} members, {w['out_of_recall']} out of recall, {w['forgotten']} forgotten, load {w['load']:.0f}")
    print(f"How many plates answer a cue: typically {rs['median']}, at most {rs['most']}; the limit is {rs['limit']}. "
          f"Cues for which the limit turned a plate away: {rs['cues_with_plates_turned_away']}.")
    known = d["known_links"]
    print(f"Associates a cue is known to have (from the log, or the sources a conclusion was stored with): "
          f"{known['cues_with_any']} of the cues have any.")

    def line(name: str, key: str) -> None:
        t = rc[key]
        share = f"{100 * t['found'] / t['expected']:.0f}%" if t["expected"] else "n/a"
        print(f"  {name:<46} known associates found: {t['found']}/{t['expected']} ({share});  returned in all: {t['returned']};  "
              f"of those, linked by nothing on record: {t['unexplained']}")
    line("as recall does it now", "as it is")
    line("with out-of-recall members left out", "without them")
    line("with no limit on plates read", "no limit")
    print(f"Cues whose result changed when out-of-recall members were left out: {d['cues_where_leaving_them_out_changed_the_result']}.")
    for e in d["examples"]:
        print(f"    cue #{e['cue']}: gained {e['gained'] or 'nothing'} ({e['gained_expected']} known), "
              f"lost {e['lost'] or 'nothing'} ({e['lost_expected']} known)")
    print("Reading this: more returned is not better in itself. What counts is known associates found, and how many\n"
          "returns nothing on record explains. A conversation's neighbouring lines are linked too but were not recorded\n"
          "before the log, so they are never 'known', only explained.")


def _backup_cmd(args) -> None:
    from hermes_constants import get_hermes_home
    from . import backup as bk
    from .provider import load_config
    home = get_hermes_home()
    cfg = load_config(home)
    folder = bk.default_folder(dict(cfg, backup_dir=args.folder) if args.folder else cfg)
    mb = lambda n: f"{n / 1_000_000:.1f} MB"
    try:
        if args.list:
            have = bk.backups(folder)
            print(f"Backups in {folder} (newest first):" if have else f"There are no backups in {folder}.")
            for b in have:
                print(f"  {b['name']}   {mb(b['bytes'])}")
        elif args.holonomic_action == "backup":
            keep = int(cfg.get("backup_keep", 10)) if args.keep is None else args.keep
            done = bk.backup(home, folder, label=args.label or "", keep=keep, version=bk.plugin_version())
            print(f"Backup written: {done['file']}\n  {done['files']} files, {mb(done['bytes'])} (the store is {mb(done['store_bytes'])})")
            for name in done["removed"]:
                print(f"  removed old backup: {name}")
            print("To put it back: hermes holonomic restore   (with Hermes closed)")
        else:
            file = bk.pick(folder, args.file or "")
            info = bk.inspect(file)
            if not args.yes:
                print(f"This would put back: {file}\n  made {info['made_at'] or 'at an unknown time'}"
                      + (f" by version {info['plugin_version']}" if info["plugin_version"] else "") + f", {info['files']} files"
                      + f"\ninto: {home / 'holonomic'}\nWhat is there now would be set aside beside it, not deleted. "
                        "Close Hermes first, then add --yes to do it.")
                return
            done = bk.restore(home, file)
            print(f"Restored from {file.name} into {done['into']}")
            if done["set_aside"]:
                print(f"What was there before is kept at: {done['set_aside']}\nDelete that folder yourself once you are sure "
                      "the restore is what you wanted.")
    except bk.BackupError as exc:
        print(exc)


def holonomic_command(args) -> None:
    try:
        sys.stdout.reconfigure(errors="replace")       # Windows consoles choke on some characters
    except Exception:
        pass
    action = getattr(args, "holonomic_action", None)
    if action not in ("stats", "list", "recall", "reflect", "profile", "show", "forget", "sleep", "dreams", "dreamtalk", "relabel", "images", "tidy", "context", "faces", "library", "backup", "restore", "plates"):
        print('Usage: hermes holonomic stats | list [-n N] | recall "query" [-k N] [--deep] | show ID... | forget ID... [--yes] | '
              'reflect status|on|off|now | sleep status|on|off|now | dreams | images | profile [--history]')
        return
    if action == "faces" and args.faces_action in ("learn", "ask", "unasked"):
        _faces_settings(args)
        return
    if action == "images" and args.images_action in ("on", "off"):
        _images_toggle(args)
        return
    if action == "reflect" and args.reflect_action in ("on", "off"):
        _reflect_toggle(args)
        return
    if action in ("backup", "restore"):          # before the store is opened: a restore has to be able to move it
        _backup_cmd(args)
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
        elif action == "faces":
            _faces_cmd(engine, cfg, args)
        elif action == "library":
            _library_cmd(engine, cfg, args)
        elif action == "plates":
            _plates_cmd(engine, args)
        elif action == "context":
            try:
                print((engine.path / "last_context.txt").read_text(encoding="utf-8").rstrip())
            except OSError:
                print("Nothing yet: this shows what memory gave her for the most recent message, once there has been one.")
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
            table = {"dream": to_dream, "said": to_said, "project": {"fact": "project_fact"},
                     "personal": {"project_fact": "fact"}}[args.to]
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
        elif action == "dreams" and args.dreams_action == "sort":
            from . import images as _img
            done = _img.sort_dream_files(engine)
            where = engine.path / _img.DREAM_FOLDER
            if done["moved"]:
                print(f"Moved {done['moved']} dream picture(s) into {len(done['folders'])} folder(s) under {where}:")
                for folder in sorted(done["folders"]):
                    print(f"  {folder}")
            else:
                print(f"Dream pictures are already in their own folders, under {where}.")
            if done["missing"]:
                print(f"  {done['missing']} file(s) on record were not found on disk.")
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
                        print(f"  #{mid} described image #{mem['meta']['image_id']}, which is set aside with it. To put the image back: "
                              f"hermes holonomic images restore {mem['meta']['image_id']}")
                        continue
                    engine.forget(mid)
                print(f"Forgot {', '.join('#' + str(m) for m in real)}. This cannot be undone. "
                      "Profiles are not changed by this; see `hermes holonomic profile`.")
        elif action == "reflect":
            _reflect(engine, cfg, args)
        elif action == "profile":
            if args.set:
                who, text = args.set
                if who not in ("user", "projects", "self", "us") or len(text.strip()) < 3:
                    print('Usage: hermes holonomic profile --set user|projects|self|us "text"')
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
            for who, title in (("user", "About the user"), ("projects", "What the user is working on"),
                               ("self", "About herself"), ("us", "About the two of you")):
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
    if getattr(args, "who", None):
        values["dream_image_people"] = args.who
    if getattr(args, "name_people", None):
        values["dream_name_people"] = args.name_people
    if args.attempts:
        values["dream_image_candidates"] = max(1, min(args.attempts, 8))
    if args.swap:
        values["dream_image_swap"] = args.swap == "on"
    if args.enlarge is not None:
        values["dream_image_enlarge"] = 0 if args.enlarge <= 1 else min(args.enlarge, 4.0)
    if args.enlarge_model is not None:
        values["dream_image_enlarge_model"] = args.enlarge_model.strip()
    if args.signature:
        values["dream_image_keep_signature"] = args.signature == "keep"
    if args.redraw_below is not None:
        values["dream_image_redraw_below"] = max(0, min(args.redraw_below, 11))
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
        if mode == "from_images" and int(sc["dream_image_redraw_below"] or 0) > 0 and int(sc["dream_image_candidates"]) >= 1:
            print(f"  a picture drawn from an image is drawn again, holding less to it, if her best attempt scores below "
                  f"{sc['dream_image_redraw_below']} of 10 (--redraw-below 0 to stop)")
        if float(sc["dream_image_enlarge"] or 0) > 1:
            f = float(sc["dream_image_enlarge"])
            print(f"  the picture she keeps is enlarged {f:g} times, to {round(int(sc['dream_image_width']) * f)}x{round(int(sc['dream_image_height']) * f)}"
                  + ("" if sc["dream_image_api"] in ("comfyui", "a1111") else "  [this image server cannot enlarge: comfyui or a1111 only]"))
        if mode == "from_images":
            print("  a signature or watermark on an image is " + ("carried into pictures drawn from it (--signature remove to stop)"
                  if sc["dream_image_keep_signature"] else "kept out of pictures drawn from it"))
        if sc["dream_image_swap"]:
            print("  one graphics card: the language models are unloaded while pictures are drawn and reloaded afterwards")
    if mode == "from_images":
        from . import faces as _f
        policy = _f.dream_policy(load_config(home), bool(sc["dream_image_use_people"]))
        print("  images with real people in them: " + {"none": "are not drawn from", "anyone": "may be drawn from, whoever is in them",
              "me": "are drawn from only if every face in them is yours", "named": "are drawn from only if every face in them is someone you named"}[policy]
              + "   (--who none|me|named|anyone)")
        named = _f.face_config(load_config(home))["dream_name_people"]
        print("  people in a dream by name, going by their faces: " + {"none": "nobody", "me": "only you", "named": "anyone you have named"}.get(named, "nobody")
              + "   (--name-people none|me|named)")
        print(f"  how far a picture may move from the images it starts from: {sc['dream_image_strength']} (--strength)")
    if mode in ("pictures", "from_images"):
        print(f"  style added to every scene: {sc['dream_image_style'] or '(none)'} (--style)")
    if not cfg.get("image_enabled"):
        print("  Image memory is off, so there are no images to dream of. Turn on with: hermes holonomic images on")
    if values:
        print("Restart Hermes for a running session to pick this up.")


_LEARN_SAYS = {"none": "no face is looked for at all",
               "me": "only your own face is learned and recognised; no other face is kept",
               "named": "people you have named are learned and recognised; no other face is kept",
               "often": "every face is kept, so that someone who keeps appearing can be noticed"}


def _faces_status(cfg) -> None:
    from . import faces as _f
    fc = _f.face_config(cfg)
    print(f"Whose faces she may learn: {fc['face_learn']}  ({_LEARN_SAYS[fc['face_learn']]})")
    print(f"  she may ask who someone is who keeps appearing: {'yes' if fc['face_ask_names'] else 'no'}"
          + ("" if fc["face_learn"] == "often" or not fc["face_ask_names"] else "   (only applies when every face is kept: faces learn often)"))
    print(f"  she is told who is in an image without being asked: {'yes' if fc['face_name_unasked'] else 'no, only when you ask who someone is'}")
    print("  change with: hermes holonomic faces learn none|me|named|often,  faces ask on|off,  faces unasked on|off")


def _faces_settings(args) -> None:
    from hermes_constants import get_hermes_home
    from . import faces as _f
    from .provider import load_config, write_config
    home = get_hermes_home()
    what, value = args.faces_action, (args.items[0].lower() if args.items else "")
    if what == "learn":
        if value not in _f.LEARN:
            print("Usage: hermes holonomic faces learn none|me|named|often")
            for k in _f.LEARN:
                print(f"  {k:<6} {_LEARN_SAYS[k]}")
            return
        write_config(home, {"face_learn": value})
        print(f"Whose faces she may learn: {value}  ({_LEARN_SAYS[value]}).")
        if value == "none":
            print("  Faces already kept stay where they are, unused. To drop them all: hermes holonomic faces forget --all --yes")
        else:
            print("  Every image is looked at again for faces under this rule when things are quiet (or now: hermes holonomic faces scan).")
            print("  The helper server needs OpenCV for this: see hermes holonomic faces")
        if (home / "holonomic" / "holonomic.db").exists():     # what is kept depends on the rule: everything is gone through again
            engine, _ = _open()
            if engine is not None:
                try:
                    engine.kv_set("faces:look_again", "1")
                finally:
                    engine.close()
    elif value in ("on", "off"):
        key = "face_ask_names" if what == "ask" else "face_name_unasked"
        write_config(home, {key: value == "on"})
        print({"face_ask_names": {True: "She may ask, once, who a person is who keeps appearing (applies when every face is kept: faces learn often).",
                                  False: "She does not ask who people are."},
               "face_name_unasked": {True: "She is told who is in an image whenever she recognises someone, and may say so.",
                                     False: "She is told who is in an image only when you ask who someone is."}}[key][value == "on"])
    else:
        print(f"Usage: hermes holonomic faces {what} on|off")
        return
    print("Restart Hermes for a running session to pick this up.")


def _faces_cmd(engine, cfg, args) -> None:
    from . import faces as _f
    from . import images as _img
    what, items = args.faces_action, list(args.items or [])
    fc = _f.face_config(cfg)
    if what == "status":
        _faces_status(cfg)
        model = _f.available(cfg)
        print(f"  helper server at {fc['host']}: " + (f"faces ready ({model})" if model else
              "faces NOT available. It needs OpenCV in the Python that runs it, then a restart of the helper:"))
        if not model:
            print("    <ComfyUI>\\.venv\\Scripts\\python.exe -m pip install opencv-python-headless")
        known = _f.people(engine)
        print(f"  people she knows: {', '.join(p['name'] + (' (you)' if p['is_user'] else '') for p in known) or 'nobody'}"
              + (f"   images not yet looked at for faces: {_f.waiting(engine)}" if _f.faces_on(cfg) else ""))
        from .sleep import sleep_config
        print(f"  in images a dream picture is drawn from: {_f.dream_policy(cfg, bool(sleep_config(cfg)['dream_image_use_people']))}"
              "   (hermes holonomic dreams images --who none|me|named|anyone)")
        print(f"  named in dreams, going by their faces: {fc['dream_name_people']}   (hermes holonomic dreams images --name-people none|me|named)")
    elif what == "people":
        known = _f.people(engine)
        print(f"{len(known)} person(s) she knows by face." if known else "She knows nobody by face.")
        ids = lambda xs: ", ".join("#" + str(i) for i in xs) or "none"
        for p in known:
            print(f"  {p['name']}" + ("  (you)" if p["is_user"] else "") + ("" if p["dream"] is None else f"   in dreams: {'yes' if p['dream'] else 'never'}"))
            print(f"    you said so in image {ids(p['said'])};  she recognised them in image {ids(p['seen'])}")
    elif what == "show":
        if not items or not items[0].isdigit():
            print("Usage: hermes holonomic faces show ID      (the faces in an image, numbered from the left)")
            return
        try:
            found = _f.look(engine, cfg, int(items[0]))
        except _f.FaceError as exc:
            print(str(exc))
            return
        print(f"image #{items[0]}: {len(found)} face(s)" + (":" if found else "."))
        for f in found:
            print(f"  face {f['n']}  {f['where']}:  " + (f"{f['name']} ({'you said so' if f['said'] else 'recognised'})" if f["name"] else "not known"))
    elif what == "name":
        if len(items) < 2 or not items[0].isdigit():
            print("Usage: hermes holonomic faces name ID NAME [--face N] [--me]      (a face in that image is this person; --me: it is you)")
            return
        try:
            done = _f.name_person(engine, cfg, int(items[0]), " ".join(items[1:]), face=args.face, me=args.me)
        except _f.FaceError as exc:
            print(str(exc) + ("\n  e.g. hermes holonomic faces name " + items[0] + " " + " ".join(items[1:]) + " --face 1"
                              if isinstance(exc, _f.NeedsChoice) else ""))
            return
        print(f"image #{done['image_id']}: face {done['face']} ({done['where']}) is {done['name']}" + (", which is you." if done["is_user"] else "."))
        print("  Other images are looked at again for that face when things are quiet (or now: hermes holonomic faces scan).")
    elif what == "not":
        if len(items) < 2 or not items[0].isdigit():
            print("Usage: hermes holonomic faces not ID NAME [--face N]      (a face in that image is not this person)")
            return
        n = _f.not_person(engine, cfg, int(items[0]), " ".join(items[1:]), face=args.face)
        print(f"image #{items[0]}: noted, that is not {' '.join(items[1:])}. The face will not be taken for them again." if n
              else f"Nothing in image #{items[0]} was taken for {' '.join(items[1:])}.")
    elif what == "dream":
        if len(items) < 2 or items[-1] not in ("yes", "no", "default"):
            print("Usage: hermes holonomic faces dream NAME yes|no|default      (may this person be in an image a dream picture is drawn from?)")
            return
        name = " ".join(items[:-1])
        ok = _f.set_dream(engine, name, {"yes": True, "no": False, "default": None}[items[-1]])
        print(({"yes": f"{name} may be in images dreams are drawn from, where the general rule allows people.",
                "no": f"An image with {name} in it is never drawn from in a dream.",
                "default": f"{name}: back to the general rule."}[items[-1]]) if ok else f"She knows nobody called {name!r}.")
    elif what == "forget":
        if args.all:
            if not args.yes:
                print("This drops every face and every person she knows by face. Images and their descriptions stay. "
                      "To do it: hermes holonomic faces forget --all --yes")
                return
            print(f"Dropped {_f.forget_all(engine)} face(s) and everyone she knew by face.")
            return
        if not items:
            print("Usage: hermes holonomic faces forget NAME   |   hermes holonomic faces forget --all --yes")
            return
        name = " ".join(items)
        print(f"Forgot who {name} is; their faces are no longer anyone's." if _f.forget_person(engine, name) else f"She knows nobody called {name!r}.")
    elif what == "often":
        if fc["face_learn"] != "often":
            print("People who keep appearing are only noticed when every face is kept: hermes holonomic faces learn often")
            return
        groups = _f.strangers(engine, cfg)
        print(f"{len(groups)} person(s) she does not know who are in {fc['face_often_images']} or more images." if groups
              else f"Nobody she does not know is in {fc['face_often_images']} or more images.")
        for g in groups:
            print(f"  in image {', '.join('#' + str(i) for i in g['images'])}" + ("   (she has asked who this is)" if g["asked"] else "")
                  + f"\n    to say who: hermes holonomic faces show {g['images'][0]}   then   hermes holonomic faces name {g['images'][0]} NAME --face N")
    elif what == "image":
        if len(items) < 2 or items[-1] not in ("on", "off") or not all(raw.isdigit() for raw in items[:-1]):
            print("Usage: hermes holonomic faces image ID... off|on      (do not look for faces in these images, e.g. a street full of strangers)")
            return
        for raw in items[:-1]:
            if not _f.set_image(engine, int(raw), items[-1] == "on"):
                print(f"image #{raw}: no such image")
            elif items[-1] == "off":
                print(f"image #{raw}: no face is looked for in it from now on, and every face kept from it has been dropped.")
            else:
                print(f"image #{raw}: faces are looked for in it again, under the general rule, on the next pass (or now: hermes holonomic faces scan).")
    elif what == "scan":
        if not _f.faces_on(cfg):
            print("Learning faces is off. Turn it on first: hermes holonomic faces learn me|named|often")
            return
        t0 = time.time()
        done = _f.scan(engine, cfg, again=True)
        print(f"Looked for faces in {len(done['done'])} image(s) in {time.time() - t0:.0f} s.")
        for err in done["errors"]:
            print(f"  stopped: {err}")


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
        for n in img.get("named") or []:
            print(f"    shows {n['name']}: " + ("you said so" if n["said"] else f"she recognised it by its look (alike {n['alike']:.2f})")
                  + f"   (wrong? hermes holonomic images name {img['id']} \"{n['name']}\" --not)")
        if img.get("faces_off"):
            print(f"    faces: not looked for in this image   (hermes holonomic faces image {img['id']} on)")
        signed = img.get("signature")
        print(f"    signature or watermark: {'not checked yet (checked the first time a dream draws from it)' if not signed else signed}"
              f"   (wrong? hermes holonomic images signature {img['id']} none|bottom-right|...)")
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
        aside = len(_img.removed_images(engine))
        if aside:
            print(f"  set aside (forgotten, not deleted): {aside}   (hermes holonomic images removed)")
        from . import fingerprints as _fp
        if _fp.fingerprint_config(cfg)["image_fingerprints"]:
            st = _fp.status(engine)
            print(f"  fingerprints: on, {st['fingerprinted']} of {st['images']} image(s) have one   (hermes holonomic images fingerprints)")
        print(f"  files are in: {engine.path / 'images'}\n  dream pictures are in: {engine.path / _img.DREAM_FOLDER}, a folder for each sleep")
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
        prints = report.get("fingerprints") or {}
        if prints.get("done"):
            print(f"  made fingerprints for {len(prints['done'])} image(s).")
        for err in prints.get("errors", []):
            print(f"  fingerprints not made: {err}")
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
    elif what == "fingerprints":
        from hermes_constants import get_hermes_home
        from . import fingerprints as _fp
        from .provider import load_config, write_config
        sub = items[0] if items else "status"
        if sub not in ("status", "on", "off", "redo"):
            print("Usage: hermes holonomic images fingerprints [status|on|off|redo] [--host URL]")
            return
        if sub in ("on", "off") or args.host:
            write_config(get_hermes_home(), dict({"image_fingerprints": sub == "on"} if sub in ("on", "off") else {},
                                                 **({"image_fingerprint_host": args.host.rstrip("/")} if args.host else {})))
            cfg = dict(cfg, **load_config(get_hermes_home()))
        fc = _fp.fingerprint_config(cfg)
        print(f"Picture fingerprints are {'ON' if fc['image_fingerprints'] else 'OFF'}.  Helper server: {fc['image_fingerprint_host']}")
        try:
            about = _fp.health(cfg)
            print(f"  the helper is running: {about.get('model')}, {about.get('dim')} numbers per picture, on {about.get('device')}")
        except _fp.FingerprintError as exc:
            about = None
            print(f"  the helper is NOT answering: {exc}")
            print("  start it with a Python that has torch and transformers, for example ComfyUI's:")
            print(f"    <ComfyUI>\\.venv\\Scripts\\python.exe \"{Path(__file__).parent / 'tools' / 'fingerprint_server.py'}\"")
        st = _fp.status(engine, (about or {}).get("model"))
        print(f"  {st['fingerprinted']} of {st['images']} image(s) have a fingerprint" + (f" from {', '.join(st['models'])}" if st["models"] else ""))
        if fc["image_fingerprints"] and about and (st["waiting"] or sub == "redo"):
            t0 = time.time()
            done = _fp.fingerprint(engine, cfg, redo=sub == "redo")
            print(f"  made fingerprints for {len(done['done'])} image(s) and their parts in {time.time() - t0:.0f} s")
            for err in done["errors"]:
                print(f"  stopped: {err}")
        if not fc["image_fingerprints"]:
            print("  Turn on with: hermes holonomic images fingerprints on [--host URL]")
    elif what == "names":
        from . import fingerprints as _fp
        if items and items[0] == "forget":
            for raw in items[1:] or [""]:
                print(f"Forgot the name {raw!r}: nothing is recognised as it any more." if _fp.forget_name(engine, raw)
                      else "Usage: hermes holonomic images names forget NAME      (a name from `hermes holonomic images names`)")
            return
        known = _fp.names(engine)
        on = _fp.names_on(cfg)
        print(f"{len(known)} named thing(s)." + ("" if on else "  Recognising them is OFF: it needs picture fingerprints "
                                                 "(hermes holonomic images fingerprints on)."))
        for k in known:
            ids = lambda xs: ", ".join("#" + str(i) for i in xs) or "none"
            print(f"  {k['name']}" + (f"  ({k['what']})" if k["what"] else ""))
            print(f"    you said so in image {ids(k['examples'])};  she recognised it in image {ids(k['recognised'])}"
                  + (f";  you said it is not in image {ids(k['not'])}" if k["not"] else ""))
        if not known:
            print('  Name something: hermes holonomic images name ID NAME [--what "a long-haired black and white cat"]')
            print("  or tell her when you show her a picture: \"This is Sushi, my cat.\"")
    elif what == "name":
        from . import fingerprints as _fp
        if len(items) < 2 or not items[0].isdigit():
            print('Usage: hermes holonomic images name ID NAME [--what "what it is"]      this image shows that particular thing\n'
                  "       hermes holonomic images name ID NAME --not                   this image does not show it")
            return
        image_id, name = int(items[0]), " ".join(items[1:])
        if not _fp.names_on(cfg):
            print("Named things need picture fingerprints. Turn them on: hermes holonomic images fingerprints on")
            return
        if getattr(args, "not_it", False):
            print(f"image #{image_id}: noted, it does not show {name}. It will not be taken for it again."
                  if _fp.not_named(engine, cfg, image_id, name, key_fn=extract_keys)
                  else f"Nothing is known by the name {name!r}, or there is no image #{image_id}.")
            return
        try:
            done = _fp.name_thing(engine, cfg, image_id, name, what=args.what or "", key_fn=extract_keys)
        except _fp.NameRefused as exc:
            print(str(exc))
            return
        print(f"image #{image_id} shows {done['name']}" + (f" ({done['what']})" if done["what"] else "") + ". "
              + ("She looked at it again with that in mind; its parts will be looked at again when things are quiet "
                 "(or now: hermes holonomic images process)." if done["looked_again"] else "What is written about it already says so."))
        print(f"  Later images in which something looks like {done['name']} will be checked for it.")
    elif what == "similar":
        from . import fingerprints as _fp
        if not items or not items[0].isdigit():
            print("Usage: hermes holonomic images similar ID [-n N] [--all]      (images that look like this one; --all shows every figure)")
            return
        img = _img.get_image(engine, int(items[0]))
        if not img:
            print(f"image #{items[0]}: no such image")
            return
        if not img.get("fingerprint"):
            print(f"image #{img['id']} has no fingerprint yet. Run: hermes holonomic images fingerprints")
            return
        floor = _fp.fingerprint_config(cfg)["image_fingerprint_min"]
        found = _fp.similar(engine, cfg, img["id"], n=args.n if not args.all else 10_000, minimum=-1.0 if args.all else None)
        print(f"image #{img['id']}  {img['origin'] or ''}  {_clip(img['description'], args.width)}")
        print(f"{len(found)} image(s) " + ("compared" if args.all else f"look like it (alike {floor:g} or more; 1 = the same picture)") + ":"
              if found else f"Nothing she has been shown looks like it (alike {floor:g} or more). --all shows the figures for every image.")
        for f in found:
            other = _img.get_image(engine, f["id"]) or {}
            print(f"  {f['alike']:.2f}  image #{f['id']}  {other.get('origin') or ''}   alike: {_fp.describe_match(f)}   (whole pictures {f['whole']:.2f})")
            print(f"        {_clip(other.get('description') or '(not described yet)', args.width)}")
    elif what == "signature":
        place = " ".join(items[1:]).lower().replace("-", " ") if len(items) > 1 else ""
        if not items or not items[0].isdigit() or place not in _img.SIGNATURE_PLACES:
            print("Usage: hermes holonomic images signature ID none|top-left|top-right|bottom-left|bottom-right"
                  "      (where the image is signed or watermarked)")
            return
        if not _img.set_signature(engine, int(items[0]), place):
            print(f"image #{items[0]}: no such image")
            return
        print(f"image #{items[0]}: " + ("no signature or watermark." if place == "none" else
                                        f"signed or watermarked at the {place}. That corner is smoothed over before dream pictures are drawn from it."))
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
        found = [img for img in (_img.get_image(engine, int(raw)) for raw in items if raw.isdigit()) if img]
        if not found:
            print("No such image. Usage: hermes holonomic images forget ID...")
            return
        for img in found:
            _print_image(img, args.width)
            _img.forget_image(engine, img["id"])
        ids = " ".join(str(i["id"]) for i in found)
        print(f"Set aside image {', '.join('#' + str(i['id']) for i in found)}. She no longer recalls or shows "
              f"{'it' if len(found) == 1 else 'them'}; nothing has been deleted.\n"
              f"  to put {'it' if len(found) == 1 else 'them'} back as before:  hermes holonomic images restore {ids}\n"
              f"  to remove for good, files included:  hermes holonomic images delete {ids}")
    elif what == "removed":
        gone = _img.removed_images(engine)
        if not gone:
            print("No images are set aside.")
            return
        print(f"{len(gone)} image(s) set aside. `images restore ID` puts one back as it was; `images delete ID` removes it for good.")
        for g in gone:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(g["removed_at"])) if g["removed_at"] else "by an earlier version"
            print(f"  image #{g['id']}  {g['origin'] or '(no file name)'}  set aside {when}"
                  + ("" if g["files_present"] else "  [its file is missing: cannot be restored]"))
            if g["description"]:
                print(f"    {_clip(g['description'], args.width)}")
            print(f"    {g['original']}")
    elif what == "restore":
        if not items or not all(raw.isdigit() for raw in items):
            print("Usage: hermes holonomic images restore ID...      (see what is set aside: hermes holonomic images removed)")
            return
        for raw in items:
            img = _img.restore_image(engine, cfg, int(raw))
            if not img:
                print(f"image #{raw}: nothing to restore (it is not set aside, or its file is gone).")
                continue
            print(f"Restored image #{raw}" + (" with her description, its parts and what was said about it." if img["description"]
                                               else ". It had no description; she will describe it afresh (hermes holonomic images process)."))
            _print_image(img, args.width)
    elif what == "delete":
        aside = {g["id"]: g for g in _img.removed_images(engine)}
        if not items or not all(raw.isdigit() for raw in items):
            print("Usage: hermes holonomic images delete ID... [--yes]      (only images already set aside with `images forget`)")
            return
        wanted = [int(raw) for raw in items]
        for i in [i for i in wanted if i not in aside]:
            print(f"image #{i} has not been set aside, so it is not deleted. Set it aside first: hermes holonomic images forget {i}")
        wanted = [i for i in wanted if i in aside]
        if not wanted:
            return
        for i in wanted:
            print(f"  image #{i}  {aside[i]['origin'] or '(no file name)'}  {_clip(aside[i]['description'], args.width)}")
        if not args.yes:
            print(f"Nothing deleted. To delete {'this image' if len(wanted) == 1 else 'these images'} and "
                  f"{'its' if len(wanted) == 1 else 'their'} files for good, run the same command with --yes. This cannot be undone.")
            return
        done = [i for i in wanted if _img.delete_image(engine, i)]
        print(f"Deleted image {', '.join('#' + str(i) for i in done)} and the files. This cannot be undone.")


def _redrawn(p) -> str:
    r = p["redrawn"]
    after = "she could not judge the new attempts" if r["after"] is None else f"the best of the new attempts scored {r['after']}"
    return (f"\n            the one she chose scored {r['before']} of 10, so it was drawn again holding less to the image: {after}; "
            "compared side by side, " + ("she kept a new one" if r["kept"] else "she kept the earlier one" + (f": {r['why']}" if r.get("why") else "")))


def _print_calls(report) -> None:
    for c in report["calls"]:
        if c.get("drawing"):
            print(f"  step {c['step']}: {c.get('seconds', 0):.0f} s")
            continue
        print(f"  step {c['step']}{' (thinking)' if c.get('think') else ''}: {c.get('seconds', 0):.0f} s, prompt {c.get('prompt_tokens')} tokens, "
              f"reply {c.get('reply_tokens')} tokens, finished: {c.get('done_reason')}" + (f"  [{c['failed']}]" if c.get("failed") else ""))
        if c.get("noted"):
            print(f"      {c['noted']}")
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
                  + (_redrawn(p) if p.get("redrawn") else "")
                  + (f"\n            she chose attempt {p['chosen']} of {p['of']}" + (" drawn again" if (p.get("redrawn") or {}).get("kept") else "")
                     + (f": {p['why']}" if p.get("why") else "") if p.get("of") else "")
                  + (f"\n            enlarged to {p['enlarged'][0]}x{p['enlarged'][1]}" if p.get("enlarged") else "")
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
    if args.reflect_action == "merge":
        from .reflect import merge_facts
        if args.restore:
            for mid in args.restore:
                print(f"[#{mid}] " + ("brought back" if engine.unsupersede(mid) else "is not a retired statement"))
            return
        try:
            report = merge_facts(engine, cfg, apply=bool(args.apply), ask=not args.quick)
        except ReflectionError as exc:
            print(f"Merging failed: {exc}")
            return
        if not report["merged"]:
            print("Nothing says the same thing twice.")
        for m in report["merged"]:
            print(f"  keep   [#{m['keep']}] {_clip(m['kept'], 150)}\n  {'retired' if args.apply else 'retire '} [#{m['drop']}] {_clip(m['dropped'], 150)}   ({m['how']})")
        print(f"{len(report['merged'])} statement(s) {'retired' if args.apply else 'would be retired'}; {report['kept_apart']} close pair(s) "
              f"are different facts and were left alone" + (f"; the model was asked about {report['asked']} pair(s)." if report["asked"] else "."))
        if report["merged"] and not args.apply:
            print("Nothing was changed. If this looks right, run it again with --apply.")
        elif report["merged"]:
            print("A retired statement leaves recall but stays on record. To bring one back: hermes holonomic reflect merge --restore ID")
        return
    if args.reflect_action == "sort":
        from .reflect import sort_facts
        try:
            report = sort_facts(engine, cfg, apply=bool(args.apply))
        except ReflectionError as exc:
            print(f"Sorting failed: {exc}")
            return
        print(f"{report['personal']} fact(s) are about you as a person; {report['project']} are about what you are working on.")
        if report["to_project"]:
            print(f"{'Marked' if args.apply else 'Would mark'} as project facts:")
            for m in report["to_project"]:
                print(f"  [#{m['id']}] {_clip(m['text'], 150)}")
        print("Staying personal:")
        for m in report["stays_personal"]:
            print(f"  [#{m['id']}] {_clip(m['text'], 150)}")
        for who, title in (("user", "About you"), ("projects", "What you are working on")):
            if report["profiles"].get(who):
                print(f"{title} ({'written' if args.apply else 'would be written'}):\n  {report['profiles'][who]}")
        if not args.apply:
            print("Nothing was changed. If the sorting looks right, run it again with --apply. To put one fact right "
                  "afterwards: hermes holonomic relabel ID --as personal   (or --as project)")
        else:
            print("Done. To put one fact right: hermes holonomic relabel ID --as personal   (or --as project)")
        return
    if args.reflect_action == "now":
        if args.think or args.no_think:
            cfg = dict(cfg, reflect_think=bool(args.think))
        if args.depth:
            cfg = dict(cfg, reflect_depth=args.depth)
        accepted: list = []                          # what earlier passes of this run took in, for the profiles

        def one(start_after=None):
            try:
                t0 = time.perf_counter()
                from hermes_constants import get_hermes_home
                from .reflect import read_foundation
                report = reflect_once(engine, cfg, dry_run=args.dry_run, key_fn=extract_keys,
                                      foundation=read_foundation(get_hermes_home()), start_after=start_after, earlier=accepted)
                accepted.extend(report.get("accepted") or [])
            except ReflectionError as exc:
                from .reflect import LAST_CALL
                print(f"Reflection failed: {exc}")
                if LAST_CALL:
                    print(f"  last model call: {LAST_CALL['seconds']:.0f} s, prompt {LAST_CALL['prompt_tokens']} tokens, "
                          f"reply {LAST_CALL['reply_tokens']} tokens, finished: {LAST_CALL['done_reason']}")
                return None
            print(f"Read {report['read']} memories in {time.perf_counter() - t0:.0f} s at depth {report['depth']}"
                  + (" (dry run: nothing stored)" if args.dry_run else ""))
            for c in report["calls"]:
                print(f"  step {c['step']}: {c.get('seconds', 0):.0f} s, prompt {c.get('prompt_tokens')} tokens, "
                      f"reply {c.get('reply_tokens')} tokens, thinking {'on' if c.get('think') else 'off'}, "
                      f"finished: {c.get('done_reason')}" + (f"  [{c['failed']}]" if c.get("failed") else ""))
            if report.get("cut_short"):
                print(f"  NOTE: the model's reply ran to its limit and was cut off. The {report['cut_short']} items it had "
                      f"written out in full were kept; anything after them was lost.")
            for c in report.get("checked") or []:
                if c["verdict"] == "drop":
                    print(f"  CHECK dropped ({c['kind']}): {c['was']}")
                elif c["verdict"] == "unquote":
                    print(f"  UNQUOTED ({c['kind']}), not the user's own words: {c['was']}\n             ->  {c['now']}")
                elif c["verdict"] == "respell":
                    print(f"  NAME put right ({c['kind']}): {c['was']}\n             ->  {c['now']}")
                else:
                    print(f"  CHECK rewrote ({c['kind']}): {c['was']}\n             ->  {c['now']}")
            for kind, items in (report.get("proposed") or {}).items():
                for item in items:
                    print(f"  {kind:<9} {item['text']}   <- {', '.join('#' + str(s) for s in item['sources'])}")
            for text in report.get("cut_off") or []:
                print(f"  cut off   {text}   (stopped mid-sentence; not stored)")
            for text in report.get("dropped_assistant_only") or []:
                print(f"  dropped   {text}   (rested only on the assistant's own words)")
            for text in report.get("dropped_library") or []:
                print(f"  dropped   {text}   (a reference library is her tool, not part of your work)")
            for item in report.get("superseded") or []:
                print(f"  REPLACES  [#{item['fact']}] {item['was']}\n        ->  {item['replacement']}   <- "
                      f"{', '.join('#' + str(s) for s in item['sources'])}")
            for who, text in (report.get("profiles") or {}).items():
                print(f"  profile ({who}): {text or '(empty)'}")
            for who, less in (report.get("profile_shortened") or {}).items():
                print(f"  profile ({who}) came back too long and was said again {less} characters shorter.")
            for who, lost in (report.get("profile_cut") or {}).items():
                print(f"  NOTE: the profile ({who}) was still too long and its last {lost} characters were cut. "
                      f"Raise profile_max_chars if what was cut matters.")
            if not args.dry_run:
                print(f"Stored {len(report['stored'])} new, reinforced {len(report['reinforced'])} existing, "
                      f"profiles updated: {', '.join(report['profiles_updated']) or 'none'}")
            return report

        if args.again is None:
            one()
            return
        # Going over conversation that was read before: a pass can miss something.
        first = engine.first_id_since(time.time() - float(args.again) * 86400.0)
        if first is None:
            print(f"Nothing was said in the last {args.again:g} day(s).")
            return
        print(f"Going over the last {args.again:g} day(s) again, from memory #{first}"
              + (" (dry run: nothing stored)." if args.dry_run else ". Facts she already has are reinforced, not stored twice."))
        at, passes, more = first - 1, 0, True
        while more and passes < 25:                  # to the end: what is new since the last pass is read as well
            report = one(at)
            passes += 1
            more = bool(report and report.get("read") and report.get("last_id", at) > at)
            if more:
                at = report["last_id"]
        if more:
            print(f"Stopped after {passes} passes, at memory #{at}. Run it again with fewer days to go over the rest.")
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
    subs.add_parser("context", help="Show what memory gave the agent for the most recent message")
    fa = subs.add_parser("faces", help="Knowing particular people by their faces (off until you turn it on)")
    fa.add_argument("faces_action", nargs="?", default="status",
                    choices=["status", "learn", "ask", "unasked", "people", "show", "name", "not", "dream", "forget", "often", "scan", "image"])
    fa.add_argument("items", nargs="*", help="A setting, an image id, a name")
    fa.add_argument("--face", type=int, help="With 'name' and 'not': which face in the image, 1 = leftmost")
    fa.add_argument("--me", action="store_true", help="With 'name': this is you")
    fa.add_argument("--all", action="store_true", help="With 'forget': every face and every person")
    fa.add_argument("--yes", action="store_true", help="With 'forget --all': actually do it")
    fa.add_argument("--width", type=int, default=110, help="Characters of text to show")
    lb = subs.add_parser("library", help="Reference libraries: material to work from, kept apart from memory")
    lb.add_argument("library_action", nargs="?", default="list", choices=["list", "create", "update", "rebuild", "show", "search", "delete"])
    lb.add_argument("items", nargs="*", help="A library name; for 'create' the folder too; for 'search' the words")
    lb.add_argument("--about", help="With 'create': one line saying what the library is for")
    lb.add_argument("--yes", action="store_true", help="With 'delete': actually do it")
    lb.add_argument("--left-out", action="store_true", help="With 'show': the files that were left out, and why")
    lb.add_argument("-n", type=int, default=6, help="With 'search': how many results")
    lb.add_argument("--width", type=int, default=300, help="Characters of text to show")
    bkp = subs.add_parser("backup", help="Write the whole store (memories, images, libraries, settings) to one zip file")
    bkp.add_argument("--to", dest="folder", help="Folder for backups (default: holonomic-backups in your Documents, or 'backup_dir')")
    bkp.add_argument("--label", help="A word to add to the file name, e.g. before-update")
    bkp.add_argument("--keep", type=int, help="How many backups to keep; older ones are removed (default 10; 0 keeps all)")
    bkp.add_argument("--list", action="store_true", help="Show the backups there are")
    rst = subs.add_parser("restore", help="Put the store back from a backup; what is there now is set aside, not deleted")
    rst.add_argument("file", nargs="?", help="A backup's name or path (default: the newest)")
    rst.add_argument("--from", dest="folder", help="Folder the backups are in")
    rst.add_argument("--list", action="store_true", help="Show the backups there are")
    rst.add_argument("--yes", action="store_true", help="Actually do it")
    plt = subs.add_parser("plates", help="Check the write log against the plates, and measure what the plates do at recall (changes nothing)")
    plt.add_argument("plates_action", nargs="?", choices=["status", "check"], default="status")
    plt.add_argument("-n", type=int, default=300, help="With 'check': how many memories to use as cues (default 300)")
    td = subs.add_parser("tidy", help="Find stored messages that are Hermes' notes about attachments, and remove them")
    td.add_argument("--apply", action="store_true", help="Remove what is found (without this, it is only listed)")
    td.add_argument("--width", type=int, default=110, help="Characters of text to show")
    rl = subs.add_parser("relabel", help="Correct whether pieces of conversation are labelled as dream talk")
    rl.add_argument("ids", type=int, nargs="+", help="Memory ids, as shown in brackets")
    rl.add_argument("--as", dest="to", choices=["said", "dream", "personal", "project"], required=True,
                    help="'said' = ordinary conversation, 'dream' = talk about a dream; for a fact about you: "
                         "'personal' = about you as a person, 'project' = about something you are working on")
    dt = subs.add_parser("dreamtalk", help="Find stored conversation that is talk about a dream, and label it")
    dt.add_argument("--apply", action="store_true", help="Label what is found (without this, it is only listed)")
    dt.add_argument("--width", type=int, default=110, help="Characters of text to show")
    drm = subs.add_parser("dreams", help="Show recent dreams, or set what images do in dreams")
    drm.add_argument("dreams_action", nargs="?", choices=["show", "images", "sort"], default="show")
    drm.add_argument("mode", nargs="?", help="With 'images': off, words, pictures or from_images")
    drm.add_argument("-n", type=int, default=5, help="How many (default 5)")
    drm.add_argument("--api", choices=["comfyui", "a1111", "openai"], help="With 'images': which interface the image generator speaks")
    drm.add_argument("--host", help="With 'images': address of the image generator, e.g. http://10.0.0.21:8188")
    drm.add_argument("--test", metavar="WORDS", help="With 'images': draw one picture from these words now, to check the image generator")
    drm.add_argument("--model", help="With 'images': model or checkpoint name, if the server needs one")
    drm.add_argument("--size", help="With 'images': picture size, e.g. 768x512")
    drm.add_argument("--count", type=int, help="With 'images': pictures per dream")
    drm.add_argument("--people", choices=["yes", "no"], help="With 'images': may images with real people in them be drawn from")
    drm.add_argument("--name-people", choices=["none", "me", "named"], help="With 'images': whether a dream is told who the people in "
                     "the images it draws on are, so they can be in it by name: nobody, only you, or anyone you have named")
    drm.add_argument("--who", choices=["none", "me", "named", "anyone"], help="With 'images': whose images may be drawn from, going by "
                     "faces: no image with people; only images where every face is yours; only where every face is someone you named; anyone")
    drm.add_argument("--attempts", type=int, help="With 'images': how many times each picture is drawn; she keeps the one she thinks best")
    drm.add_argument("--enlarge", type=float, metavar="TIMES", help="With 'images': make the picture she keeps this many times larger "
                     "with an upscaling model on the image server, e.g. 2 (0 = off)")
    drm.add_argument("--enlarge-model", help="With 'images': the upscaling model (comfyui: a file in models/upscale_models, default "
                     "RealESRGAN_x2plus.pth; a1111: an upscaler name)")
    drm.add_argument("--signature", choices=["remove", "keep"], help="With 'images': whether a signature or watermark on an image is "
                     "carried into dream pictures drawn from it (default: remove)")
    drm.add_argument("--redraw-below", type=int, metavar="SCORE", help="With 'images': a picture drawn from an image is drawn again, holding "
                     "less to the image, when her best attempt scores below this out of 10 (0 = never)")
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
    ref.add_argument("reflect_action", choices=["status", "on", "off", "now", "sort", "merge"], nargs="?", default="status")
    ref.add_argument("--apply", action="store_true", help="With 'sort' and 'merge': make the changes (without this, they are only shown)")
    ref.add_argument("--quick", action="store_true", help="With 'merge': only what the words settle; do not ask the model about rewordings")
    ref.add_argument("--restore", type=int, nargs="+", metavar="ID", help="With 'merge': bring back statements that were retired")
    ref.add_argument("--model", help="Ollama model that does the reflecting (with 'on')")
    ref.add_argument("--host", help="Ollama server for that model, if different from the embedding server (with 'on')")
    ref.add_argument("--depth", type=int, choices=[1, 2, 3],
                     help="1 facts and user profile only; 2 everything, unchecked; 3 everything, each item checked (default)")
    ref.add_argument("--dry-run", action="store_true", help="With 'now': show what would be stored, store nothing")
    ref.add_argument("--again", nargs="?", type=float, const=3.0, default=None, metavar="DAYS",
                     help="With 'now': go over the last DAYS days of conversation again (default 3), for something a pass missed")
    ref.add_argument("--think", action="store_true", help="With 'now': let the model reason first (slow; can run away on small models)")
    ref.add_argument("--no-think", action="store_true", help="With 'now': answer without reasoning first (the default)")
    im = subs.add_parser("images", help="Image memory: what she has been shown")
    im.add_argument("images_action", nargs="?", default="status",
                    choices=["status", "on", "off", "list", "show", "find", "labels", "add", "process", "look", "forget", "redo", "people", "signature", "dream", "removed", "restore", "delete", "fingerprints", "similar", "name", "names"])
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
    im.add_argument("--what", help="With 'name': what the named thing is, e.g. \"a long-haired black and white cat\"")
    im.add_argument("--not", dest="not_it", action="store_true", help="With 'name': the image does NOT show that thing")
    im.add_argument("--all", action="store_true", help="With 'similar': list every image with its figures, not only those alike enough")
    im.add_argument("--yes", action="store_true", help="With 'forget': actually remove")
    im.add_argument("--keep-files", action="store_true", help="With 'forget': leave the image files on disk")
    prof = subs.add_parser("profile", help="Show the profiles written by reflection")
    prof.add_argument("--history", action="store_true", help="Show earlier versions too")
    prof.add_argument("--set", nargs=2, metavar=("WHO", "TEXT"), help="Write a profile yourself: user, self or us")
    subparser.set_defaults(func=holonomic_command)
