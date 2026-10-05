import base64, io, json, sys, time, types

import pytest

from test_provider import HAVE_HERMES, make, tool
from conftest import track
from holonomic import HolonomicMemory, HashEmbedder

CFG = {"image_enabled": True, "image_model": "eyes", "image_view_width": 1024, "image_view_height": 768}


def store(tmp_path):
    return track(HolonomicMemory(tmp_path / "m", HashEmbedder()))


def picture(width=1600, height=1200, fmt="PNG", colour=(200, 30, 30)):
    """A picture whose four quarters differ, so each part of it is a different image."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), colour)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width // 2, height // 2], fill=(20, 40, 200))
    draw.rectangle([width // 2, height // 2, width, height], fill=(30, 180, 60))
    out = io.BytesIO()
    img.save(out, fmt)
    return out.getvalue()


def size_of(jpeg):
    from PIL import Image
    return Image.open(io.BytesIO(jpeg)).size


WHOLE = {"description": "A photo of a black cat asleep on a grey sofa beside a window. Afternoon light falls across the cushions.",
         "labels": ["Black Cat", "sofa", "window", "cushion"], "text": ""}


def eyes(parts=None, whole=WHOLE, seen=None):
    """A stand-in for the vision model.  `parts` maps a place to what it shows; other places show nothing."""
    parts = parts or {}

    def see(step, system, prompt, jpeg, schema, max_tokens):
        if seen is not None:
            seen.append((step, prompt, jpeg))
        if step == "image":
            return json.dumps(whole)
        if step == "look":
            return json.dumps({"answer": "The cat is black."})
        place = prompt.split("the ", 1)[1].split(".", 1)[0]
        if place in parts:
            return json.dumps(dict({"notable": True}, **parts[place]))
        return json.dumps({"notable": False, "description": "", "labels": []})
    return see


def test_an_image_is_kept_once_with_a_standard_size_copy(tmp_path):
    from holonomic import images
    m = store(tmp_path)
    data = picture(1600, 1200)
    img = images.add_image(m, data, CFG, origin="C:/photos/cat.png", caption="This is Pixel.", session="s1")
    assert img["new"] and img["width"] == 1600 and img["origin"] == "cat.png" and img["caption"] == "This is Pixel."
    assert open(img["original"], "rb").read() == data                      # the original is kept exactly
    assert img["file"].endswith(".view.jpg") and size_of(open(img["file"], "rb").read()) == (1024, 768)
    assert img["sections_total"] == 9 and img["sections_waiting"] == 9 and img["description"] == ""
    again = images.add_image(m, data, CFG, session="s2")                   # the same file shown again
    assert not again["new"] and again["id"] == img["id"] and again["seen"] == 2 and images.count_images(m) == 1
    assert images.known(m, data)["id"] == img["id"] and images.known(m, picture(colour=(1, 2, 3))) is None
    tall = images.add_image(m, picture(900, 1800, "JPEG"), CFG)            # a portrait image keeps its shape
    assert size_of(open(tall["file"], "rb").read()) == (384, 768) or size_of(open(tall["file"], "rb").read())[1] == 1024
    small = images.add_image(m, picture(400, 300), CFG)                    # nothing to gain from cutting this up
    assert small["file"] == small["original"] and small["sections_total"] == 0
    with pytest.raises(images.ImageError):
        images.add_image(m, b"this is not a picture", CFG)
    # the default copy is larger, and a portrait photo keeps its shape inside the box turned on its side
    photo = images.add_image(m, picture(3000, 4000, "JPEG", colour=(4, 4, 4)), {"image_enabled": True})
    assert size_of(open(photo["file"], "rb").read()) == (1536, 2048) and size_of(open(photo["original"], "rb").read()) == (3000, 4000)
    images.forget_image(m, photo["id"], delete_files=True)
    with pytest.raises(images.ImageError):
        images.add_image(m, data, dict(CFG, image_max_bytes=1000))
    assert images.count_images(m) == 3


def test_sections_overlap_by_half(tmp_path):
    from holonomic import images
    grid = images.section_grid(3)
    assert len(grid) == 9 and grid[0] == (0, 0.0, 0.0, 0.5, 0.5, "top left") and grid[4][5] == "centre"
    assert grid[8][1:5] == (0.5, 0.5, 0.5, 0.5) and grid[5][5] == "middle right" and images.section_grid(1) == []
    assert [g[5] for g in images.section_grid(2)] == ["top left", "top right", "bottom left", "bottom right"]
    m = store(tmp_path)
    seen = []
    big = images.add_image(m, picture(4096, 3072, "JPEG"), CFG)
    images.describe(m, CFG, big["id"], see=eyes(seen=seen))
    assert size_of(seen[0][2]) == (1024, 768)                               # the whole image, at the standard size
    images.describe_sections(m, CFG, big["id"], see=eyes(seen=seen))
    assert len(seen) == 10 and {size_of(s[2]) for s in seen[1:]} == {(1024, 768)}     # cut from the original: 2048x1536, shrunk
    seen.clear()
    other = images.add_image(m, picture(4096, 3072, "JPEG", colour=(9, 9, 9)), CFG)
    images.describe(m, CFG, other["id"], see=eyes(seen=seen))
    images.describe_sections(m, dict(CFG, image_from_original=False), other["id"], see=eyes(seen=seen))
    assert {size_of(s[2]) for s in seen[1:]} == {(512, 384)}                # cut from the 1024x768 copy


def test_description_parts_and_labels(tmp_path):
    from holonomic import images
    from holonomic.reflect import pending as reflection_pending
    m = store(tmp_path)
    said = m.remember("This is Pixel asleep on the sofa", kind="said_user", session="s1")[0]
    before = reflection_pending(m)
    img = images.add_image(m, picture(), CFG, caption="This is Pixel.", session="s1", links=[said])
    assert images.pending(m) == {"images": 1, "sections": 9}
    seen = []
    parts = {"top left": {"description": "A window with a blue curtain, and a small spider plant on the sill.", "labels": ["window", "spider plant", "curtain"]},
             "centre": {"description": "The cat's face, eyes closed, one white whisker bent", "labels": ["cats", "whisker"]}}      # no full stop
    done = images.describe(m, CFG, img["id"], see=eyes(parts, seen=seen), key_fn=lambda t: ["Pixel"])
    assert "The person who showed it said: This is Pixel." in seen[0][1]
    assert done["description"].startswith("A photo of a black cat") and done["labels"] == ["black cat", "cushion", "sofa", "window"]
    whole = m.get(done["memory_id"])
    assert whole["kind"] == "image" and whole["meta"]["image_id"] == img["id"] and whole["session"] == "s1"
    assert said in [h.id for h in m.associates(done["memory_id"])]           # linked to what was said when it was shown
    assert [h.id for h in m.recall(keys=["Pixel"], k=3, min_score=0.0)][0] == done["memory_id"]      # and filed under the name given
    assert images.describe(m, CFG, img["id"], see=eyes()) is None             # already described: nothing to do
    assert images.describe_sections(m, CFG, img["id"], see=eyes(parts, seen=seen)) == 9
    assert "the top left" in seen[1][1] and "The whole image was described as: A photo of a black cat" in seen[1][1]
    full = images.get_image(m, img["id"], sections=True)
    described = {s["place"]: s["description"] for s in full["sections"] if s["description"]}
    assert set(described) == {"top left", "centre"} and full["sections_waiting"] == 0 and all(s["looked_at"] for s in full["sections"])
    assert described["centre"] == "The cat's face, eyes closed, one white whisker bent"       # a phrase with no full stop is kept
    part = m.get(next(s["memory_id"] for s in full["sections"] if s["place"] == "top left"))
    assert part["kind"] == "image_part" and part["meta"] == {"image_id": img["id"], "section": 0, "place": "top left"}
    assert m.stats()["memories"] == 4 and images.pending(m) == {"images": 0, "sections": 0}
    assert reflection_pending(m) == before                                   # reflection does not read what an image showed
    # every image in which a thing was noticed, by either the singular or the plural
    for word in ("cat", "cats", "Cat"):
        found = images.find_by_label(m, word)
        assert [f["id"] for f in found] == [img["id"]] and found[0]["matched"] == ["black cat", "cats"] and found[0]["matched_in"] == ["centre"]
    assert images.find_by_label(m, "spider plant")[0]["matched_in"] == ["top left"] and images.find_by_label(m, "dog") == []
    assert ("window", 1) in images.all_labels(m) and images.label_forms("berries")[:2] == ["berries", "berry"]
    assert images.look(m, CFG, img["id"], "What colour is the cat?", see=eyes(seen=seen)) == "The cat is black."
    assert size_of(seen[-1][2]) == (1024, 768)
    images.look(m, CFG, img["id"], "What is on the sill?", section=0, see=eyes(seen=seen))
    assert size_of(seen[-1][2]) == (800, 600)                                # the top left part of the 1600x1200 original
    assert sorted(images.memory_ids(m)) == sorted([done["memory_id"]] + [s["memory_id"] for s in full["sections"] if s["memory_id"]])
    assert images.memory_ids(m, described_before=0) == []


def test_a_description_that_fails_is_tried_again(tmp_path):
    from holonomic import images
    from holonomic.reflect import ReflectionError
    m = store(tmp_path)
    img = images.add_image(m, picture(), CFG)

    def off(*a):
        raise ReflectionError("Could not reach model 'eyes'")
    with pytest.raises(ReflectionError):
        images.describe(m, CFG, img["id"], see=off)
    with pytest.raises(images.ImageError):
        images.describe(m, CFG, img["id"], see=eyes(whole={"description": "A", "labels": [], "text": ""}))
    with pytest.raises(images.ImageError):
        images.describe(m, {"image_enabled": True}, img["id"])              # no model named
    report = images.process(m, CFG, see=off)
    assert report["errors"] == ["Could not reach model 'eyes'"] and report["described"] == [] and images.pending(m)["images"] == 1
    calls = []
    stop = lambda: len(calls) >= 4                                           # the conversation resumes after a few parts
    report = images.process(m, CFG, see=eyes({"centre": {"description": "A cat.  It is asleep.", "labels": ["cat"]}}, seen=calls), should_stop=stop)
    assert report["described"] == [img["id"]] and report["sections"] == 3 and images.pending(m) == {"images": 0, "sections": 6}
    cut = images.add_image(m, picture(colour=(5, 5, 5)), CFG)
    images.describe(m, CFG, cut["id"], see=eyes(whole={"description": "A photo of a red door. It has a brass knocker and a num", "labels": ["door"], "text": "No 7"}))
    assert images.get_image(m, cut["id"])["description"] == "A photo of a red door. Writing in the image: No 7"      # the cut-off sentence is dropped
    report = images.process(m, CFG, see=eyes())
    assert report["sections"] == 15 and images.pending(m) == {"images": 0, "sections": 0} and not report["errors"]
    assert images.process(m, dict(CFG, image_sections=False), see=off)["errors"] == []      # nothing waiting: the model is never called


def test_forgetting_an_image(tmp_path):
    from holonomic import images
    m = store(tmp_path)
    data = picture()
    img = images.add_image(m, data, CFG)
    images.process(m, CFG, see=eyes({"centre": {"description": "A cat asleep on a cushion.", "labels": ["cat"]}}))
    img = images.get_image(m, img["id"])
    assert m.stats()["memories"] == 2
    assert images.forget_image(m, img["id"]) and not images.forget_image(m, img["id"])
    assert m.stats()["memories"] == 0 and images.get_image(m, img["id"]) is None and images.find_by_label(m, "cat") == []
    assert open(img["original"], "rb").read() == data and images.count_images(m) == 0       # the file stays unless asked
    back = images.add_image(m, data, CFG)                                    # shown again after being forgotten: starts over
    assert back["new"] and back["id"] == img["id"] and back["seen"] == 1 and back["sections_waiting"] == 9
    images.forget_image(m, back["id"], delete_files=True)
    import os
    assert not os.path.exists(img["original"]) and not os.path.exists(img["file"])


def test_images_fade_with_sleep_but_stay_within_deep_recall(tmp_path):
    from holonomic import images
    from holonomic.sleep import sleep_once
    m = store(tmp_path)
    img = images.add_image(m, picture(), CFG)
    images.process(m, CFG, see=eyes({"centre": {"description": "A black cat asleep on a cushion.", "labels": ["cat"]}}))
    mid = images.get_image(m, img["id"])["memory_id"]
    now = time.time()
    sleep_once(m, {}, steps=["fade"], now=now + 60)                           # the first sleep starts the clock
    report = sleep_once(m, {}, steps=["fade"], now=now + 60 + 15 * 86400)     # three half-lives later
    assert report["fade"]["images"] == 2 and report["fade"]["changed"] == 2 and m.get(mid)["strength"] < 0.2
    assert m.recall("black cat asleep on a grey sofa", k=5, min_strength=0.35) == []
    assert mid in [h.id for h in m.recall("black cat asleep on a grey sofa", k=5, min_strength=0.0)]
    assert images.get_image(m, img["id"])["description"]                      # and the image itself is all still there


def test_images_in_a_message(tmp_path):
    from holonomic import images
    a, b = tmp_path / "upload_1.png", tmp_path / "gone.png"
    a.write_bytes(picture())
    text = f"[2 images] Look at these\n\n[Image attached at: {a}]\n[Image attached at: {b}]"
    assert images.attached_paths(text) == [str(a), str(b)]
    assert images.strip_image_markers(text) == "Look at these"
    assert images.strip_image_markers(f"What do you see in this image?\n\n[Image attached at: {a}]") == ""
    assert [origin for _, origin in images.images_in_turn(text)] == [str(a)]
    # the other form: Hermes had the picture described because it took the model for one that cannot see
    told = (f"[The user attached an image. Here's what it contains:\nA cat [black] on a sofa.\nIt is asleep.]\n"
            f"[If you need a closer look, use vision_analyze with image_url: {a}]\n\nThis is Pixel.")
    assert images.attached_paths(told) == [str(a)] and images.strip_image_markers(told) == "This is Pixel."
    failed = f"[The user attached an image but it couldn't be analyzed. You can try examining it with vision_analyze using image_url: {a}]"
    assert images.attached_paths(failed) == [str(a)] and images.strip_image_markers(failed) == ""
    url = "data:image/png;base64," + base64.b64encode(picture(colour=(7, 7, 7))).decode()
    messages = [{"role": "user", "content": [{"type": "text", "text": "old"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [{"type": "text", "text": "new"}, {"type": "image_url", "image_url": {"url": url}},
                                             {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]},
                {"role": "assistant", "content": "I see."}]
    got = images.images_in_turn(f"[1 image] new\n\n[Image attached at: {b}]", messages)       # the file is gone: use the message
    assert len(got) == 1 and got[0][0] == picture(colour=(7, 7, 7)) and got[0][1] == str(b)
    later = messages + [{"role": "user", "content": "no image here"}, {"role": "assistant", "content": "ok"}]
    assert images.images_in_turn("no image here", later) == []                # only the message just answered


def test_provider_keeps_recalls_and_shows_images(tmp_path):
    if not HAVE_HERMES: return
    from holonomic import images
    p = make(tmp_path)
    p._cfg.update(CFG)
    (tmp_path / "home" / "holonomic.json").write_text(json.dumps(dict({"embedder": "hash", "min_score": 0.3}, **CFG)))
    e = p._engine
    parts = {"top left": {"description": "A window with a blue curtain and a spider plant on the sill.", "labels": ["window", "spider plant"]}}
    keep = images._seer
    images._seer = lambda ic, report: eyes(parts)
    try:
        shot = tmp_path / "upload_1.png"
        shot.write_bytes(picture())
        message = f"[1 image] This is Pixel, my black cat, asleep on the sofa.\n\n[Image attached at: {shot}]"
        assert p.prefetch(message, session_id="s1") == ""                        # never seen before
        p.sync_turn(message, "Pixel looks very comfortable there. The afternoon light suits a black cat.", session_id="s1",
                    messages=[{"role": "user", "content": message}])
        kinds = [(r["kind"], r["text"]) for r in e.recent(10)]
        assert ("said_user", "This is Pixel, my black cat, asleep on the sofa.") in kinds     # Hermes' notes are not her words
        assert not any("Image attached" in t or "[1 image]" in t for _, t in kinds)
        img = images.list_images(e)[0]
        assert img["description"].startswith("A photo of a black cat") and img["session"] == "s1" and img["sections_waiting"] == 9
        assert "Images you are shown are kept (1 so far)" in p.system_prompt_block()
        reflector = __import__("holonomic.provider", fromlist=["reflector_for"]).reflector_for(e)
        assert reflector.images_if_due() is None                                 # mid-conversation: the parts wait
        reflector.last_activity = time.time() - 600
        report = reflector.images_if_due()                                       # quiet: now they are looked at
        assert report["sections"] == 9 and images.pending(e) == {"images": 0, "sections": 0}
        assert reflector.images_if_due() is None
        # recalled in a later conversation: one entry for the image, with its file and the part that matched
        block = p.prefetch("Do you remember the spider plant on the window sill with the blue curtain?", session_id="s2")
        line = next(l for l in block.splitlines() if "image #" in l)
        assert f"image #{img['id']} you were shown; file: {img['file']})" in line and "A photo of a black cat" in line
        assert block.count("image #") == 1 and "\n    in the top left [#" in block and "spider plant on the sill" in block
        # shown the same file again: she is told it is one she has seen
        again = p.prefetch(f"[1 image] Who is this?\n\n[Image attached at: {shot}]", session_id="s2")
        assert "## An image you have seen before" in again and f"image #{img['id']}, first shown" in again and "black cat" in again
        p.sync_turn(f"[1 image] Who is this?\n\n[Image attached at: {shot}]", "That is Pixel, asleep on the sofa again I think.", session_id="s2")
        assert images.count_images(e) == 1 and images.get_image(e, img["id"])["seen"] == 2
        # the tool
        by_label = tool(p, action="images", label="cats")
        assert by_label["count"] == 1 and by_label["images"][0]["file"] == img["file"] and "MEDIA:" in by_label["note"]
        assert by_label["images"][0]["things_in_it"] == ["black cat", "cushion", "sofa", "window", "spider plant"]
        assert by_label["images"][0]["said_when_shown"].startswith("This is Pixel") and by_label["images"][0]["times_shown"] == 2
        assert tool(p, action="images", label="window")["images"][0]["noticed_in"] == ["top left"]
        assert tool(p, action="images", label="giraffe")["count"] == 0
        one = tool(p, action="images", image_id=img["id"])
        assert [s["place"] for s in one["parts"]] == ["top left"] and one["size"] == "1600x1200"
        assert tool(p, action="images")["total"] == 1 and "error" in tool(p, action="images", image_id=99)
        searched = tool(p, action="images", query="a blue curtain and a spider plant on the sill")
        assert searched["images"][0]["image_id"] == img["id"] and searched["images"][0]["matching_parts"][0]["place"] == "top left"
        assert tool(p, action="look", image_id=img["id"], question="What colour is the cat?")["answer"] == "The cat is black."
        assert "error" in tool(p, action="look", image_id=img["id"])
        hit = tool(p, action="recall", query="black cat asleep on a grey sofa beside a window")["results"][0]
        assert hit["kind"] == "image" and hit["image_id"] == img["id"] and hit["file"] == img["file"]
        assert tool(p, action="stats")["images"] == 1
        assert tool(p, action="forget", memory_id=img["memory_id"]) == {"forgotten": True, "image_id": img["id"]}
        assert images.count_images(e) == 0 and open(img["original"], "rb").read() == picture()
    finally:
        images._seer = keep
        p.shutdown()


def test_provider_with_image_memory_off_or_the_model_unreachable(tmp_path):
    if not HAVE_HERMES: return
    from holonomic import images
    from holonomic.reflect import ReflectionError
    p = make(tmp_path)
    e = p._engine
    shot = tmp_path / "upload_1.png"
    shot.write_bytes(picture())
    message = f"[1 image] Here is the garden this morning.\n\n[Image attached at: {shot}]"
    p.sync_turn(message, "The garden looks lovely in that light, the tomatoes especially.", session_id="s1")
    assert images.count_images(e) == 0 and "Images you are shown" not in p.system_prompt_block()      # off: nothing is kept
    assert [r["text"] for r in e.recent(5) if r["kind"] == "said_user"] == ["Here is the garden this morning."]
    p._cfg.update(CFG)
    keep = images._seer

    def off(ic, report):
        def see(*a):
            raise ReflectionError("Could not reach model 'eyes'")
        return see
    images._seer = off
    try:
        p.sync_turn(message, "The garden looks lovely in that light, the tomatoes especially.", session_id="s1")
        assert images.count_images(e) == 1 and images.pending(e)["images"] == 1       # kept, and waiting to be described
        images._seer = lambda ic, report: eyes()
        assert images.process(e, p._cfg)["described"] == [1]
    finally:
        images._seer = keep
        p.shutdown()


def test_cli_images(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib
    from holonomic import images
    p = make(tmp_path)
    p._engine.remember("The tomato plants need water early tomorrow.", kind="said_user", session="s1")
    p.shutdown()
    home = tmp_path / "home"
    shot = tmp_path / "cat.png"
    shot.write_bytes(picture())
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
    keep_seer = images._seer
    images._seer = lambda ic, report: eyes({"centre": {"description": "The cat's face, eyes closed.", "labels": ["cat", "whisker"]}})
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
        assert "Image memory is OFF" in run("images") and "images on --model NAME" in run("images")
        assert "needs a model that can see" in run("images", "on")
        out = run("images", "on", "--model", "eyes", "--host", "http://10.0.0.5:11434/", "--sections", "now")
        assert "Image memory is ON (model eyes at http://10.0.0.5:11434; parts of each image: looked at straight away)" in out
        saved = json.loads((home / "holonomic.json").read_text())
        assert saved["image_enabled"] is True and saved["image_host"] == "http://10.0.0.5:11434" and saved["image_sections_when"] == "now"
        out = run("images", "add", str(shot), str(tmp_path / "missing.png"), "--say", "This is Pixel.")
        assert "kept and described" in out and "image #1" in out and "things in it: black cat, cushion, sofa, window" in out
        assert "missing.png: " in out and "9 part(s) waiting to be looked at" in out and "said when shown: This is Pixel." in out
        assert "she has seen this exact image before" in run("images", "add", str(shot))
        status = run("images")
        assert "images kept: 1" in status and "parts waiting to be looked at: 9" in status
        out = run("images", "process")
        assert "Described 0 image(s), looked at 9 part(s)" in out and "Nothing is waiting." in run("images", "process")
        assert "1 most recent image(s)" in run("images", "list") and "cat (1)" in run("images", "labels")
        out = run("images", "find", "cats")
        assert "1 image(s) in which 'cats' was noticed" in out and "noticed in the: centre" in out
        assert "No image in which 'giraffe' was noticed." in run("images", "find", "giraffe")
        out = run("images", "show", "1", "7")
        assert "part 4 (centre)" in out and "part 0 (top left): nothing notable" in out and "image #7: no such image" in out
        assert run("images", "look", "1", "What colour is the cat?").strip() == "The cat is black."
        out = run("images", "redo", "1", "9", "--fix", "The sofa is green.", "--say", "This is Pixel on the green sofa.")
        assert "image #1: described again" in out and "said when shown: This is Pixel on the green sofa." in out and "image #9: no such image" in out
        assert "9 part(s) waiting to be looked at again" in out and "looked at 9 part(s)" in run("images", "process")
        assert "Usage: hermes holonomic images redo" in run("images", "redo")
        assert "images: 1" in run("stats")
        assert "real people in it: no" in run("images", "show", "1") and "Dreams will not draw from it" in run("images", "people", "1", "yes")
        assert "real people in it: yes" in run("images", "show", "1") and "Usage:" in run("images", "people", "1", "maybe")
        assert "Nothing removed." in run("images", "forget", "1") and "images kept: 1" in run("images")
        assert "Forgot image #1 and deleted the files" in run("images", "forget", "1", "--yes")
        assert "images kept: 0" in run("images") and not list((home / "holonomic" / "images").glob("*"))
        assert "Image memory is OFF" in run("images", "off")
    finally:
        embed.OllamaEmbedder = keep
        images._seer = keep_seer
        sys.modules.pop("hermes_constants", None)


def test_the_vision_model_is_sent_the_picture(tmp_path):
    """Against a stand-in for Ollama: the picture goes in the user message, base64-encoded, with the reply format."""
    import http.server, threading
    from holonomic import images
    got = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            got.append((self.path, body))
            reply = json.dumps({"message": {"content": json.dumps(WHOLE)}, "done_reason": "stop", "eval_count": 40}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *a):
            pass
    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        m = store(tmp_path)
        cfg = {"image_view_width": 1024, "image_view_height": 768, "image_enabled": True, "image_model": "gemma-eyes", "image_host": f"http://127.0.0.1:{server.server_port}/",
               "reflect_model": "other", "reflect_host": "http://127.0.0.1:9"}
        img = images.add_image(m, picture(), cfg)
        report = {}
        assert images.describe(m, cfg, img["id"], report=report)["labels"] == ["black cat", "cushion", "sofa", "window"]
        path, body = got[0]
        assert path == "/api/chat" and body["model"] == "gemma-eyes" and body["think"] is False and body["stream"] is False
        assert body["format"]["required"] == ["description", "labels", "text", "people"] and body["options"]["num_predict"] == 700
        user = body["messages"][1]
        assert user["role"] == "user" and size_of(base64.b64decode(user["images"][0])) == (1024, 768) and "Look at this image." in user["content"]
        assert report["calls"][0]["step"] == "image"
        # with no image model named, the reflection model and its server are used
        assert images.image_config({"reflect_model": "other", "reflect_host": "http://h:1/"})["image_model"] == "other"
        assert images.image_config({"reflect_model": "other", "reflect_host": "http://h:1/"})["image_host"] == "http://h:1"
        assert images.image_config({"ollama_host": "http://o:2"})["image_host"] == "http://o:2"
    finally:
        server.shutdown()


def test_a_wrong_description_can_be_corrected(tmp_path):
    from holonomic import images
    m = store(tmp_path)
    img = images.add_image(m, picture(), CFG, caption="This is Sushi.", session="s1")
    lap = {"description": "A photo of a cat named Sushi sitting on a person's lap covered in denim.", "labels": ["cat", "denim pants", "whisker"], "text": ""}
    parts = {"centre": {"description": "A cat's face with long whiskers.", "labels": ["cats", "whiskers", "nose"]}}
    seen = []
    images.process(m, CFG, see=eyes(parts, whole=lap, seen=seen))
    assert "it can be mistaken about what things are" in seen[1][1]              # a part is not bound by the whole description
    first = images.get_image(m, img["id"], sections=True)
    assert first["labels"] == ["cat", "denim pants", "whisker", "nose"]          # 'cats' and 'whiskers' were already there in another form
    assert images.find_by_label(m, "whiskers")[0]["matched_in"] == ["centre"] and m.stats()["memories"] == 2
    couch = {"description": "A photo of a cat named Sushi sitting on a blue couch.", "labels": ["cat", "couch"], "text": ""}
    seen.clear()
    again = images.redescribe(m, CFG, img["id"], correction="That is a couch,  not someone's lap.", see=eyes(whole=couch, seen=seen),
                              key_fn=lambda t: ["Sushi"])
    assert "The person who showed it said: This is Sushi." in seen[0][1]
    assert "the person who showed it corrected it: That is a couch, not someone's lap." in seen[0][1]
    assert again["description"].endswith("sitting on a blue couch.") and again["labels"] == ["cat", "couch"] and again["sections_waiting"] == 9
    assert m.get(first["memory_id"]) is None and m.stats()["memories"] == 1 and images.find_by_label(m, "denim pants") == []
    assert images.process(m, CFG, see=eyes(parts))["sections"] == 9 and m.stats()["memories"] == 2
    renamed = images.redescribe(m, CFG, img["id"], caption="This is my daughter's cat, Sushi.", see=eyes(whole=couch, seen=seen))
    assert renamed["caption"] == "This is my daughter's cat, Sushi." and "corrected it: That is a couch" in seen[-1][1]     # the correction is kept
    assert images.redescribe(m, CFG, 99, see=eyes()) is None


def test_the_desktop_app_marks_an_image_its_own_way(tmp_path):
    from holonomic import images
    a, spaced = tmp_path / "IMG_0570_6e0c47.jpg", tmp_path / "my photo.jpg"
    a.write_bytes(picture(fmt="JPEG"))
    spaced.write_bytes(picture(fmt="JPEG", colour=(3, 3, 3)))
    text = f"This is me! @image:{a}"
    assert images.attached_paths(text) == [str(a)] and images.strip_image_markers(text) == "This is me!"
    assert images.attached_paths(f'@image:"{spaced}"\n@image:`{a}`\n\nTwo of them') == [str(spaced), str(a)]
    assert images.strip_image_markers(f'@image:"{spaced}"\n\nTwo of them [screenshot]') == "Two of them"
    assert [origin for _, origin in images.images_in_turn(text)] == [str(a)]       # read from the file: the original, with its name
    if not HAVE_HERMES: return
    p = make(tmp_path)
    p._cfg.update(CFG)
    keep = images._seer
    images._seer = lambda ic, report: eyes()
    try:
        p.sync_turn(text, "It is lovely to finally see you. I really like your glasses.", session_id="s1")
        img = images.list_images(p._engine)[0]
        assert img["caption"] == "This is me!" and img["origin"] == "IMG_0570_6e0c47.jpg"
        assert [r["text"] for r in p._engine.recent(9) if r["kind"] == "said_user"] == ["This is me!"]
        assert "## An image you have seen before" in p.prefetch(f"Who is this? @image:{a}", session_id="s2")
        fixed = tool(p, action="images", image_id=img["id"], correction="I am in my car, not a bus.")
        assert fixed["image_id"] == img["id"] and "looked at again" in fixed["note"]
    finally:
        images._seer = keep
        p.shutdown()


# ------------------------------------------------------------------ dreams

def dream_store(tmp_path, people=False):
    """A day of talk, two images seen today (one of a cat, one optionally with a person) and an older, related image."""
    from holonomic import images
    m = store(tmp_path)
    now = time.time()
    for i, text in enumerate(["The tomato plants in the garden finally have fruit this week",
                              "I also planted an assembly of herbs beside the tomato plants",
                              "Tomorrow I will water the garden early before work"]):
        m.remember(text, kind="said_user", session="s-today", created_at=now - 3600 + i)
    def show(colour, whole, when):
        img = images.add_image(m, picture(colour=colour), CFG, session="s-today")
        images.describe(m, CFG, img["id"], see=eyes(whole=whole))
        mid = images.get_image(m, img["id"])["memory_id"]
        m._db.execute("UPDATE memories SET created_at = ? WHERE id = ?", (when, mid))
        return img["id"]
    old = show((1, 1, 1), {"description": "A photo of a black cat asleep on a grey sofa in the afternoon sun.", "labels": ["cat", "sofa"], "text": "", "people": False}, now - 30 * 86400)
    cat = show((2, 2, 2), {"description": "A photo of a black cat asleep on a green couch beside a window.", "labels": ["cat", "couch"], "text": "", "people": False}, now - 1800)
    me = show((3, 3, 3), {"description": "A photo of a woman with glasses sitting in a parked car.", "labels": ["woman", "car"], "text": "", "people": people}, now - 1700)
    m.close(); m2 = track(HolonomicMemory(tmp_path / "m", HashEmbedder()))       # reload, so the changed dates are what is in memory
    return m2, {"old": old, "cat": cat, "me": me}, now


def dreamer(scenes=None, seen=None):
    def llm(system, user, step):
        if seen is not None:
            seen.setdefault(step.split(" ")[0], []).append(user)
        if step.startswith("dream"):
            return json.dumps({"dream": "I am in a greenhouse in winter and the cat on the green couch is asleep among the tomato plants, and the car windows fog over with basil."})
        if step.startswith("wake"):
            return json.dumps({"thoughts": "An odd one, mostly the garden and the cat.", "connections": []})
        return json.dumps({"scenes": scenes if scenes is not None else [
            {"picture": "A black cat asleep on a green couch inside a winter greenhouse, tomato vines growing over the cushions.", "images": []}]})
    return llm


def test_images_join_what_a_dream_is_made_from(tmp_path):
    import random
    from holonomic.sleep import gather_fragments, sleep_once, sleep_config
    m, ids, now = dream_store(tmp_path)
    assert sleep_config({})["dream_images"] == "words" and sleep_config({"dream_images": "nonsense"})["dream_images"] == "words"
    got = gather_fragments(m, {}, random.Random(1), now)
    pictures = [f for f in got if f["kind"] == "image"]
    assert {f["image_id"] for f in pictures if f["age"] == "RECENT"} == {ids["cat"], ids["me"]}        # what she saw lately
    assert [f["image_id"] for f in pictures if f["age"] == "OLDER"] == [ids["old"]]                    # and an older image it resembles
    assert not any(f["kind"] == "image_part" for f in got) and len([f for f in got if f["kind"] == "said_user"]) == 3
    assert [f for f in gather_fragments(m, {"dream_images": "off"}, random.Random(1), now) if f["kind"] == "image"] == []
    assert len([f for f in gather_fragments(m, {"dream_image_seeds": 1}, random.Random(1), now) if f["kind"] == "image" and f["age"] == "RECENT"]) == 1
    seen = {}
    report = sleep_once(m, {}, llm=dreamer(seen=seen), steps=["dream"], rng=random.Random(1), now=now)
    assert "- (I saw this in an image I was shown) A photo of a black cat asleep on a green couch beside a window." in seen["dream"][0]
    assert "(I saw this in an image I was shown): A photo of a woman with glasses" in seen["wake"][0]
    d = report["dreams"][0]
    assert d["pictures"] == [] and "scenes" not in seen and not report["errors"]             # words only: nothing is drawn
    assert sorted(f["image_id"] for f in d["fragments"] if f.get("image_id")) == sorted(ids.values())
    assert m.get(ids and __import__("holonomic.images", fromlist=["x"]).get_image(m, ids["old"])["memory_id"])["strength"] == 1.0     # nothing strengthened


def test_dream_pictures_from_words(tmp_path):
    import random
    from holonomic import images
    from holonomic.sleep import dreams, latest_dream, sleep_once
    m, ids, now = dream_store(tmp_path)
    cfg = dict(CFG, dream_images="pictures", dream_image_count=2, dream_image_style="soft light")
    painted, seen = [], {}

    def paint(prompt, start):
        painted.append((prompt, start))
        return picture(768, 512, colour=(len(painted) * 40, 10, 10))
    scenes = [{"picture": "A black cat asleep on a green couch inside a winter greenhouse.", "images": [ids["cat"]]},
              {"picture": "Car windows fogging over with basil leaves.", "images": []},
              {"picture": "A third moment that is over the limit of two.", "images": []}]
    dry = sleep_once(m, cfg, llm=dreamer(scenes, seen), steps=["dream"], rng=random.Random(1), now=now, dry_run=True, paint=paint)
    assert [p["scene"] for p in dry["dreams"][0]["pictures"]] == [s["picture"] for s in scenes[:2]] and painted == []
    assert "IMAGES:" not in seen["scenes"][0] and "Choose 2 moments" in seen["scenes"][0] and "DREAM:\nI am in a greenhouse" in seen["scenes"][0]
    report = sleep_once(m, cfg, llm=dreamer(scenes), steps=["dream"], rng=random.Random(1), now=now, paint=paint)
    d = report["dreams"][0]
    assert [p[0] for p in painted] == [scenes[0]["picture"] + ", soft light", scenes[1]["picture"] + ", soft light"]
    assert all(start is None for _, start in painted)                                # from words alone, even if she named an image
    assert [p["from"] for p in d["pictures"]] == [[], []] and all(open(p["file"], "rb").read() for p in d["pictures"])
    # dream pictures live in the dream realm: not among images she was shown, not waiting to be described
    assert images.count_images(m) == 3 and images.count_images(m, "dream") == 2 and len(images.list_images(m)) == 3
    assert images.pending(m)["images"] == 0 and images.process(m, cfg, see=eyes())["described"] == []
    assert [p["scene"] for p in dreams(m, 1)[0]["pictures"]] == [s["picture"] for s in scenes[:2]]
    assert len(latest_dream(m, cfg, now)["pictures"]) == 2 and images.dream_pictures(m, 999) == []

    def broken(prompt, start):
        raise RuntimeError("Could not reach the image server")
    report = sleep_once(m, cfg, llm=dreamer(scenes), steps=["dream"], rng=random.Random(2), now=now + 9, paint=broken)
    assert report["dreams"][0]["id"] and report["dreams"][0]["pictures"] == []       # the dream is kept without pictures
    assert report["errors"] == ["dream pictures: Could not reach the image server"]
    report = sleep_once(m, cfg, llm=dreamer(scenes), steps=["dream"], rng=random.Random(3), now=now + 20)       # no generator set
    assert report["dreams"][0]["id"] and "No image generator is set" in report["errors"][0]


def test_dream_pictures_drawn_from_images_she_has_seen(tmp_path):
    import random
    from holonomic import images
    from holonomic.sleep import sleep_once
    m, ids, now = dream_store(tmp_path, people=True)
    cfg = dict(CFG, dream_images="from_images", dream_image_count=2, dream_image_style="")
    painted, seen = [], {}

    def paint(prompt, start):
        painted.append((prompt, start))
        return picture(768, 512, colour=(len(painted) * 40, 200, 10))
    scenes = [{"picture": "A black cat asleep on a couch that is also a sofa, in two kinds of light.", "images": [ids["cat"], ids["old"], ids["me"]]},
              {"picture": "A woman with glasses in a parked car full of tomato plants.", "images": [ids["me"], 4242]}]
    report = sleep_once(m, cfg, llm=dreamer(scenes, seen), steps=["dream"], rng=random.Random(1), now=now, paint=paint)
    offered = seen["scenes"][0].split("IMAGES:")[1]
    assert f"[{ids['cat']}] A photo of a black cat asleep on a green couch" in offered and f"[{ids['old']}]" in offered
    assert f"[{ids['me']}]" not in offered                                           # an image with a real person in it is not offered
    first, second = report["dreams"][0]["pictures"]
    assert first["from"] == [ids["cat"], ids["old"]] and second["from"] == []        # at most two, and only ones that were offered
    assert size_of(painted[0][1]) == (768, 512) and painted[1][1] is None            # the first starts from the two laid over each other
    assert images.dream_pictures(m, report["dreams"][0]["id"])[0]["from"] == [ids["cat"], ids["old"]]
    before = open(images.get_image(m, ids["cat"])["original"], "rb").read()
    assert before == picture(colour=(2, 2, 2))                                       # the stored image is only read
    seen.clear(); painted.clear()
    sleep_once(m, dict(cfg, dream_image_use_people=True), llm=dreamer(scenes, seen), steps=["dream"], rng=random.Random(1), now=now + 9, paint=paint)
    assert f"[{ids['me']}]" in seen["scenes"][0].split("IMAGES:")[1] and painted[1][1] is not None     # allowed when the user says so
    # an image described before the model was asked about people: judged by what was noticed in it
    m._db.execute("UPDATE images SET meta = '{}' WHERE id = ?", (ids["me"],))
    assert images.has_people(m, ids["me"]) and not images.has_people(m, ids["cat"]) and not images.has_people(m, 999)
    assert images.blend(m, [999], (64, 64)) is None


def test_painters_speak_to_an_image_server(tmp_path):
    import http.server, threading
    from holonomic import paint
    got = []
    png = picture(64, 64)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            got.append((self.path, self.headers["Content-Type"], raw))
            if self.path.startswith("/sdapi"):
                reply = {"images": [base64.b64encode(png).decode()]}
            elif self.path == "/v1/images/edits":
                reply = {"data": [{"url": f"http://127.0.0.1:{self.server.server_port}/made.png"}]}
            else:
                reply = {"data": [{"b64_json": base64.b64encode(png).decode()}]}
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(reply).encode())

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(png)

        def log_message(self, *a):
            pass
    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{server.server_port}/"
    try:
        a = paint.make_painter({"dream_image_api": "a1111", "dream_image_host": host, "dream_image_model": "dreamshaper", "dream_image_steps": 20,
                                "dream_image_width": 640, "dream_image_height": 384, "dream_image_strength": 0.5, "dream_image_negative": "text"})
        assert a("a cat", None) == png and a("a cat", b"START") == png
        assert a("a cat", None, size=(384, 640)) == png and (json.loads(got[2][2])["width"], json.loads(got[2][2])["height"]) == (384, 640)
        del got[2]
        path, _, raw = got[0]; body = json.loads(raw)
        assert path == "/sdapi/v1/txt2img" and body["prompt"] == "a cat" and (body["width"], body["height"], body["steps"]) == (640, 384, 20)
        assert body["override_settings"] == {"sd_model_checkpoint": "dreamshaper"} and body["negative_prompt"] == "text" and "init_images" not in body
        path, _, raw = got[1]; body = json.loads(raw)
        assert path == "/sdapi/v1/img2img" and base64.b64decode(body["init_images"][0]) == b"START" and body["denoising_strength"] == 0.5
        o = paint.make_painter({"dream_image_api": "openai", "dream_image_host": host, "dream_image_model": "z-image"})
        assert o("a cat", None) == png and o("a cat", b"START") == png              # the second comes back as a link, which is fetched
        path, _, raw = got[2]; body = json.loads(raw)
        assert path == "/v1/images/generations" and body == {"prompt": "a cat", "size": "768x512", "n": 1, "response_format": "b64_json", "model": "z-image"}
        path, ctype, raw = got[3]
        assert path == "/v1/images/edits" and ctype.startswith("multipart/form-data; boundary=") and b"START" in raw and b'name="prompt"' in raw
    finally:
        server.shutdown()
    for bad in ({}, {"dream_image_api": "a1111"}, {"dream_image_api": "midjourney", "dream_image_host": host}):
        with pytest.raises(paint.PaintError):
            paint.make_painter(bad)
    with pytest.raises(paint.PaintError):
        paint.make_painter({"dream_image_api": "a1111", "dream_image_host": "http://127.0.0.1:9", "dream_image_timeout": 2})("a cat", None)


def test_dream_pictures_reach_the_agent_and_the_terminal(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, random
    from holonomic import images
    from holonomic.sleep import sleep_once
    p = make(tmp_path)
    p._cfg.update(CFG, dream_images="pictures", dream_image_count=1)
    e = p._engine
    for text in ["The tomato plants in the garden finally have fruit this week", "I also planted herbs beside the tomato plants",
                 "Tomorrow I will water the garden early before work"]:
        e.remember(text, kind="said_user", session="s0")
    sleep_once(e, p._cfg, llm=dreamer(), steps=["dream"], rng=random.Random(1), paint=lambda prompt, start: picture(768, 512))
    file = images.dream_pictures(e, e.recent(1, realm="dream", kind="dream")[0]["id"])[0]["file"]
    block = p.system_prompt_block()
    assert "Pictures of moments in this dream (made from the dream; not photographs of anything real)" in block and f"(file: {file})" in block
    assert f"(file: {file})" in p.prefetch("Tell me about the dream you had about the greenhouse", session_id="s9")
    told = tool(p, action="dreams")["dreams"][0]
    assert told["pictures_of_it"][0]["file"] == file and "MEDIA:" in told["to_show_a_picture"]
    shown = tmp_path / "pasted.png"
    shown.write_bytes(open(file, "rb").read() if file.endswith(".png") else picture(768, 512))
    if file.endswith(".png"):
        assert "a picture from one of your own dreams" in p.prefetch(f"What is this picture? @image:{shown}", session_id="s9")
    assert "Images you are shown are kept (0 so far)" in block                    # a dream picture is not an image she was shown
    p.shutdown()
    home = tmp_path / "home"
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
        assert f"Picture: A black cat asleep on a green couch" in run("dreams") and file in run("dreams")
        out = run("dreams", "images")
        assert "Images in dreams: words" in out and "Image memory is off" in out and "Restart Hermes" not in out
        out = run("dreams", "images", "pictures")
        assert "Images in dreams: pictures" in out and "image generator: NOT SET" in out
        out = run("dreams", "images", "from_images", "--api", "a1111", "--host", "http://10.0.0.21:7860/", "--size", "640x384", "--count", "2", "--people", "yes")
        assert "image generator: a1111 at http://10.0.0.21:7860" in out and "2 picture(s) per dream, 640x384" in out and "may be drawn from" in out
        saved = json.loads((home / "holonomic.json").read_text())
        assert saved["dream_images"] == "from_images" and saved["dream_image_use_people"] is True and saved["dream_image_width"] == 640
        assert "are not drawn from" in run("dreams", "images", "--people", "no")
        out = run("dreams", "images", "--strength", "7", "--style", "ink wash, mist")
        assert "may move from the images it starts from: 1.0" in out and "style added to every scene: ink wash, mist" in out
        out = run("dreams", "images", "--api", "comfyui", "--host", "http://127.0.0.1:9", "--test", "a cat asleep in a greenhouse")
        assert "The image generator did not produce a picture: Could not reach the image server" in out
        assert "Choose one of" in run("dreams", "images", "sometimes") and "WIDTHxHEIGHT" in run("dreams", "images", "--size", "big")
    finally:
        embed.OllamaEmbedder = keep
        sys.modules.pop("hermes_constants", None)


def test_painter_for_comfyui(tmp_path):
    import http.server, threading, urllib.parse
    from holonomic import paint
    got, png, polls = [], picture(64, 64), {"n": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def reply(self, data, raw=False):
            self.send_response(200); self.end_headers(); self.wfile.write(data if raw else json.dumps(data).encode())

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            got.append((self.path, raw))
            if self.path == "/upload/image":
                return self.reply({"name": "holonomic_start.png", "subfolder": "", "type": "input"})
            self.reply({"prompt_id": "job-1"} if b"refuse" not in raw else {"error": "bad graph"})

        def do_GET(self):
            got.append((self.path, b""))
            if self.path.startswith("/object_info"):
                return self.reply({"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["v1-5-pruned-emaonly-fp16.safetensors", "other.safetensors"]]}}}})
            if self.path.startswith("/history"):
                polls["n"] += 1
                if polls["n"] % 2:                       # not finished the first time it is asked
                    return self.reply({})
                return self.reply({"job-1": {"outputs": {"7": {"images": [{"filename": "holonomic_dream_00001_.png", "subfolder": "", "type": "output"}]}},
                                             "status": {"completed": True}}})
            self.reply(png, raw=True)

        def log_message(self, *a):
            pass
    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    sc = {"dream_image_api": "comfyui", "dream_image_host": f"http://127.0.0.1:{server.server_port}", "dream_image_width": 768,
          "dream_image_height": 512, "dream_image_strength": 0.55, "dream_image_poll_seconds": 0.01, "dream_image_negative": "text"}
    try:
        draw = paint.make_painter(sc)
        assert draw("a cat in a greenhouse", None) == png
        paths = [p for p, _ in got]
        assert paths[0] == "/object_info/CheckpointLoaderSimple" and paths[1] == "/prompt" and paths[2] == paths[3] == "/history/job-1"
        assert urllib.parse.parse_qs(paths[4].split("?")[1]) == {"filename": ["holonomic_dream_00001_.png"], "type": ["output"]}
        graph = json.loads(got[1][1])["prompt"]
        assert graph["1"]["inputs"]["ckpt_name"] == "v1-5-pruned-emaonly-fp16.safetensors"       # none named: the server's first
        assert graph["2"]["inputs"]["text"] == "a cat in a greenhouse" and graph["3"]["inputs"]["text"] == "text"
        assert graph["4"] == {"class_type": "EmptyLatentImage", "inputs": {"width": 768, "height": 512, "batch_size": 1}}
        assert graph["5"]["inputs"]["denoise"] == 1.0 and graph["5"]["inputs"]["steps"] == 20 and graph["7"]["class_type"] == "SaveImage"
        got.clear()
        assert paint.make_painter(dict(sc, dream_image_model="dreamy.safetensors", dream_image_steps=12))("a cat", b"START") == png
        assert [p for p, _ in got][:2] == ["/upload/image", "/prompt"] and b"START" in got[0][1]
        graph = json.loads(got[1][1])["prompt"]
        assert graph["1"]["inputs"]["ckpt_name"] == "dreamy.safetensors" and graph["8"]["inputs"]["image"] == "holonomic_start.png"
        assert graph["4"]["class_type"] == "VAEEncode" and graph["5"]["inputs"]["denoise"] == 0.55 and graph["5"]["inputs"]["steps"] == 12
        # Z-Image-Turbo, recognised by its name: three files, few steps, no negative prompt
        got.clear()
        assert paint.make_painter(dict(sc, dream_image_model="z_image_turbo_bf16.safetensors"))("a cat", None, size=(768, 1024)) == png
        graph = json.loads(got[0][1])["prompt"]
        assert graph["10"] == {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}}
        assert graph["11"]["inputs"] == {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"} and graph["12"]["inputs"]["vae_name"] == "ae.safetensors"
        assert graph["2"]["inputs"] == {"text": "a cat", "clip": ["11", 0]} and graph["3"]["class_type"] == "ConditioningZeroOut" and "1" not in graph
        assert graph["4"] == {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1024, "batch_size": 1}}
        k = graph["5"]["inputs"]
        assert (k["steps"], k["cfg"], k["sampler_name"], k["scheduler"], k["model"], k["denoise"]) == (9, 1.0, "euler", "simple", ["13", 0], 1.0)
        assert graph["6"]["inputs"]["vae"] == ["12", 0] and graph["13"]["inputs"] == {"model": ["10", 0], "shift": 3.0}
        got.clear()
        paint.make_painter(dict(sc, dream_image_family="zimage", dream_image_text_encoder="qwen_fp8.safetensors"))("a cat", b"START")
        graph = json.loads(got[1][1])["prompt"]
        assert graph["10"]["inputs"]["unet_name"] == "z_image_turbo_bf16.safetensors" and graph["11"]["inputs"]["clip_name"] == "qwen_fp8.safetensors"
        assert graph["4"] == {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["12", 0]}} and graph["5"]["inputs"]["denoise"] == 0.55
        with pytest.raises(paint.PaintError):
            paint.make_painter(dict(sc, dream_image_model="x"))("refuse this", None)
    finally:
        server.shutdown()


def test_a_picture_drawn_from_a_tall_image_is_tall(tmp_path):
    import random
    from holonomic import images
    from holonomic.sleep import sleep_config, sleep_once
    m, ids, now = dream_store(tmp_path)
    tall = images.add_image(m, picture(900, 1600, colour=(8, 8, 8)), CFG, session="s-today")
    images.describe(m, CFG, tall["id"], see=eyes(whole={"description": "A tall photo of a lighthouse on a cliff above the sea.", "labels": ["lighthouse"], "text": "", "people": False}))
    cfg = dict(CFG, dream_images="from_images", dream_image_count=3, dream_image_seeds=9)
    assert sleep_config({})["dream_image_strength"] == 0.75
    drawn = []

    def paint(prompt, start, size=None):
        drawn.append((size, size_of(start) if start else None))
        return picture(*(size or (768, 512)), colour=(len(drawn), 9, 9))
    scenes = [{"picture": "A lighthouse standing in a garden of tomato plants.", "images": [tall["id"]]},
              {"picture": "A black cat asleep on a couch in the lighthouse lamp room.", "images": [ids["cat"]]},
              {"picture": "A garden path with no image behind it at all.", "images": []}]
    report = sleep_once(m, cfg, llm=dreamer(scenes), steps=["dream"], rng=random.Random(1), now=now, paint=paint)
    assert drawn == [((512, 768), (512, 768)), ((768, 512), (768, 512)), ((768, 512), None)] and not report["errors"]
    assert size_of(open(report["dreams"][0]["pictures"][0]["file"], "rb").read()) == (512, 768)
    # set to draw tall pictures: a wide image turns the frame the other way
    drawn.clear()
    sleep_once(m, dict(cfg, dream_image_width=512, dream_image_height=768), llm=dreamer(scenes), steps=["dream"], rng=random.Random(1), now=now + 9, paint=paint)
    assert [d[0] for d in drawn] == [(512, 768), (768, 512), (512, 768)]


def test_a_cat_with_a_face_is_not_a_person(tmp_path):
    from holonomic import images
    m = store(tmp_path)
    img = images.add_image(m, picture(), CFG)
    images.process(m, CFG, see=eyes({"centre": {"description": "The cat's face and head, eyes half closed.", "labels": ["face", "head", "whisker"]},
                                     "top left": {"description": "A hand-knitted blanket over the arm of the couch.", "labels": ["blanket", "arm"]}}))
    m._db.execute("UPDATE images SET meta = '{}' WHERE id = ?", (img["id"],))        # described before the model was asked about people
    assert not images.has_people(m, img["id"]) and images.get_image(m, img["id"])["people"] is False
    assert images.set_people(m, img["id"], True) and images.has_people(m, img["id"]) and not images.set_people(m, 99, True)
    m._db.execute("UPDATE images SET meta = ? WHERE id = ?", (json.dumps({"people": True, "people_said": False}), img["id"]))
    assert not images.has_people(m, img["id"])                                       # what the user said wins over what the model judged
