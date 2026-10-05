import base64, io, json, sys, time, types

import pytest

from test_provider import HAVE_HERMES, make, tool
from conftest import track
from holonomic import HolonomicMemory, HashEmbedder

CFG = {"image_enabled": True, "image_model": "eyes"}


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
        assert "images: 1" in run("stats")
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
        cfg = {"image_enabled": True, "image_model": "gemma-eyes", "image_host": f"http://127.0.0.1:{server.server_port}/",
               "reflect_model": "other", "reflect_host": "http://127.0.0.1:9"}
        img = images.add_image(m, picture(), cfg)
        report = {}
        assert images.describe(m, cfg, img["id"], report=report)["labels"] == ["black cat", "cushion", "sofa", "window"]
        path, body = got[0]
        assert path == "/api/chat" and body["model"] == "gemma-eyes" and body["think"] is False and body["stream"] is False
        assert body["format"]["required"] == ["description", "labels", "text"] and body["options"]["num_predict"] == 700
        user = body["messages"][1]
        assert user["role"] == "user" and size_of(base64.b64decode(user["images"][0])) == (1024, 768) and "Look at this image." in user["content"]
        assert report["calls"][0]["step"] == "image"
        # with no image model named, the reflection model and its server are used
        assert images.image_config({"reflect_model": "other", "reflect_host": "http://h:1/"})["image_model"] == "other"
        assert images.image_config({"reflect_model": "other", "reflect_host": "http://h:1/"})["image_host"] == "http://h:1"
        assert images.image_config({"ollama_host": "http://o:2"})["image_host"] == "http://o:2"
    finally:
        server.shutdown()
