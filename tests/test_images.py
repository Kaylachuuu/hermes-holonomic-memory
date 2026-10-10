from pathlib import Path
import base64, io, json, sys, time, types

import pytest

from test_provider import HAVE_HERMES, make, tool
from conftest import track
from holonomic import HolonomicMemory, HashEmbedder

CFG = {"image_enabled": True, "image_model": "eyes", "image_view_width": 1024, "image_view_height": 768, "dream_image_candidates": 1,
       "dream_image_whole": "off"}          # the moments only; the picture of the whole dream has a test of its own


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
    assert report["errors"] == ["image #1: Could not reach model 'eyes'"] and report["described"] == [] and images.pending(m)["images"] == 1
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
    """Forgetting sets an image aside and can be undone exactly; deleting works only on an image already set
    aside, so getting rid of one for good takes two steps."""
    import os
    from holonomic import images
    m = store(tmp_path)
    data = picture()
    img = images.add_image(m, data, CFG, caption="This is Sushi.", origin="IMG_2018.HEIC")
    images.process(m, CFG, see=eyes({"centre": {"description": "A cat asleep on a cushion.", "labels": ["cat"]}}))
    assert images.set_people(m, img["id"], False) and images.set_signature(m, img["id"], "bottom right")
    before = images.get_image(m, img["id"], sections=True)
    assert m.stats()["memories"] == 2 and not images.delete_image(m, img["id"])              # not set aside: not deleted
    assert os.path.exists(before["original"]) and images.count_images(m) == 1
    assert images.forget_image(m, img["id"]) and not images.forget_image(m, img["id"])
    assert m.stats()["memories"] == 0 and images.get_image(m, img["id"]) is None and images.find_by_label(m, "cat") == []
    assert images.all_labels(m) == [] and images.pending(m) == {"images": 0, "sections": 0} and images.count_images(m) == 0
    assert open(before["original"], "rb").read() == data                                     # nothing is destroyed
    aside = images.removed_images(m)
    assert [(a["id"], a["origin"], a["description"], a["parts"], a["files_present"]) for a in aside] == \
        [(img["id"], "IMG_2018.HEIC", before["description"], 1, True)]
    # restored: the same words, parts, labels, what was said and what was corrected
    back = images.restore_image(m, CFG, img["id"])
    again = images.get_image(m, img["id"], sections=True)
    same = lambda i: (i["description"], i["caption"], i["origin"], i["labels"], i["people"], i["signature"], i["sections_waiting"],
                      [(x["place"], x["description"], x["looked_at"]) for x in i["sections"]])
    assert back and same(again) == same(before) and m.stats()["memories"] == 2 and images.removed_images(m) == []
    assert [i["id"] for i in images.find_by_label(m, "cat")] == [img["id"]] and images.restore_image(m, CFG, img["id"]) is None
    part = next(x for x in again["sections"] if x["description"])
    assert m.get(part["memory_id"])["meta"] == {"image_id": img["id"], "section": part["section"], "place": "centre"}
    assert abs(m.get(again["memory_id"])["created_at"] - m.get(part["memory_id"])["created_at"]) < 60    # not made to look new
    assert [h.id for h in m.recall("a cat asleep on a cushion", k=3)][:1] in ([again["memory_id"]], [part["memory_id"]])
    # shown again while set aside: she starts over with it
    images.forget_image(m, img["id"])
    fresh = images.add_image(m, data, CFG)
    assert fresh["new"] and fresh["id"] == img["id"] and fresh["seen"] == 1 and fresh["sections_waiting"] == 9 and fresh["labels"] == []
    # deleted for good: only after being set aside, and then nothing is left
    images.forget_image(m, img["id"])
    assert images.delete_image(m, img["id"]) and not images.delete_image(m, img["id"])
    assert not os.path.exists(before["original"]) and not os.path.exists(before["file"])
    assert images.removed_images(m) == [] and images.restore_image(m, CFG, img["id"]) is None
    row = images._row(m, img["id"])
    assert (row["caption"], row["origin"], row["meta"]) == ("", "", "{}")
    assert images.add_image(m, data, CFG)["new"]                                             # and it can be shown afresh


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
        gone = tool(p, action="forget", memory_id=img["memory_id"])
        assert (gone["forgotten"], gone["image_id"]) == (True, img["id"]) and "restore=true" in gone["note"]
        assert images.count_images(e) == 0 and open(img["original"], "rb").read() == picture()
        back = tool(p, action="images", image_id=img["id"], restore=True)                    # a mistake she can undo herself
        assert back["image_id"] == img["id"] and images.count_images(e) == 1 and "error" in tool(p, action="images", image_id=img["id"], restore=True)
        assert tool(p, action="recall", query="black cat asleep on a grey sofa beside a window")["results"][0]["image_id"] == img["id"]
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
        assert "never drawn from it" in run("images", "dream", "1", "no") and "dream pictures drawn from it: never" in run("images", "show", "1")
        out = run("images", "dream", "1", "9", "yes")
        assert "image #1: dream pictures may be drawn from it" in out and "image #9: no such image" in out
        assert "dream pictures drawn from it: always allowed" in run("images", "show", "1")
        assert "back to the general rule" in run("images", "dream", "1", "default") and "Usage:" in run("images", "dream", "1", "perhaps")
        assert "has not been set aside, so it is not deleted" in run("images", "delete", "1", "--yes") and "images kept: 1" in run("images")
        assert "No images are set aside." in run("images", "removed")
        out = run("images", "forget", "1")
        assert "Set aside image #1" in out and "nothing has been deleted" in out and "images restore 1" in out and "images delete 1" in out
        assert "images kept: 0" in run("images") and list((home / "holonomic" / "images").glob("*"))
        assert "image #1" in run("images", "removed") and "1 image(s) set aside" in run("images", "removed")
        assert "Restored image #1 with her description" in run("images", "restore", "1") and "images kept: 1" in run("images")
        assert "nothing to restore" in run("images", "restore", "1")
        run("images", "forget", "1")
        assert "Nothing deleted." in run("images", "delete", "1") and list((home / "holonomic" / "images").glob("*"))
        assert "Deleted image #1 and the files" in run("images", "delete", "1", "--yes")
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
        assert body["format"]["required"] == ["description", "labels", "text", "people"] and body["options"]["num_predict"] == 1000
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
        if step.startswith("whole"):
            return json.dumps({"picture": "A winter greenhouse holding a green couch, a sleeping cat and a fogged car all at once."})
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
    # what she remembers of the images a picture starts from is added to the scene; a picture from words alone gets the scene only
    assert painted[0][0] == ("A black cat asleep on a couch that is also a sofa, in two kinds of light. As remembered: A photo of a black cat "
                             "asleep on a green couch beside a window. A photo of a black cat asleep on a grey sofa in the afternoon sun")
    assert painted[1][0] == "A woman with glasses in a parked car full of tomato plants."
    assert size_of(painted[0][1]) == (768, 512) and painted[1][1] is None            # the first starts from the two laid over each other
    assert images.dream_pictures(m, report["dreams"][0]["id"])[0]["from"] == [ids["cat"], ids["old"]]
    before = open(images.get_image(m, ids["cat"])["original"], "rb").read()
    assert before == picture(colour=(2, 2, 2))                                       # the stored image is only read
    painted.clear()
    sleep_once(m, dict(cfg, dream_image_describe_source=False, dream_image_style="soft light"), llm=dreamer(scenes), steps=["dream"],
               rng=random.Random(1), now=now + 5, paint=paint)
    assert painted[0][0] == "A black cat asleep on a couch that is also a sofa, in two kinds of light., soft light"
    sleep_once(m, dict(cfg, dream_image_style="soft light"), llm=dreamer(scenes), steps=["dream"], rng=random.Random(1), now=now + 7, paint=paint)
    assert painted[2][0].endswith("asleep on a grey sofa in the afternoon sun. soft light") and painted[3][0].endswith("tomato plants., soft light")
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
        assert "dream-images" in file and "already in their own folders" in run("dreams", "sort")
        assert "dream pictures are in:" in run("images")
        out = run("dreams", "images")
        assert "Images in dreams: words" in out and "Image memory is off" in out and "Restart Hermes" not in out
        out = run("dreams", "images", "pictures")
        assert "Images in dreams: pictures" in out and "image generator: NOT SET" in out
        out = run("dreams", "images", "from_images", "--api", "a1111", "--host", "http://10.0.0.21:7860/", "--size", "640x384", "--count", "2", "--people", "yes")
        assert "image generator: a1111 at http://10.0.0.21:7860" in out and "2 picture(s) per dream, 640x384" in out and "may be drawn from" in out
        saved = json.loads((home / "holonomic.json").read_text())
        assert saved["dream_images"] == "from_images" and saved["dream_image_use_people"] is True and saved["dream_image_width"] == 640
        assert "are not drawn from" in run("dreams", "images", "--people", "no")
        assert "reasoning before she chooses" in run("dreams", "images", "--think", "on") and "reasoning before" not in run("dreams", "images", "--think", "off")
        out = run("dreams", "images", "--attempts", "4", "--swap", "on")
        assert "each drawn 4 time(s) and she keeps the best" in out and "language models are unloaded while pictures are drawn" in out
        assert "language models are unloaded" not in run("dreams", "images", "--swap", "off", "--attempts", "1")
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
        assert "device" not in graph["11"]["inputs"]
        got.clear()
        paint.make_painter(dict(sc, dream_image_family="zimage", dream_image_text_encoder_on="cpu"))("a cat", None)
        assert json.loads(got[0][1])["prompt"]["11"]["inputs"]["device"] == "cpu"
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


def test_each_picture_is_drawn_several_times_and_she_keeps_one(tmp_path):
    import random
    from holonomic import images
    from holonomic.sleep import sleep_config, sleep_once
    m, ids, now = dream_store(tmp_path)
    assert sleep_config({})["dream_image_candidates"] == 3 and sleep_config({})["dream_image_swap"] is False
    cfg = dict(CFG, dream_images="from_images", dream_image_count=2, dream_image_candidates=3, dream_image_style="")
    drawn, asked = [], []

    def paint(prompt, start, size=None):
        drawn.append(prompt)
        return picture(768, 512, colour=(len(drawn) * 20, 0, 0))                # every attempt is a different picture
    keep = images._chooser

    def chooser(ic, report):
        def look_at(step, prompt, jpegs, schema, max_tokens, think=False):
            if step == "what to look for":                                      # no list: she is asked the open question
                return json.dumps({"checks": []})
            asked.append((step, prompt, [size_of(j) for j in jpegs]))
            if step != "choose":                                                # one attempt at a time, each with its own notes
                n = int(step.split()[-1])
                report.setdefault("calls", []).append({"step": step})
                return json.dumps({"shows": f"A cat on couch number {n}.", "faults": "The cat has five legs." if n == 1 else "", "score": 4 + n})
            return json.dumps({"best": 2 if len([a for a in asked if a[0] == "choose"]) == 1 else 7, "why": "The couch is green and the cat is long-haired"})
        return look_at
    images._chooser = chooser
    scenes = [{"picture": "A black cat asleep on a couch in a greenhouse.", "images": [ids["cat"]]},
              {"picture": "Car windows fogging over with basil leaves.", "images": []}]
    try:
        report = sleep_once(m, cfg, llm=dreamer(scenes), steps=["dream"], rng=random.Random(1), now=now, paint=paint)
    finally:
        images._chooser = keep
    first, second = report["dreams"][0]["pictures"]
    assert len(drawn) == 6 and drawn[0] == drawn[1] == drawn[2] and drawn[3] == drawn[5]      # all drawn before any is chosen
    looks, chose = [a for a in asked if a[0] != "choose"], [a for a in asked if a[0] == "choose"]
    assert [a[0] for a in asked[:4]] == ["look at attempt 1", "look at attempt 2", "look at attempt 3", "choose"]
    assert all(len(a[2]) == 1 for a in looks) and looks[0][2] == [(768, 512)] and all(a[2] == [] for a in chose)    # never several pictures at once
    assert "A black cat asleep on a couch in a greenhouse." in looks[0][1] and "A black cat asleep on a couch in a greenhouse." in chose[0][1]
    assert "It draws on something you remember: A photo of a black cat asleep on a green couch" in looks[0][1]
    assert "It draws on something you remember" not in looks[3][1] and "It draws on something you remember" not in chose[1][1]
    assert "Attempt 1 (score 5 of 10): A cat on couch number 1. Faults: The cat has five legs." in chose[0][1]
    assert "Attempt 3 (score 7 of 10): A cat on couch number 3. Faults: none seen." in chose[0][1]
    assert [c["noted"] for c in report["calls"] if c.get("noted")][0] == "5 of 10. A cat on couch number 1. Faults: The cat has five legs."
    assert [c["step"] for c in report["calls"] if c.get("drawing")] == ["draw 3 attempt(s)"] * 2
    assert (first["chosen"], first["of"], first["why"]) == (2, 3, "The couch is green and the cat is long-haired")
    assert open(first["file"], "rb").read() == picture(768, 512, colour=(40, 0, 0))            # the second attempt is the one kept
    assert (second["chosen"], second["of"]) == (3, 3) and "scored highest" in second["why"]    # an answer that makes no sense: the best-scored
    assert images.count_images(m, "dream") == 2                                                # the others are not kept
    assert images.pick_best(cfg, "x", [picture()]) == (0, "") and images.pick_best({"image_enabled": True}, "x", [picture(), picture()]) == (0, "")


def test_one_graphics_card_is_shared_and_given_back(tmp_path):
    """Against stand-ins for Ollama and ComfyUI: the language model leaves before pictures are drawn and is back,
    loaded the way it was, before she is asked to choose between them."""
    import http.server, random, threading
    from holonomic import images, paint
    from holonomic.sleep import sleep_once
    log, loaded = [], {"gemma4-64k:latest": "2318-01-01T00:00:00Z", "nomic-embed-text:latest": "2318-01-01T00:00:00Z", "tiny:latest": "2026-10-05T12:00:00Z"}

    class Handler(http.server.BaseHTTPRequestHandler):
        def reply(self, data):
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(data).encode())

        def do_GET(self):
            log.append("ps")
            self.reply({"models": [{"name": k, "expires_at": v} for k, v in loaded.items()]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            if self.path == "/free":
                log.append(("free", body))
            elif body.get("keep_alive") == 0:
                log.append(("unload", body["model"])); loaded.pop(body["model"], None)
            else:
                log.append(("load", body["model"], body.get("keep_alive")))
                loaded[body["model"]] = "2318-01-01T00:00:00Z" if body.get("keep_alive") == -1 else "2026-10-05T12:00:00Z"
            self.reply({})

        def log_message(self, *a):
            pass
    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{server.server_port}"
    m, ids, now = dream_store(tmp_path)
    cfg = dict(CFG, dream_images="pictures", dream_image_count=1, dream_image_candidates=2, dream_image_swap=True, ollama_host=host,
               reflect_host=host + "/", dream_image_api="comfyui", dream_image_host=host, embed_model="nomic-embed-text")
    keep = images._chooser

    def chooser(ic, report):
        def look_at(step, prompt, jpegs, schema, max_tokens, think=False):
            if step != "choose":
                return json.dumps({"shows": "A picture.", "faults": "", "score": 6})
            log.append("choose")
            return json.dumps({"best": 1, "why": "It is the clearer one."})
        return look_at
    images._chooser = chooser

    def draw(prompt, start, size=None):
        log.append("draw")
        return picture(768, 512, colour=(len(log), 0, 0))
    try:
        report = sleep_once(m, cfg, llm=dreamer(), steps=["dream"], rng=random.Random(1), now=now, paint=draw)
        assert log == ["ps", ("unload", "gemma4-64k:latest"), ("unload", "tiny:latest"), "draw", "draw",
                       ("free", {"unload_models": True, "free_memory": True}),
                       ("load", "gemma4-64k:latest", -1), ("load", "tiny:latest", None), "choose"]      # the embedding model never left
        assert report["dreams"][0]["pictures"][0]["of"] == 2 and not report["errors"]
        log.clear()

        def broken(prompt, start, size=None):
            raise RuntimeError("out of memory")
        report = sleep_once(m, cfg, llm=dreamer(), steps=["dream"], rng=random.Random(2), now=now + 9, paint=broken)
        assert ("load", "gemma4-64k:latest", -1) in log and report["errors"] == ["dream pictures: out of memory"]      # she comes back even so
        assert paint.release({"dream_image_api": "openai", "dream_image_host": host}) is False
    finally:
        images._chooser = keep
        server.shutdown()
    m2, _, _ = dream_store(tmp_path / "b")
    report = sleep_once(m2, dict(cfg, ollama_host="http://127.0.0.1:9", reflect_host="", dream_image_host="http://127.0.0.1:9"),
                        llm=dreamer(), steps=["dream"], rng=random.Random(1), now=now, paint=draw)
    assert report["dreams"][0]["pictures"] and "could not make room" in report["errors"][0]            # no server to ask: the pictures are still drawn


def test_a_low_scoring_picture_from_an_image_is_drawn_again_more_freely(tmp_path):
    """The best attempt at a picture drawn from an image scores low: it is drawn again holding less to the image,
    and the new best is kept only if she scores it higher.  A picture drawn from nothing is not drawn again."""
    import random
    from holonomic import images
    from holonomic.sleep import sleep_once, sleep_config
    assert sleep_config({})["dream_image_redraw_below"] == 9 and sleep_config({})["dream_image_redraw_strength"] == 0.85
    m, ids, now = dream_store(tmp_path)
    cfg = dict(CFG, dream_images="from_images", dream_image_count=2, dream_image_candidates=2, dream_image_style="", dream_image_strength=0.7)
    scenes = [{"picture": "A black cat asleep on a couch at the bottom of a lake.", "images": [ids["cat"]]},
              {"picture": "Car windows fogging over with basil leaves.", "images": []}]
    keep = images._chooser

    compared = []

    def run(scores, painter=None, picks=(), **extra):
        drawn, looked, picks = [], [], list(picks)
        compared.clear()

        def paint(prompt, start, size=None, strength=None):
            drawn.append((bool(start), strength))
            return picture(768, 512, colour=(len(drawn) * 20, 0, 0))

        def chooser(ic, report):
            def look_at(step, prompt, jpegs, schema, max_tokens, think=False):
                if step == "what to look for":
                    return json.dumps({"checks": []})
                if step != "choose":
                    looked.append(step)
                    return json.dumps({"shows": "A cat.", "faults": "", "score": scores[len(looked) - 1]})
                compared.append(prompt)
                return json.dumps({"best": picks.pop(0) if picks else 1, "why": "The first."})
            return look_at
        images._chooser = chooser
        try:
            store, _, when = dream_store(tmp_path / f"run{len(list(tmp_path.iterdir()))}")
            report = sleep_once(store, dict(cfg, **extra), llm=dreamer(scenes), steps=["dream"], rng=random.Random(1), now=when,
                                paint=painter or paint)
        finally:
            images._chooser = keep
        return report, drawn, looked

    # cat: 6 and 7, she chooses the first.  basil: 5 and 5, from no image, never drawn again.  cat again: 9 and 8,
    # compared with the one she had; she takes the first of the new ones.
    report, drawn, looked = run([6, 7, 5, 5, 9, 8])
    cat, basil = report["dreams"][0]["pictures"]
    assert drawn == [(True, None)] * 2 + [(False, None)] * 2 + [(True, 0.85)] * 2 and len(looked) == 6
    assert "Attempt 3 (the one you chose from the first drawing) (score 6 of 10)" in compared[2] and "from 1 to 3" in compared[2]
    assert "the one you chose" not in compared[0]
    assert cat["redrawn"] == {"before": 6, "after": 9, "kept": True, "why": "The first."} and (cat["chosen"], cat["of"], cat["score"]) == (1, 2, 9)
    assert open(cat["file"], "rb").read() == picture(768, 512, colour=(100, 0, 0))             # the fifth picture drawn
    assert "redrawn" not in basil and basil["score"] == 5
    assert [c["step"] for c in report["calls"] if c.get("drawing")] == ["draw 2 attempt(s)"] * 2 + ["draw 2 attempt(s) again, holding less to the image"]
    # the new ones score higher, but side by side she prefers the one she had: it stays
    report, drawn, looked = run([8, 7, 9, 9, 9, 9], picks=[1, 1, 3])
    cat = report["dreams"][0]["pictures"][0]
    assert cat["redrawn"]["kept"] is False and cat["redrawn"]["after"] == 9 and (cat["chosen"], cat["score"]) == (1, 8)
    assert open(cat["file"], "rb").read() == picture(768, 512, colour=(20, 0, 0))
    # the comparison fails: the best score, and on a tie the one she already had
    report, drawn, looked = run([8, 7, 9, 9, 8, 6], picks=[1, 1, 99])
    cat = report["dreams"][0]["pictures"][0]
    assert cat["redrawn"]["kept"] is False and cat["score"] == 8 and open(cat["file"], "rb").read() == picture(768, 512, colour=(20, 0, 0))
    # good enough, switched off, already as free as the redraw, or a painter that cannot be asked: drawn once
    assert len(run([9, 9, 9, 9])[1]) == 4 and len(run([1, 1, 1, 1], dream_image_redraw_below=0)[1]) == 4
    assert len(run([1, 1, 1, 1], dream_image_strength=0.85)[1]) == 4
    plain = []
    assert len(run([1, 1, 1, 1], painter=lambda prompt, start, size=None: plain.append(1) or picture(768, 512))[2]) == 4 and len(plain) == 4


def test_a_signature_is_kept_out_of_dream_pictures(tmp_path):
    """A signed photo: the vision model is asked once which corner, that corner is smoothed over in what the
    image generator is given, and sentences about the signature are left out of what she is told she remembers."""
    import io, random
    from PIL import Image, ImageDraw
    from holonomic import images
    from holonomic.sleep import sleep_once, sleep_config
    assert sleep_config({})["dream_image_keep_signature"] is False
    assert images.without_signature("A lake at dusk. A signature is in the corner. Pines line the bank.") == "A lake at dusk. Pines line the bank."
    m, ids, now = dream_store(tmp_path)
    photo = Image.new("RGB", (1024, 768), (90, 110, 150))
    ImageDraw.Draw(photo).line([(940, 740), (1000, 748), (950, 752), (1005, 738)], fill=(10, 10, 10), width=3)      # a scrawl, bottom right
    row = images._row(m, ids["cat"])
    for rel in {row["file"], row["view"]}:
        photo.save(m.path / rel, "PNG")
    asked = []

    def see(step, system, prompt, jpeg, schema, max_tokens):
        asked.append(step)
        return json.dumps({"where": "bottom right"})
    assert images.signature_known(m, ids["cat"]) == ""
    assert images.signature_place(m, CFG, ids["cat"], see=see) == "bottom right" and images.signature_place(m, CFG, ids["cat"], see=see) == "bottom right"
    assert asked == ["signature"] and images.get_image(m, ids["cat"])["signature"] == "bottom right"      # asked once, then on record
    darkest = lambda data, box: min(Image.open(io.BytesIO(data)).convert("L").crop(box).getdata())
    corner, elsewhere = (900, 700, 1024, 768), (0, 0, 600, 600)
    plain, hidden = images.blend(m, [ids["cat"]], (1024, 768)), images.blend(m, [ids["cat"]], (1024, 768), {ids["cat"]: "bottom right"})
    assert darkest(plain, corner) < 40 and darkest(hidden, corner) > 90                              # the strokes are gone
    assert Image.open(io.BytesIO(hidden)).crop(elsewhere).tobytes() == Image.open(io.BytesIO(plain)).crop(elsewhere).tobytes()
    # in a dream: the painter is given the smoothed image, and no word of a signature
    given = []

    def paint(prompt, start, size=None):
        given.append((prompt, start))
        return picture(768, 512)
    cfg = dict(CFG, dream_images="from_images", dream_image_count=1, dream_image_candidates=1, dream_image_style="",
               dream_image_width=1024, dream_image_height=768, dream_image_redraw_below=0)
    scenes = [{"picture": "A black cat asleep on a couch in a greenhouse.", "images": [ids["cat"]]}]
    sleep_once(m, cfg, llm=dreamer(scenes), steps=["dream"], rng=random.Random(1), now=now, paint=paint)
    assert darkest(given[0][1], corner) > 90 and "ignature" not in given[0][0]
    # the user says it is not signed, or wants signatures kept: the image is given as it is
    assert images.set_signature(m, ids["cat"], "none") and not images.set_signature(m, ids["cat"], "middle")
    sleep_once(m, cfg, llm=dreamer(scenes), steps=["dream"], rng=random.Random(2), now=now + 90000, paint=paint)
    assert darkest(given[-1][1], corner) < 40
    assert images.set_signature(m, ids["cat"], "bottom-right")
    sleep_once(m, dict(cfg, dream_image_keep_signature=True), llm=dreamer(scenes), steps=["dream"], rng=random.Random(3), now=now + 180000, paint=paint)
    assert darkest(given[-1][1], corner) < 40 and len(given) == 3


def test_each_attempt_is_questioned_about_what_the_moment_needs(tmp_path):
    """She first lists what a picture of the moment must show, then answers for each attempt whether each thing is
    there, and about writing and bodies.  The score comes from those answers; a second drawing gets the same list."""
    from holonomic import images
    asked = []
    answers = [dict(q1="yes", q2="yes", q3="no", writing="none", bodies="None.", other="none", shows="A cat on a couch."),
               dict(q1="yes", q2="yes", q3="yes", writing="'Rcnonu' on a cushion", bodies="The cat has five legs.", other="none", shows="A cat under water."),
               dict(q1="yes", q2="unclear", q3="yes", writing="none", bodies="none", other="none", shows="A cat on a couch under water.")]
    keep = images._chooser

    def chooser(ic, report):
        def look_at(step, prompt, jpegs, schema, max_tokens, think=False):
            asked.append((step, prompt, len(jpegs), schema))
            report.setdefault("calls", []).append({"step": step})
            if step == "what to look for":
                return json.dumps({"checks": ["a long-haired black and white cat.", "a white stripe between the cat's eyes", "the couch is under water", "x"]})
            if step == "choose":
                return json.dumps({"best": 3, "why": "Only the third has the water and nothing wrong."})
            return json.dumps(answers[int(step.split()[-1]) - 1])
        return look_at
    images._chooser = chooser
    try:
        notes, checks, report = {}, [], {}
        three = [picture(colour=(i, 0, 0)) for i in range(3)]
        scene = "A cat with a white stripe between its eyes sits on a couch at the bottom of a lake."
        assert images.pick_best(CFG, scene, three, noted=notes, checks=checks, report=report) == (2, "Only the third has the water and nothing wrong.")
        assert checks == ["a long-haired black and white cat", "a white stripe between the cat's eyes", "the couch is under water"]
        assert [a[0] for a in asked] == ["what to look for", "look at attempt 1", "look at attempt 2", "look at attempt 3", "choose"]
        assert asked[0][2] == 0 and asked[1][2] == 1 and "- q2: can you see this in the picture: a white stripe between the cat's eyes?" in asked[1][1]
        assert asked[1][3]["properties"]["q3"]["enum"] == ["yes", "no", "unclear"] and "q4" not in asked[1][3]["properties"]
        assert "Do not answer from what the moment says should be there" in asked[1][1]
        # two of three there; all there but two things wrong; one unclear and nothing wrong
        assert [notes[i]["score"] for i in range(3)] == [7, 6, 8]
        assert notes[0]["missing"] == ["the couch is under water"] and notes[0]["faults"] == "" and "Missing: the couch is under water." in notes[0]["shows"]
        assert notes[1]["faults"] == "The cat has five legs. Writing: 'Rcnonu' on a cushion"
        assert "Could not tell: a white stripe between the cat's eyes." in notes[2]["shows"]
        assert report["calls"][0]["noted"].startswith("a long-haired black and white cat; a white stripe")
        assert "Attempt 2 (score 6 of 10)" in asked[4][1] and "Faults: The cat has five legs." in asked[4][1]
        # a second drawing: the same list, not asked for again
        asked.clear()
        images.pick_best(CFG, scene, three[:2], noted=notes, checks=checks, earlier={"shows": "x", "faults": "", "score": 8})
        assert [a[0] for a in asked] == ["look at attempt 1", "look at attempt 2", "choose"]
        # writing the moment asks for is not counted against the picture
        asked.clear()
        images.pick_best(CFG, "A neon sign that reads 'Bridgestone Arena' over a crowd.", three[:2], noted=notes, checks=["a neon sign"])
        assert notes[1]["faults"] == "The cat has five legs." and notes[1]["score"] == 8
    finally:
        images._chooser = keep


def test_the_picture_she_keeps_is_enlarged(tmp_path):
    """Only the kept picture is enlarged, after she has chosen it, to the size drawn times the factor.  If the
    image server cannot do it the picture stands as drawn and the dream is not lost."""
    import io, random
    from PIL import Image
    from holonomic import images, paint as painting
    from holonomic.sleep import sleep_once, sleep_config
    assert sleep_config({})["dream_image_enlarge"] == 0
    cfg = dict(CFG, dream_images="pictures", dream_image_count=2, dream_image_candidates=2, dream_image_style="", dream_image_enlarge=2,
               dream_image_width=768, dream_image_height=512)
    scenes = [{"picture": "A black cat asleep on a couch in a greenhouse.", "images": []},
              {"picture": "Car windows fogging over with basil leaves.", "images": []}]
    order = []

    def paint(prompt, start, size=None):
        order.append("draw")
        return picture(768, 512)

    def enlarge(data, size):
        order.append(("enlarge", size))
        return picture(*size)
    paint.enlarge = enlarge
    m, ids, now = dream_store(tmp_path / "a")
    report = sleep_once(m, cfg, llm=dreamer(scenes), steps=["dream"], rng=random.Random(1), now=now, paint=paint)
    made = report["dreams"][0]["pictures"]
    assert order == ["draw"] * 4 + [("enlarge", (1536, 1024))] * 2 and [p["enlarged"] for p in made] == [[1536, 1024]] * 2
    kept = images.get_image(m, made[0]["id"])                # the test's view size is smaller; the original is what was made
    assert (kept["width"], kept["height"]) == (1536, 1024) and Image.open(kept["original"]).size == (1536, 1024)
    assert [c["step"] for c in report["calls"] if c.get("drawing")][-2:] == ["enlarge the one she kept"] * 2

    def broken(data, size):
        raise painting.PaintError("no upscale model")
    paint.enlarge = broken
    m, ids, now = dream_store(tmp_path / "b")
    report = sleep_once(m, cfg, llm=dreamer(scenes), steps=["dream"], rng=random.Random(1), now=now, paint=paint)
    made = report["dreams"][0]["pictures"]
    assert len(made) == 2 and "enlarged" not in made[0] and images.get_image(m, made[0]["id"])["width"] == 768
    assert any("could not enlarge a picture (no upscale model)" in e for e in report["errors"])
    # the job given to ComfyUI, and where enlarging is not on offer
    sc = {"dream_image_api": "comfyui", "dream_image_host": "http://x", "dream_image_enlarge": 2}
    assert painting.make_enlarger(dict(sc, dream_image_enlarge=0)) is None and painting.make_enlarger(dict(sc, dream_image_api="openai")) is None
    seen = {}
    keep = painting._comfy_upload, painting._comfy_run
    painting._comfy_upload = lambda host, data, timeout: "up.png"
    painting._comfy_run = lambda host, graph, timeout, poll: seen.update(graph) or b"big"
    try:
        assert painting.make_enlarger(sc)(b"small", (2048, 1536)) == b"big"
    finally:
        painting._comfy_upload, painting._comfy_run = keep
    assert seen["2"] == {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x2plus.pth"}}
    assert seen["3"]["class_type"] == "ImageUpscaleWithModel" and (seen["4"]["inputs"]["width"], seen["4"]["inputs"]["height"]) == (2048, 1536)
    assert seen["1"]["inputs"]["image"] == "up.png"


def test_she_can_reason_before_choosing(tmp_path):
    """Against a stand-in for Ollama: reasoning is asked for with its own allowance; if it runs away she is asked
    again without it; if that fails too the first attempt is kept."""
    import http.server, threading
    from holonomic import images
    got, plan = [], []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "List the separate things" in body["messages"][1]["content"]:       # no list: the open question is asked
                reply = {"message": {"content": json.dumps({"checks": []})}, "done_reason": "stop"}
                self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(reply).encode())
                return
            got.append(body)
            reply = plan.pop(0)
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(reply).encode())

        def log_message(self, *a):
            pass
    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    answer = lambda n, why, **extra: {"message": dict({"content": json.dumps({"best": n, "why": why})}, **extra), "done_reason": "stop"}
    note = lambda score, faults="": {"message": {"content": json.dumps({"shows": "A cat in a boat.", "faults": faults, "score": score})}, "done_reason": "stop"}
    ran_away = {"message": {"content": "", "thinking": "Reading the first note again and again"}, "done_reason": "length", "eval_count": 3000}
    cfg = {"image_model": "gemma-eyes", "image_host": f"http://127.0.0.1:{server.server_port}", "dream_image_choose_think": True}
    three = [picture(colour=(i, 0, 0)) for i in range(3)]
    looks = lambda: [note(5, "No lake."), note(8), note(6)]
    try:
        plan[:] = looks() + [answer(3, "Only the third has the lake.", thinking="The first has no boat.  The second has no lake.")]
        report = {}
        assert images.pick_best(cfg, "A cat in a boat on a lake.", three, report=report) == (2, "Only the third has the lake.")
        assert [len(g["messages"][1].get("images", [])) for g in got] == [1, 1, 1, 0]             # one picture per look, none to compare
        assert [g["think"] for g in got] == [False, False, False, True] and got[3]["options"]["num_predict"] == 3000
        assert "Attempt 2 (score 8 of 10): A cat in a boat. Faults: none seen." in got[3]["messages"][1]["content"]
        assert report["calls"][4]["think"] is True and report["calls"][4]["thinking"] == "The first has no boat. The second has no lake."
        got.clear(); plan[:] = looks() + [ran_away, answer(1, "The first is clearest.")]
        report = {}
        assert images.pick_best(cfg, "A cat in a boat on a lake.", three, report=report) == (0, "The first is clearest.")
        assert [g["think"] for g in got[3:]] == [True, False] and got[4]["options"]["num_predict"] == 200
        failed = report["calls"][4]
        assert "hit the 3000-token reply limit" in failed["failed"] and failed["reply_tokens"] == 3000 and "seconds" in failed
        assert failed["thinking"].startswith("Reading the first note") and report["calls"][5]["think"] is False
        got.clear(); plan[:] = looks() + [ran_away, ran_away]                                     # no comparison: the best-scored attempt
        assert images.pick_best(cfg, "A cat in a boat on a lake.", three)[0] == 1
        got.clear(); plan[:] = [note(7), note(7), note(7), answer(9, "Nonsense.")]                # a tie: the earlier one
        assert images.pick_best(dict(cfg, dream_image_choose_think=False), "A cat.", three)[0] == 0 and [g["think"] for g in got] == [False] * 4
        got.clear(); plan[:] = [ran_away, note(4), ran_away]                                      # only one could be looked at: that one
        assert images.pick_best(cfg, "A cat.", three)[0] == 1 and len(got) == 3
        got.clear(); plan[:] = [ran_away, ran_away, ran_away]
        assert images.pick_best(cfg, "A cat.", three) == (0, "")
    finally:
        server.shutdown()


def test_dreaming_from_an_image_can_be_decided_image_by_image(tmp_path):
    import random
    from holonomic import images
    from holonomic.sleep import sleep_once
    m, ids, now = dream_store(tmp_path, people=True)
    assert images.may_dream_from(m, ids["cat"], False) and not images.may_dream_from(m, ids["me"], False) and images.may_dream_from(m, ids["me"], True)
    assert images.set_dream_use(m, ids["me"], True) and images.set_dream_use(m, ids["cat"], False) and not images.set_dream_use(m, 999, True)
    assert images.may_dream_from(m, ids["me"], False) and not images.may_dream_from(m, ids["cat"], True)       # what was said about the image decides
    assert images.get_image(m, ids["me"])["dream_from"] is True and images.get_image(m, ids["old"])["dream_from"] is None
    seen = {}
    cfg = dict(CFG, dream_images="from_images", dream_image_count=1, dream_image_seeds=9)
    sleep_once(m, cfg, llm=dreamer(seen=seen), steps=["dream"], rng=random.Random(1), now=now, paint=lambda p, s: picture(768, 512))
    offered = seen["scenes"][0].split("IMAGES:")[1]
    assert f"[{ids['me']}]" in offered and f"[{ids['cat']}]" not in offered and f"[{ids['old']}]" in offered
    assert "A photo of a black cat asleep on a green couch" in seen["dream"][0]        # still part of what the dream is made from
    images.set_dream_use(m, ids["me"], None)
    assert not images.may_dream_from(m, ids["me"], False)                             # back to the general rule
    assert images.has_people(m, ids["me"])                                            # saying it may be dreamt from does not change what is in it


def file_message(path, words="Here is another picture from Nashville. The city looks like a circuit board from above."):
    ref = f"@file:`Downloads/New Pictures/{path.name}`"
    return (f"{ref}\n\n{words}\n\n--- Attached Context ---\n\n📎 {ref} (image/heic, 2.5 MB) — binary file, not inlined as text. "
            f"It is available on disk at `{path}`. Use your tools to work with it (read or convert it, extract its text, or view/render it "
            "as needed); do not tell the user the file type is unsupported.")


def test_a_picture_attached_as_a_file_is_still_a_picture(tmp_path):
    from holonomic import images
    photo, notes = tmp_path / "IMG_9469.HEIC", tmp_path / "notes.pdf"
    photo.write_bytes(picture())                       # the name says HEIC; what matters is that it opens as a picture
    notes.write_bytes(b"%PDF-1.4")
    text = file_message(photo)
    assert images.attached_as_files(text) == [str(photo)] and images.attached_paths(text) == [str(photo)]
    assert images.strip_image_markers(text) == "Here is another picture from Nashville. The city looks like a circuit board from above."
    assert [origin for _, origin in images.images_in_turn(text)] == [str(photo)]
    other = file_message(notes, "Can you summarise this?")
    assert images.attached_as_files(other) == [] and images.strip_image_markers(other) == "@file:`Downloads/New Pictures/notes.pdf`\n\nCan you summarise this?"
    if not HAVE_HERMES: return
    p = make(tmp_path)
    p._cfg.update(CFG)
    keep = images._seer
    images._seer = lambda ic, report: eyes(whole={"description": "An aerial photo of a city at dusk, its roads lit like traces on a circuit board.",
                                                  "labels": ["city", "road"], "text": "", "people": False})
    try:
        block = p.prefetch(text, session_id="s1")          # she is told what the picture shows before she replies
        assert "## The picture attached to this message (you were not shown it directly" in block
        assert "image #1 (IMG_9469.HEIC; file: " in block and "An aerial photo of a city at dusk" in block
        img = images.list_images(p._engine)[0]
        assert img["caption"].startswith("Here is another picture from Nashville") and img["seen"] == 1 and img["session"] == "s1"
        p.sync_turn(text, "From up there the roads really do look like traces on a board.", session_id="s1")
        assert images.count_images(p._engine) == 1 and images.get_image(p._engine, img["id"])["seen"] == 1      # one showing, not two
        said = [r["text"] for r in p._engine.recent(20) if r["kind"] == "said_user"]
        assert sorted(said) == ["Here is another picture from Nashville.", "The city looks like a circuit board from above."]
        again = p.prefetch(text, session_id="s2")          # shown again later: recognised, not looked at afresh
        assert "## An image you have seen before" in again and "you were not shown it directly" not in again
        p.sync_turn(text, "That is the Nashville picture again, the one that looks like a circuit board.", session_id="s2")
        assert images.get_image(p._engine, img["id"])["seen"] == 2
        images._seer = lambda ic, report: (_ for _ in ()).throw(images.ImageError("the vision model is off"))
        other = tmp_path / "IMG_1039.heic"
        other.write_bytes(picture(colour=(9, 1, 1)))
        assert "it could not be looked at just now" in p.prefetch(file_message(other, "A sunset behind my old apartment."), session_id="s3")
        assert images.pending(p._engine)["images"] == 1                         # kept, and waiting to be described
    finally:
        images._seer = keep
        p.shutdown()


def test_cli_tidy(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib
    p = make(tmp_path)
    e = p._engine
    keepers = [e.remember(t, kind="said_user", session="s1")[0] for t in
               ["This was a sunset, taken out back of the apartment I used to live in.", "I love the brilliant orange sunlight on the clouds."]]
    junk = [e.remember(t, kind="said_user", session="s1")[0] for t in
            ["@file:`Downloads/New Pictures/IMG_1039.heic`", "--- Attached Context ---",
             "📎 @file:`Downloads/New Pictures/IMG_1039.heic` (image/heic, 637.0 KB) — binary file, not inlined as text.",
             "It is available on disk at `C:\\Users\\Kayla\\Downloads\\New Pictures\\IMG_1039.heic`.",
             "Use your tools to work with it (read or convert it, extract its text, or view/render it as needed); do not tell the user the file type is unsupported."]]
    after = e.remember("Use your tools to find out what the weather is like in Casper today.", kind="said_user", session="s1")[0]
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
        out = run("tidy")
        assert "5 stored message(s) are Hermes' notes" in out and "Nothing changed." in out and all(f"[#{j}]" in out for j in junk)
        assert not any(f"[#{k}]" in out for k in keepers + [after])
        assert "Removed." in run("tidy", "--apply") and "0 stored message(s)" in run("tidy")
        assert "brilliant orange" in run("show", str(keepers[1])) and "weather" in run("show", str(after))
    finally:
        embed.OllamaEmbedder = keep
        sys.modules.pop("hermes_constants", None)


def test_one_image_that_runs_long_does_not_hold_up_the_rest(tmp_path):
    from holonomic import images
    from holonomic.reflect import ReflectionError
    m = store(tmp_path)
    busy = images.add_image(m, picture(colour=(1, 1, 1)), dict(CFG, image_sections=False), caption="A street in Nashville at night.")
    calm = images.add_image(m, picture(colour=(2, 2, 2)), dict(CFG, image_sections=False))
    asked = []

    def see(step, system, prompt, jpeg, schema, max_tokens):
        asked.append((step, max_tokens, "Be brief this time" in prompt))
        if "Nashville" in prompt and "Be brief this time" not in prompt:
            raise ReflectionError("The model hit the 1000-token reply limit without finishing. Its reply began: '{'")
        return json.dumps(WHOLE)
    report = images.process(m, CFG, see=see)
    assert asked == [("image", 1000, False), ("image (again, briefly)", 2000, True), ("image", 1000, False)]      # asked again, shorter, with more room
    assert report["described"] == [busy["id"], calm["id"]] and report["errors"] == []

    def never(step, system, prompt, jpeg, schema, max_tokens):
        if "Nashville" in prompt:
            raise ReflectionError("The model hit the reply limit without finishing.")
        return json.dumps(WHOLE)
    for i in (busy["id"], calm["id"]):
        m._db.execute("UPDATE images SET memory_id = NULL, described_at = NULL WHERE id = ?", (i,))
    report = images.process(m, CFG, see=never)
    assert report["described"] == [calm["id"]] and len(report["errors"]) == 1 and report["errors"][0].startswith(f"image #{busy['id']}: ")
    assert images.pending(m)["images"] == 1                                # the one that failed waits; the other was not held up


def test_writing_is_kept_only_when_two_looks_agree(tmp_path):
    from holonomic import images
    m = store(tmp_path)
    said = m.remember("Here is another picture from Nashville", kind="said_user", session="s1")[0]
    img = images.add_image(m, picture(), CFG, caption="A street in Nashville at night.", session="s1", links=[said])
    whole = {"description": "A night photograph of a busy street with a 'Country Music Square' sign and a 'Hilton' building.",
             "labels": ["street", "country music square", "hilton sign"], "text": "Hilton, Country Music Square", "people": True}
    parts = {"top centre": {"description": "A building with a red 'Hilton' sign.", "labels": ["hilton sign"]},
             "top right": {"description": "A lower building with a 'Bridgestone Arena' sign and a 'T.J. Maxx' sign.", "labels": ["tj maxx sign", "arena sign"]},
             "centre": {"description": "A brick building with neon signs, including one that says 'TIGER BEER'.", "labels": ["tiger beer sign", "brick building"]},
             "middle right": {"description": "A tall sign for 'Bridgestone Arena' with a crowd beneath it.", "labels": ["crowd"]},
             "bottom centre": {"description": "A white SUV marked 'POLICE' beside a dark one. It's parked at the kerb.", "labels": ["suv"]}}
    seen = []
    report = images.process(m, CFG, see=eyes(parts, whole=whole, seen=seen))
    part_prompt = seen[1][1]
    assert "Country Music Square" not in part_prompt and "Hilton" not in part_prompt and "a busy street with a [writing] sign and a [writing] building." in part_prompt
    assert "The person who showed the image said this about the whole image, which is true: A street in Nashville at night." in part_prompt
    assert report["unclear"] == 4                                   # Country Music Square, T.J. Maxx, TIGER BEER and POLICE: one look each
    got = images.get_image(m, img["id"], sections=True)
    by_place = {s["place"]: s["description"] for s in got["sections"] if s["description"]}
    assert got["description"] == (f"A night photograph of a busy street with a {images.UNCLEAR} sign and a 'Hilton' building. "
                                  "Writing in the image: Hilton")
    assert by_place["top right"] == f"A lower building with a 'Bridgestone Arena' sign and a {images.UNCLEAR} sign."      # two parts read the arena sign
    assert by_place["centre"] == f"A brick building with neon signs, including one that says {images.UNCLEAR}."
    assert by_place["top centre"] == "A building with a red 'Hilton' sign." and "It's parked at the kerb." in by_place["bottom centre"]
    assert set(got["labels"]) == {"street", "hilton sign", "arena sign", "brick building", "crowd", "suv"}       # labels made from doubtful writing go too
    assert m.stats()["memories"] == 7 and said in [h.id for h in m.associates(got["memory_id"])]        # still linked to what was said
    assert got["memory_id"] in [h.id for h in m.associates(next(s["memory_id"] for s in got["sections"] if s["place"] == "centre"), k=3)]
    assert images.settle_writing(m, img["id"]) == [] and images.process(m, CFG, see=eyes())["unclear"] == 0      # done once
    # what the user states is never doubted, even if only one look reported it
    seen.clear()
    images.redescribe(m, CFG, img["id"], correction="The white SUV is marked POLICE. The big sign says Bridgestone Arena.", see=eyes(whole=whole))
    report = images.process(m, CFG, see=eyes(parts, seen=seen))
    assert "which is true: A street in Nashville at night. The white SUV is marked POLICE." in seen[0][1]
    # a denial is not a statement that the thing is there, and is not repeated to the parts
    assert images.what_was_stated(["The large sign says Bridgestone Arena, not Country Music Square. There is no Tiger Beer, Bud Light or Budweiser sign.",
                                   "There is a police vehicle at the corner."]) == (
        "The large sign says Bridgestone Arena, There is There is a police vehicle at the corner.",
        {"country", "music", "square", "tiger", "beer", "bud", "light", "budweiser"})
    tiger = dict(parts, centre={"description": "A brick building with neon signs, including one that says 'Tiger' and another with a 'C' logo.", "labels": ["tiger sign"]},
                 **{"middle left": {"description": "A street with a neon 'Tiger' sign above a doorway.", "labels": ["doorway"]}})
    seen.clear()
    images.redescribe(m, CFG, img["id"], correction="There is no Tiger Beer sign.", see=eyes(whole=whole))
    images.process(m, CFG, see=eyes(tiger, seen=seen))
    assert "Tiger" not in seen[0][1] and "The big sign says Bridgestone Arena." in seen[0][1]
    final = images.get_image(m, img["id"], sections=True)
    centre = next(s["description"] for s in final["sections"] if s["place"] == "centre")
    assert centre == f"A brick building with neon signs, including one that says {images.UNCLEAR} and another with a {images.UNCLEAR} logo."
    assert images.UNCLEAR in next(s["description"] for s in final["sections"] if s["place"] == "middle left")       # two looks agreed, but the user said no
    assert "tiger sign" not in final["labels"] and "doorway" in final["labels"]
    again = {s["place"]: s["description"] for s in images.get_image(m, img["id"], sections=True)["sections"] if s["description"]}
    assert "marked 'POLICE'" in again["bottom centre"] and images.UNCLEAR in again["centre"] and report["unclear"] == 3


# ---------------------------------------------------------------------------------------------- fingerprints

def colour_prints(pictures):
    """A stand-in for the fingerprint model: a picture's fingerprint is its average colour, centred so that
    different colours point in different directions."""
    from PIL import Image
    out = []
    for p in pictures:
        img = Image.open(io.BytesIO(p)).convert("RGB").resize((8, 8))
        px = list(img.getdata())
        out.append([sum(c[i] for c in px) / len(px) - 110.0 for i in range(3)])
    return "colours-1", out


def flat(colour, width=1600, height=1200):
    """One colour all over.  (`picture` has the same pale background in every image, and parts of two images
    that show the same background are, rightly, found to be alike.)"""
    return two_tone(colour, colour, width, height)


def two_tone(left, right, width=1600, height=1200):
    from PIL import Image
    img = Image.new("RGB", (width, height), left)
    img.paste(right, (width // 2, 0, width, height))
    out = io.BytesIO(); img.save(out, "PNG")
    return out.getvalue()


def test_fingerprints_find_images_that_look_alike(tmp_path):
    """Every image gets a fingerprint and so does each part.  Images are then found by what they look like:
    as whole pictures, or because a part of one looks like a part of another."""
    from holonomic import images, fingerprints as fp
    m = store(tmp_path)
    cfg = dict(CFG, image_fingerprints=True)
    red, green, blue = (220, 30, 30), (30, 200, 40), (30, 40, 220)
    a = images.add_image(m, flat(red), CFG, origin="red.png")["id"]
    b = images.add_image(m, flat((205, 45, 40)), CFG, origin="nearly-red.png")["id"]
    c = images.add_image(m, flat(blue), CFG, origin="blue.png")["id"]
    d = images.add_image(m, two_tone(green, red), CFG, origin="green-and-red.png")["id"]
    small = images.add_image(m, flat(green, 320, 240), CFG, origin="small.png")["id"]      # too small to be cut into parts
    assert fp.status(m) == {"images": 5, "fingerprinted": 0, "waiting": 5, "models": []} and fp.similar(m, cfg, a) == []
    assert not images.get_image(m, a)["fingerprint"]
    sent = []
    report = fp.fingerprint(m, cfg, embed=lambda ps: sent.append(len(ps)) or colour_prints(ps))
    assert report["done"] == [a, b, c, d, small] and report["errors"] == [] and report["model"] == "colours-1"
    assert sent == [10, 10, 10, 10, 1]                                         # the whole image and its nine parts, in one request
    assert fp.status(m, "colours-1")["waiting"] == 0 and fp.status(m, "another-model")["waiting"] == 5 and images.get_image(m, a)["fingerprint"]
    assert fp.fingerprint(m, cfg, embed=colour_prints, model="colours-1")["done"] == []          # nothing is done twice
    alike = fp.similar(m, cfg, a)
    assert [x["id"] for x in alike][:2] == [b, d] and c not in [x["id"] for x in alike]
    assert alike[0]["alike"] > 0.98 and (alike[0]["this"], alike[0]["that"]) == ("whole", "whole")
    half = next(x for x in alike if x["id"] == d)                              # the red half of that one, not the picture as a whole
    assert half["alike"] > 0.98 and half["whole"] < 0.8 and half["that"] in ("top right", "middle right", "bottom right")
    assert "the whole of this one and the" in fp.describe_match(half) and fp.describe_match(alike[0]) == "the two pictures as a whole"
    assert [x["id"] for x in fp.similar(m, cfg, small)] == [d]                 # found through the green half
    assert len(fp.similar(m, cfg, a, minimum=-1)) == 4 and len(fp.similar(m, cfg, a, n=1)) == 1
    # a part known to show nothing worth recording is not matched on
    with m._lock:
        images._db(m).execute("UPDATE image_sections SET notable = 0 WHERE image_id = ?", (d,))
    left = next(x for x in fp.similar(m, cfg, a, minimum=-1) if x["id"] == d)
    assert left["alike"] < 0.8 and (left["this"], left["that"]) == ("whole", "whole")       # only the pictures as wholes are compared
    # set aside: not found; shown again after that: fingerprinted afresh
    images.forget_image(m, b)
    assert b not in [x["id"] for x in fp.similar(m, cfg, a, minimum=-1)] and fp.status(m)["images"] == 4
    images.restore_image(m, CFG, b)
    assert fp.similar(m, cfg, a)[0]["id"] == b
    # fingerprints from another model are made again and never compared with the old ones
    other = lambda ps: ("colours-2", colour_prints(ps)[1])
    assert fp.fingerprint(m, cfg, embed=other, model="colours-2", should_stop=lambda: len(fp.status(m, "colours-2")["models"]) > 1)["done"] == [a]
    assert fp.status(m)["models"] == ["colours-1", "colours-2"] and fp.similar(m, cfg, a) == []
    assert fp.fingerprint(m, cfg, embed=other, model="colours-2")["done"] == [b, c, d, small] and fp.similar(m, cfg, a)[0]["id"] == b
    # the helper not running: reported, nothing lost
    down = fp.fingerprint(m, dict(cfg, image_fingerprint_host="http://127.0.0.1:9"), redo=True)
    assert down["done"] == [] and "Could not reach the fingerprint server" in down["errors"][0] and fp.status(m)["waiting"] == 0


def test_the_fingerprint_helper_and_the_plugin_talk_to_each_other(tmp_path):
    """The helper's web side with a stand-in for the model, and the plugin against it: status, a pass made as part
    of describing images, and what the command line and the agent's tool say."""
    import importlib.util, threading
    from holonomic import images, fingerprints as fp
    spec = importlib.util.spec_from_file_location("fingerprint_server", Path(images.__file__).parent / "tools" / "fingerprint_server.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    server = helper.make_server(lambda ps: colour_prints(ps)[1], model="colours-1", dim=3, device="cpu", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{server.server_port}"
    cfg = dict(CFG, image_fingerprints=True, image_fingerprint_host=host + "/")
    try:
        assert fp.health(cfg) == {"model": "colours-1", "dim": 3, "device": "cpu", "faces": None}
        assert fp.make_embedder(cfg)([picture(colour=(250, 0, 0))])[0] == "colours-1"
        for bad in ({"images": []}, {"images": ["not a picture"]}, {}):
            try:
                fp._call(host + "/embed", bad, 5); assert False
            except fp.FingerprintError as exc:
                assert "answered 4" in str(exc)
        m = store(tmp_path)
        a = images.add_image(m, flat((220, 30, 30)), CFG)["id"]
        b = images.add_image(m, flat((210, 40, 35)), CFG)["id"]
        report = images.process(m, cfg, see=eyes())                            # describing images makes the fingerprints too
        assert report["fingerprints"]["done"] == [a, b] and report["fingerprints"]["model"] == "colours-1"
        assert "fingerprints" not in images.process(m, CFG, see=eyes())        # switched off: not attempted
        assert fp.similar(m, cfg, a)[0]["id"] == b
    finally:
        server.shutdown()


def test_fingerprints_from_the_command_line_and_the_tool(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib, importlib.util, threading
    from holonomic import images, fingerprints as fp
    spec = importlib.util.spec_from_file_location("fingerprint_server", Path(images.__file__).parent / "tools" / "fingerprint_server.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    server = helper.make_server(lambda ps: colour_prints(ps)[1], model="colours-1", dim=3, device="cpu", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{server.server_port}"
    p = make(tmp_path)
    p._cfg.update(CFG)
    home = tmp_path / "home"
    (home / "holonomic.json").write_text(json.dumps(dict({"embedder": "hash", "min_score": 0.3}, **CFG)))
    e = p._engine
    a = images.add_image(e, flat((220, 30, 30)), CFG, origin="red.png")["id"]
    b = images.add_image(e, flat((210, 40, 35)), CFG, origin="nearly-red.png")["id"]
    c = images.add_image(e, flat((30, 40, 220)), CFG, origin="blue.png")["id"]
    try:
        assert "fingerprints are off" in tool(p, action="images", image_id=a, similar=True)["error"]
        p._cfg.update(image_fingerprints=True, image_fingerprint_host=host)
        assert "no fingerprint yet" in tool(p, action="images", image_id=a, similar=True)["note"]
        fp.fingerprint(e, p._cfg)
        found = tool(p, action="images", image_id=a, similar=True)
        assert [i["image_id"] for i in found["images"]] == [b] and found["images"][0]["alike"] > 0.98
        assert found["images"][0]["what_is_alike"] == "the two pictures as a whole" and "nothing was compared in words" in found["note"]
        assert "Nothing she has been shown looks like it" in tool(p, action="images", image_id=c, similar=True)["note"]
        with e._lock:                                   # the command line starts with none made
            images._db(e).execute("UPDATE images SET vec = NULL, vec_model = NULL")
            images._db(e).execute("UPDATE image_sections SET vec = NULL, vec_model = NULL")
        p.shutdown()
        sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
        import holonomic.cli as cli, holonomic.embed as embed
        keep = embed.OllamaEmbedder
        embed.OllamaEmbedder = lambda *x, **k: HashEmbedder()
        parser = argparse.ArgumentParser(); cli.register_cli(parser)

        def run(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                args = parser.parse_args(list(argv)); args.func(args)
            return out.getvalue()
        try:
            out = run("images", "fingerprints")
            assert "Picture fingerprints are OFF" in out and "the helper is NOT answering" in out and "fingerprint_server.py" in out
            out = run("images", "fingerprints", "on", "--host", host + "/")
            assert "Picture fingerprints are ON" in out and "the helper is running: colours-1, 3 numbers per picture, on cpu" in out
            assert "made fingerprints for 3 image(s)" in out and json.loads((home / "holonomic.json").read_text())["image_fingerprint_host"] == host
            assert "3 of 3 image(s) have a fingerprint from colours-1" in run("images", "fingerprints") and "fingerprints: on, 3 of 3" in run("images")
            out = run("images", "similar", str(a))
            assert "1 image(s) look like it" in out and f"image #{b}  nearly-red.png" in out and "blue.png" not in out
            assert "2 image(s) compared" in run("images", "similar", str(a), "--all") and "blue.png" in run("images", "similar", str(a), "--all")
            assert "Nothing she has been shown looks like it" in run("images", "similar", str(c)) and "Usage:" in run("images", "similar")
            assert "made fingerprints for 3 image(s)" in run("images", "fingerprints", "redo")
            assert "Picture fingerprints are OFF" in run("images", "fingerprints", "off")
        finally:
            embed.OllamaEmbedder = keep
    finally:
        server.shutdown()


def test_a_named_thing_is_recognised_by_its_look(tmp_path):
    """"This is Sushi."  A later image with a part that looks like Sushi is offered to the vision model as possibly
    Sushi; if she then calls it Sushi the image is recorded as showing it.  A name is never used on the fingerprint
    alone, a recognised image is never an example, people are passed over, and the user can say it is not."""
    import importlib.util, threading
    from holonomic import images, fingerprints as fp
    spec = importlib.util.spec_from_file_location("fingerprint_server", Path(images.__file__).parent / "tools" / "fingerprint_server.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    server = helper.make_server(lambda ps: colour_prints(ps)[1], model="colours-1", dim=3, device="cpu", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cfg = dict(CFG, image_fingerprints=True, image_fingerprint_host=f"http://127.0.0.1:{server.server_port}")
    red, green, blue = (220, 30, 30), (30, 200, 40), (30, 40, 220)
    seen, agrees = [], [True]

    def see(step, system, prompt, jpeg, schema, max_tokens):
        seen.append((step, prompt))
        offered = "may show something you have been shown before: Sushi (a red cat)" in prompt
        denied = "does not show Sushi" in prompt
        said = "said: This is Sushi" in prompt or "This shows" in prompt
        if step == "names":
            who = "Kayla" if "Kayla" in prompt else "Sushi"
            return json.dumps({"names": [{"name": who, "what": "a red cat", "kind": "person" if who == "Kayla" else "animal"},
                                         {"name": "Mochi", "what": "a dog", "kind": "animal"}]})      # Mochi was never said
        if step == "image":
            named = (said or (offered and agrees[0])) and not denied
            return json.dumps({"description": ("A photo of Sushi, a red cat, beside something." if named else "A red shape beside something else."),
                               "labels": ["cat"] if named else ["shape"], "text": "", "people": False})
        place = prompt.split("the ", 1)[1].split(".", 1)[0]
        if place in ("middle left", "middle right"):
            named = (("which is true: This is Sushi" in prompt and place == "middle left") or (offered and agrees[0])) and not denied
            return json.dumps({"notable": True, "description": "Sushi's red fur up close." if named else "A flat area of colour.", "labels": []})
        return json.dumps({"notable": False, "description": "", "labels": []})
    try:
        m = store(tmp_path)
        a = images.add_image(m, two_tone(red, green), CFG, caption="This is Sushi, my cat.", origin="sushi.png")["id"]
        images.process(m, cfg, see=see)
        assert [s for s, _ in seen].count("names") == 1 and not any("may show" in p for _, p in seen)       # nothing known yet
        assert fp.names(m) == [{"name": "Sushi", "what": "a red cat", "kind": "animal", "examples": [a], "recognised": [], "not": []}]
        assert images.get_image(m, a)["named"] == [{"name": "Sushi", "said": True, "alike": None}]
        assert [i["id"] for i in images.find_by_label(m, "sushi")] == [a]
        # a new image with the same red on its other side, and nothing said about it
        seen.clear()
        b = images.add_image(m, two_tone(blue, red), CFG, origin="later.png")["id"]
        assert set(fp.recognise(m, cfg, b)["sushi"]["places"]) >= {"middle right"} and "middle left" not in fp.recognise(m, cfg, b)["sushi"]["places"]
        images.process(m, cfg, see=see)
        prompts = dict((p.split("the ", 1)[1].split(".", 1)[0] if s == "image_part" else s, p) for s, p in seen)
        assert "may show something you have been shown before: Sushi (a red cat). Look for it." in prompts["image"]
        assert "this part may show" in prompts["middle right"] and "may show" not in prompts["middle left"]       # only where it was matched
        assert "names" not in prompts                                                                          # nothing was said, nothing to learn
        got = images.get_image(m, b, sections=True)
        assert "Sushi" in got["description"] and got["named"][0]["name"] == "Sushi" and got["named"][0]["said"] is False and got["named"][0]["alike"] > 0.9
        assert sorted(i["id"] for i in images.find_by_label(m, "sushi")) == [a, b] and fp.names(m)[0]["recognised"] == [b]
        # the fingerprint says yes but she does not see it: no name is recorded
        agrees[0] = False
        c = images.add_image(m, two_tone(red, blue), CFG)["id"]
        images.process(m, cfg, see=see)
        assert images.get_image(m, c)["named"] == [] and "Sushi" not in images.get_image(m, c)["description"]
        agrees[0] = True
        # nothing alike: nothing offered.  And b, only recognised, is not an example: blue is not Sushi.
        seen.clear()
        d = images.add_image(m, flat(blue), CFG)["id"]
        images.process(m, cfg, see=see)
        assert not any("may show" in p for _, p in seen) and fp.recognise(m, cfg, d) == {}
        # the user says b is not Sushi: looked at again with that taken as true, and never offered for it again
        seen.clear()
        assert fp.not_named(m, cfg, b, "sushi", see=see) and not fp.not_named(m, cfg, b, "Nobody")
        assert "does not show Sushi" in seen[0][1] and "may show" not in seen[0][1]
        assert "Sushi" not in images.get_image(m, b)["description"] and images.get_image(m, b)["named"] == [] and fp.recognise(m, cfg, b) == {}
        assert [i["id"] for i in images.find_by_label(m, "sushi")] == [a] and fp.names(m)[0]["not"] == [b] and fp.names(m)[0]["recognised"] == []
        # ... and says c is: it becomes a second example, looked at again so that what is written says so
        done = fp.name_thing(m, cfg, c, "Sushi", see=see)
        assert done["looked_again"] and done["what"] == "a red cat" and fp.names(m)[0]["examples"] == [a, c]
        assert "Sushi" in images.get_image(m, c)["description"] and [i["id"] for i in images.find_by_label(m, "sushi")] == [c, a]
        assert fp.name_thing(m, cfg, a, "Sushi", see=see)["looked_again"] is False                # already says so
        # what the user says a thing is replaces the earlier wording; what she works out for herself later does not
        assert fp.name_thing(m, cfg, a, "Sushi", what="a tuxedo cat", see=see)["what"] == "a tuxedo cat"
        assert fp.name_thing(m, cfg, a, "Sushi", what="a red cat", keep_what=True, see=see)["what"] == "a tuxedo cat"
        assert fp.name_thing(m, cfg, a, "Sushi", what="a red cat", see=see)["what"] == "a red cat"
        # people are not named this way, by her or by the user
        e = images.add_image(m, flat(green), CFG, caption="This is Kayla.")["id"]
        images.process(m, cfg, see=see)
        assert [n["name"] for n in fp.names(m)] == ["Sushi"] and images.get_image(m, e)["named"] == []
        for bad in (dict(name="Kayla", kind="person"), dict(name=" ")):
            try:
                fp.name_thing(m, cfg, e, **bad); assert False
            except fp.NameRefused:
                pass
        # names off, or fingerprints off: nothing is offered or learned
        assert fp.recognise(m, dict(cfg, image_names=False), b) == {} and fp.recognise(m, CFG, b) == {}
        # deleting an example for good takes it out of the name; forgetting the name ends it
        images.forget_image(m, c); images.delete_image(m, c)
        assert fp.names(m)[0]["examples"] == [a]
        assert fp.forget_name(m, "SUSHI") and not fp.forget_name(m, "Sushi") and fp.names(m) == [] and images.get_image(m, a)["named"] == []
    finally:
        server.shutdown()


def test_an_example_with_a_person_in_it_is_left_out_when_there_is_one_without(tmp_path):
    """Theo lying on his owner is mostly a picture of his owner.  With a photo of Theo alone to go by, that one is
    not used, so a later photo of the same owner with another cat is not taken for Theo."""
    from holonomic import images, fingerprints as fp
    assert fp._has_person("The woman's blonde hair and glasses.") and fp._has_person("A hand resting on the cat.")
    assert not fp._has_person("A tabby cat in a harness on a rug.") and not fp._has_person("The cathedral's arch.")
    m = store(tmp_path)
    cfg = dict(CFG, image_fingerprints=True)
    yellow, purple, blue = (230, 210, 30), (150, 30, 200), (30, 40, 220)
    alone = images.add_image(m, flat(yellow), CFG, origin="theo-alone.png")["id"]
    held = images.add_image(m, flat(purple), CFG, origin="theo-on-kayla.png")["id"]
    later = images.add_image(m, flat((160, 40, 190)), CFG, origin="kayla-with-another-cat.png")["id"]
    images.set_people(m, held, True)
    fp.fingerprint(m, cfg, embed=colour_prints)
    fp.name_thing(m, cfg, held, "Theo", what="a yellow cat", redo=False)
    assert "theo" in fp.recognise(m, cfg, later)                      # the only example is the one with her in it: it has to serve
    fp.name_thing(m, cfg, alone, "Theo", redo=False)
    assert fp.recognise(m, cfg, later) == {} and fp.names(m)[0]["examples"] == [alone, held]
    # a picture that has only just arrived, not kept: checked the same way, and nothing is stored
    before = images.count_images(m)
    got = fp.recognise_picture(m, cfg, two_tone(blue, yellow), embed=colour_prints)
    assert got["theo"]["shown"] == "Theo" and "middle right" in got["theo"]["places"] and "middle left" not in got["theo"]["places"]
    assert fp.recognise_picture(m, cfg, flat(blue), embed=colour_prints) == {} and images.count_images(m) == before
    assert fp.recognise_picture(m, cfg, b"not a picture", embed=colour_prints) == {}
    assert fp.recognise_picture(m, dict(cfg, image_fingerprint_host="http://127.0.0.1:9"), flat(yellow), timeout=1) == {}       # helper not running


def test_she_is_told_what_an_arriving_image_may_show_before_she_answers(tmp_path):
    if not HAVE_HERMES: return
    import importlib.util, threading
    from holonomic import images, fingerprints as fp
    spec = importlib.util.spec_from_file_location("fingerprint_server", Path(images.__file__).parent / "tools" / "fingerprint_server.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    server = helper.make_server(lambda ps: colour_prints(ps)[1], model="colours-1", dim=3, device="cpu", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    p = make(tmp_path)
    p._cfg.update(CFG, image_fingerprints=True, image_fingerprint_host=f"http://127.0.0.1:{server.server_port}")
    e = p._engine
    try:
        sushi = images.add_image(e, flat((220, 30, 30)), CFG, origin="sushi.png")["id"]
        fp.fingerprint(e, p._cfg)
        fp.name_thing(e, p._cfg, sushi, "Sushi", what="a red cat", redo=False)
        shot, other, same = tmp_path / "new.png", tmp_path / "other.png", tmp_path / "same.png"
        shot.write_bytes(two_tone((30, 40, 220), (215, 35, 30))); other.write_bytes(flat((30, 200, 40))); same.write_bytes(flat((220, 30, 30)))
        said = p.prefetch(f"[1 image] look at this\n\n[Image attached at: {shot}]", session_id="s1")
        assert "## The attached image looks like Sushi" in said and "Sushi, a red cat (you have been shown Sushi in 1 image(s) before)." in said
        assert "Do not take a name from earlier in the conversation in its place." in said and "say the name when you answer" in said
        assert "If what you see does not fit, do not use the name." in said and "## The attached image shows" not in said
        assert (tmp_path / "home" / "holonomic" / "last_context.txt").read_text(encoding="utf-8").count("## The attached image looks like Sushi") == 1
        # recognising him brings back what she knows about him, though the words were only "look at this"
        e.remember("Sushi is Kayla's cat and sleeps on the pine-green couch.", kind="said_user", session="s0")
        said = p.prefetch(f"[1 image] Look at this!\n\n[Image attached at: {shot}]", session_id="s1")
        assert "Sushi is Kayla's cat and sleeps on the pine-green couch." in said and "## The attached image looks like Sushi" in said
        assert "Sushi is Kayla's cat" not in p.prefetch(f"[1 image] Look at this!\n\n[Image attached at: {other}]", session_id="s1")
        said = p.prefetch(f"[1 image] and this\n\n[Image attached at: {other}]", session_id="s1")                    # nothing like it
        assert "## The attached image" not in said and "## Nothing in the attached image was recognised as something you know by name" in said
        assert "You know these by name: Sushi (a red cat). None of them was recognised" in said and "Do not take an animal or thing" in said
        kept = (tmp_path / "home" / "holonomic" / "last_context.txt").read_text(encoding="utf-8")
        assert "recognised in the attached image: nothing\na statement that she recognises it was not called for" in kept
        assert "she was told that nothing in the image was recognised" in kept
        again = p.prefetch(f"[1 image] this one again\n\n[Image attached at: {same}]", session_id="s1")                # the very image she has
        assert "An image you have seen before" in again and "looks like" not in again and "Nothing in the attached image" not in again
        assert "## The attached image shows Sushi" in again and "This is Sushi: you recognise it by sight." in again and "Sushi is Kayla's cat" in again
        p._cfg["image_names"] = False
        off = p.prefetch(f"[1 image] look at this\n\n[Image attached at: {shot}]", session_id="s1")
        assert "looks like" not in off and "Nothing in the attached image" not in off                # names off: nothing is said either way
    finally:
        p.shutdown(); server.shutdown()


def test_a_photo_sent_as_a_file_is_recognised_before_she_answers(tmp_path):
    """The way a phone's photos arrive in the desktop app: as a file she is not shown.  It is looked at for her,
    the named thing in it is recognised by its look, and she is told plainly that she knows it."""
    if not HAVE_HERMES: return
    import importlib.util, threading
    from holonomic import images, fingerprints as fp
    spec = importlib.util.spec_from_file_location("fingerprint_server", Path(images.__file__).parent / "tools" / "fingerprint_server.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    calls = []
    server = helper.make_server(lambda ps: calls.append(len(ps)) or colour_prints(ps)[1], model="colours-1", dim=3, device="cpu", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    p = make(tmp_path)
    p._cfg.update(CFG, image_fingerprints=True, image_fingerprint_host=f"http://127.0.0.1:{server.server_port}")
    e = p._engine
    keep = images._seer

    def see(step, system, prompt, jpeg, schema, max_tokens):
        if step == "image":
            knows = "may show something you have been shown before: Sushi" in prompt
            return json.dumps({"description": "A close-up photo shows a red cat" + (", Sushi," if knows else "") + " sitting in a paper bag on a carpet.",
                               "labels": ["cat", "paper bag"], "text": "", "people": False})
        return json.dumps({"names": [], "notable": False, "description": "", "labels": []})
    images._seer = lambda ic, report: see
    try:
        sushi = images.add_image(e, flat((220, 30, 30)), CFG, origin="sushi.png")["id"]
        fp.fingerprint(e, p._cfg)
        fp.name_thing(e, p._cfg, sushi, "Sushi", what="a red cat", redo=False)
        e.remember("Sushi is Kayla's cat and sleeps on the pine-green couch.", kind="said_user", session="s0")
        photo = tmp_path / "IMG_7733.HEIC"
        photo.write_bytes(two_tone((120, 90, 60), (215, 35, 30)))
        calls.clear()
        block = p.prefetch(file_message(photo, "Look at this!"), session_id="s1")
        assert "## The picture attached to this message (you were not shown it directly" in block and "a red cat, Sushi, sitting in a paper bag" in block
        assert "## The attached image shows Sushi" in block and "do not ask what or whose it is" in block and "That is Sushi!" in block
        assert "Sushi, a red cat (you have been shown Sushi in 2 image(s) before)." in block and "looks like Sushi" not in block
        assert "Do not call it 'a cat' or 'this cat' or by a description" in block
        assert "Sushi is Kayla's cat and sleeps on the pine-green couch." in block             # what she knows about him came back
        assert calls == [10]                                    # looked at once, not twice
        new = images.list_images(e)[0]
        assert new["named"] == [{"name": "Sushi", "said": False, "alike": new["named"][0]["alike"]}] and new["named"][0]["alike"] > 0.9
        kept = (tmp_path / "home" / "holonomic" / "last_context.txt").read_text(encoding="utf-8")
        assert ("--- not given to her; for checking ---\nrecognised in the attached image: Sushi (seen)\n"
                "a statement that she recognises it was given\nages of what was recalled: ") in kept and "\nplugin version " in kept
        assert "--- not given to her" not in block
        # another request for memory arriving while this image is still being looked at must not lose what was found in it
        second = tmp_path / "IMG_7734.HEIC"
        second.write_bytes(two_tone((60, 90, 120), (212, 38, 30)))
        describe = images.describe

        def slow(engine, cfg, image_id, **kw):
            done = describe(engine, cfg, image_id, **kw)
            p.prefetch("What do the tomato plants need this week?", session_id="s2")
            return done
        images.describe = slow
        try:
            block = p.prefetch(file_message(second, "Look at this!"), session_id="s1")
        finally:
            images.describe = describe
        assert "## The attached image shows Sushi" in block and "Sushi is Kayla's cat" in block
        # recording the name with the image fails: she is still told, from the match and from what she wrote
        third = tmp_path / "IMG_7735.HEIC"
        third.write_bytes(two_tone((90, 60, 120), (214, 36, 32)))
        keep_note = fp.note_recognised
        fp.note_recognised = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("could not write"))
        try:
            block = p.prefetch(file_message(third, "Look at this!"), session_id="s1")
        finally:
            fp.note_recognised = keep_note
        assert "## The attached image shows Sushi" in block
        assert "matched by look but not recorded with the image" in (tmp_path / "home" / "holonomic" / "last_context.txt").read_text(encoding="utf-8")
    finally:
        images._seer = keep
        p.shutdown(); server.shutdown()


# ---------------------------------------------------------------------------------------------------- faces

GREY = (128, 128, 128)


def face_finder(calls=None, not_faces=()):
    """A stand-in for the face model.  A picture's left and right halves each hold one 'face' unless that half is
    grey; the face's fingerprint is the half's colour.  One colour all over is a single face in the middle."""
    from PIL import Image

    def find(pictures):
        out = []
        for data in pictures:
            if calls is not None:
                calls.append(1)
            img = Image.open(io.BytesIO(data)).convert("RGB").resize((8, 8))
            side = lambda xs: tuple(sum(img.getpixel((x, y))[i] for x in xs for y in range(8)) // (len(xs) * 8) for i in range(3))
            left, right = side(range(0, 3)), side(range(5, 8))
            near = lambda a, b: all(abs(a[i] - b[i]) < 25 for i in range(3))
            vec = lambda c: [c[i] - 110.0 for i in range(3)]
            faces = []
            face = lambda c: not near(c, GREY) and not any(near(c, n) for n in not_faces)
            if near(left, right):
                if face(left):
                    faces.append({"box": [0.4, 0.3, 0.2, 0.3], "score": 0.95, "vector": vec(left)})
            else:
                if face(left):
                    faces.append({"box": [0.1, 0.3, 0.2, 0.3], "score": 0.95, "vector": vec(left)})
                if face(right):
                    faces.append({"box": [0.7, 0.3, 0.2, 0.3], "score": 0.95, "vector": vec(right)})
            out.append(faces)
        return "faces-1", out
    return find


def test_faces_are_learned_only_as_far_as_the_user_allows(tmp_path):
    """none: nothing is looked for.  me: only the user's face is kept.  named: only people the user named.
    often: every face, so that someone who keeps appearing is noticed.  Examples are only faces the user spoke for."""
    from holonomic import images, faces as fc
    kayla, emma, sam = (230, 60, 40), (40, 200, 60), (40, 60, 230)
    find = face_finder()
    assert fc.face_config({})["face_learn"] == "none" and fc.face_config({"face_learn": "everyone"})["face_learn"] == "none"
    assert not fc.faces_on(CFG) and fc.dream_policy({}, False) == "none" and fc.dream_policy({}, True) == "anyone"

    def setup(folder):
        m = store(tmp_path / folder)
        ids = {"me": images.add_image(m, flat(kayla), CFG, origin="me.png")["id"],
               "both": images.add_image(m, two_tone((225, 65, 45), emma), CFG, origin="me-and-emma.png")["id"],
               "emma": images.add_image(m, flat((45, 195, 65)), CFG, origin="emma.png")["id"],
               "sam": images.add_image(m, flat(sam), CFG, origin="sam.png")["id"],
               "sam2": images.add_image(m, two_tone(GREY, (45, 65, 225)), CFG, origin="sam-again.png")["id"],
               "none": images.add_image(m, flat(GREY), CFG, origin="landscape.png")["id"]}
        return m, ids
    stored = lambda m: images._db(m).execute("SELECT COUNT(*) FROM image_faces").fetchone()[0]
    # none: nothing is looked for or kept
    m, ids = setup("none")
    assert fc.scan(m, CFG, find=find) == {"done": [], "errors": []} and fc.detect(m, CFG, ids["me"], find=find) == []
    try:
        fc.name_person(m, CFG, ids["me"], "Kayla", me=True, find=find); assert False
    except fc.FaceError as exc:
        assert "off" in str(exc)
    # me: only my face
    cfg = dict(CFG, face_learn="me")
    m, ids = setup("me")
    assert len(fc.scan(m, cfg, find=find)["done"]) == 6 and stored(m) == 0 and fc.waiting(m) == 0      # nobody is known: nothing is kept
    assert fc.face_count(m, ids["both"]) == 2 and fc.face_count(m, ids["none"]) == 0
    try:
        fc.name_person(m, cfg, ids["emma"], "Emma", find=find); assert False
    except fc.FaceError as exc:
        assert "Only your own face" in str(exc)
    done = fc.name_person(m, cfg, ids["me"], "Kayla", me=True, find=find)
    assert done["is_user"] and done["new"] and done["where"] == "the only face" and m.kv_get("faces:look_again") == "1"
    fc.scan(m, cfg, find=find)                                             # everything is gone through again
    assert m.kv_get("faces:look_again") == "0" and stored(m) == 2          # mine in two images; Emma's and Sam's are not kept
    assert [(p["name"], p["is_user"], p["where"], p["said"]) for p in fc.people_in(m, ids["both"])] == [("Kayla", True, "the only face", False)]
    assert fc.people(m) == [{"name": "Kayla", "is_user": True, "dream": None, "said": [ids["me"]], "seen": [ids["both"]]}]
    # named: people I name
    cfg = dict(CFG, face_learn="named")
    m, ids = setup("named")
    fc.scan(m, cfg, find=find)
    try:
        fc.name_person(m, cfg, ids["both"], "Emma", find=find); assert False
    except fc.NeedsChoice as exc:
        assert "has 2 faces: face 1 on the left; face 2 on the right. Say which one is Emma." in str(exc)
    assert fc.name_person(m, cfg, ids["both"], "Emma", where="right", find=find)["face"] == 2
    assert fc.name_person(m, cfg, ids["both"], "Kayla", face=1, me=True, find=find)["where"] == "on the left"
    fc.scan(m, cfg, find=find)
    assert [p["name"] for p in fc.people_in(m, ids["emma"])] == ["Emma"] and [p["name"] for p in fc.people_in(m, ids["me"])] == ["Kayla"]
    assert fc.people_in(m, ids["sam"]) == [] and stored(m) == 4            # Sam is nobody I named: his face is not kept
    shown = fc.look(m, cfg, ids["both"], find=find)
    assert [(f["n"], f["where"], f["name"], f["said"]) for f in shown] == [(1, "on the left", "Kayla", True), (2, "on the right", "Emma", True)]
    # a recognised face is never an example: only the two I spoke for
    assert {k: len(v) for k, v in fc._examples(m, "faces-1").items()} == {"kayla": 1, "emma": 1}
    # that is not Emma: never taken for her again, and the fingerprint is dropped
    assert fc.not_person(m, cfg, ids["emma"], "Emma") == 1 and fc.not_person(m, cfg, ids["emma"], "Emma") == 0
    fc.scan(m, cfg, find=find, again=True)
    assert fc.people_in(m, ids["emma"]) == [] and images._db(m).execute("SELECT vec FROM image_faces WHERE state = 'not'").fetchone()["vec"] is None
    assert fc.forget_person(m, "emma") and not fc.forget_person(m, "Emma") and [p["name"] for p in fc.people(m)] == ["Kayla"]
    # often: every face, and someone who keeps appearing
    cfg = dict(CFG, face_learn="often", face_often_images=2, face_ask_names=True)
    m, ids = setup("often")
    fc.scan(m, cfg, find=find)
    assert stored(m) == 6
    groups = fc.strangers(m, cfg)
    assert sorted(tuple(g["images"]) for g in groups) == sorted([(ids["me"], ids["both"]), (ids["both"], ids["emma"]), (ids["sam"], ids["sam2"])])
    assert fc.keeps_appearing(m, cfg, image_id=ids["sam"])["images"] == 2 and fc.keeps_appearing(m, cfg, image_id=ids["none"]) is None
    assert fc.keeps_appearing(m, dict(cfg, face_ask_names=False), image_id=ids["sam"]) is None       # she may not ask
    fc.asked(m, fc.keeps_appearing(m, cfg, image_id=ids["sam"])["faces"])
    assert fc.keeps_appearing(m, cfg, image_id=ids["sam2"]) is None                                 # asked once, not again
    unknown = []
    assert fc.recognise_picture(m, cfg, flat((50, 190, 70)), find=find, unknown=unknown) == [] and len(unknown) == 1
    assert fc.keeps_appearing(m, cfg, vectors=unknown)["images"] == 3                                # Emma, in a picture just arriving
    fc.name_person(m, cfg, ids["sam"], "Sam", find=find)
    fc.scan(m, cfg, find=find)
    assert [p["name"] for p in fc.people_in(m, ids["sam2"])] == ["Sam"] and len(fc.strangers(m, cfg)) == 2
    got = fc.recognise_picture(m, cfg, two_tone((42, 62, 228), GREY), find=find)
    assert [(g["name"], g["is_user"], g["where"]) for g in got] == [("Sam", False, "the only face")] and got[0]["alike"] > 0.9
    assert fc.recognise_picture(m, cfg, b"not a picture", find=find) == []
    # one image set apart: no face is looked for in it, what was kept from it is dropped, and it stays that way
    before = stored(m)
    assert fc.set_image(m, ids["both"], False) and fc.image_off(m, ids["both"]) and not fc.set_image(m, 999, False)
    assert stored(m) == before - 2 and fc.faces_in(m, ids["both"]) == [] and fc.face_count(m, ids["both"]) == 0
    assert images.get_image(m, ids["both"])["faces_off"] and not images.get_image(m, ids["sam"])["faces_off"]
    fc.scan(m, cfg, find=find, again=True)
    assert stored(m) == before - 2 and fc.detect(m, cfg, ids["both"], find=find) == [] and fc.waiting(m) == 0
    assert fc.recognise_picture(m, cfg, two_tone((225, 65, 45), emma), find=find) == []           # the same picture, shown again
    for refused in (lambda: fc.look(m, cfg, ids["both"], find=find), lambda: fc.name_person(m, cfg, ids["both"], "Emma", face=2, find=find)):
        try:
            refused(); assert False
        except fc.FaceError as exc:
            assert "turned off for image" in str(exc)
    assert fc.set_image(m, ids["both"], True) and fc.waiting(m) == 1
    fc.scan(m, cfg, find=find)
    assert stored(m) == before and not fc.image_off(m, ids["both"])
    # the rule is tightened: what it no longer allows is dropped the next time images are gone through
    fc.scan(m, dict(cfg, face_learn="named"), find=find, again=True)
    assert stored(m) == 2 and fc.strangers(m, cfg) == []
    # an image deleted for good takes its faces; everything can be dropped at once
    images.forget_image(m, ids["sam2"]); images.delete_image(m, ids["sam2"])
    assert stored(m) == 1 and fc.forget_all(m) == 1 and fc.people(m) == [] and fc.waiting(m) == 5
    # asking who
    assert fc.asks_who("Who is this?") and fc.asks_who("do you recognise her") and fc.asks_who("Do you know who that is")
    assert not fc.asks_who("Look at this!") and not fc.asks_who("The whole garden is in bloom.")
    assert fc.may_tell(dict(cfg, face_name_unasked=True), "Look at this!") and not fc.may_tell(cfg, "Look at this!") and fc.may_tell(cfg, "who's that?")
    assert not fc.may_tell(CFG, "who's that?")                                                       # off is off


def test_who_may_be_in_an_image_a_dream_is_drawn_from(tmp_path):
    from holonomic import images, faces as fc
    kayla, emma, sam = (230, 60, 40), (40, 200, 60), (40, 60, 230)
    find = face_finder()
    cfg = dict(CFG, face_learn="named")
    m = store(tmp_path)
    me = images.add_image(m, flat(kayla), CFG)["id"]
    both = images.add_image(m, two_tone((225, 65, 45), emma), CFG)["id"]
    stranger = images.add_image(m, flat(sam), CFG)["id"]
    back = images.add_image(m, flat(GREY), CFG)["id"]              # someone seen from behind: a person, but no face
    lake = images.add_image(m, flat((120, 124, 131)), CFG, origin="lake.png")["id"]
    for i in (me, both, stranger, back):
        images.set_people(m, i, True)
    images.set_people(m, lake, False)
    fc.name_person(m, cfg, me, "Kayla", me=True, find=find)
    fc.name_person(m, cfg, both, "Emma", face=2, find=find)
    fc.scan(m, cfg, find=find, again=True)
    ok = lambda policy, i, use=False: fc.people_allowed_in_dream(m, dict(cfg, dream_image_people=policy), i, use)
    every = (me, both, stranger, back, lake)
    assert [ok("none", i) for i in every] == [False, False, False, False, True]
    assert [ok("me", i) for i in every] == [True, False, False, False, True]
    assert [ok("named", i) for i in every] == [True, True, False, False, True]
    assert [ok("anyone", i) for i in every] == [True, True, True, True, True]
    assert [ok("", i, use=False) for i in every] == [False, False, False, False, True]       # unset: as the older yes/no setting says
    assert [ok("", i, use=True) for i in every] == [True, True, True, True, True]
    # one person kept out of dreams keeps out any image they are in, whatever the rule
    assert fc.set_dream(m, "Emma", False) and not fc.set_dream(m, "Nobody", False)
    assert [ok("anyone", i) for i in every] == [True, False, True, True, True] and fc.people(m)[1]["dream"] is False
    assert fc.set_dream(m, "emma", None) and ok("named", both)
    # whose names a dream is given for the people in an image it draws on: nobody unless the user says so
    assert fc.face_config({})["dream_name_people"] == "none" and fc.dream_names(m, cfg, both) == "" and fc.dream_names(m, cfg, None) == ""
    assert fc.dream_names(m, dict(cfg, dream_name_people="me"), both) == " In it, known to you by face: Kayla (the person you talk with), on the left."
    assert fc.dream_names(m, dict(cfg, dream_name_people="named"), both) == \
        " In it, known to you by face: Kayla (the person you talk with), on the left; Emma, on the right."
    assert fc.dream_names(m, dict(cfg, dream_name_people="named"), stranger) == "" and fc.dream_names(m, dict(cfg, dream_name_people="me"), lake) == ""
    fc.set_dream(m, "Emma", False)
    assert "Emma" not in fc.dream_names(m, dict(cfg, dream_name_people="named"), both)             # kept out of dreams: not named in them either
    fc.set_dream(m, "Emma", None)
    assert fc.dream_names(m, dict(CFG, dream_name_people="named"), both) == ""                    # faces off: nothing
    # what was said about one image still decides for that image
    images.set_dream_use(m, stranger, True); images.set_dream_use(m, me, False)
    assert images.may_dream_from(m, stranger, ok("none", stranger)) and not images.may_dream_from(m, me, ok("anyone", me))


def helper_with_faces(calls=None):
    """The helper server with stand-ins for both models."""
    import importlib.util, threading
    from holonomic import images
    spec = importlib.util.spec_from_file_location("fingerprint_server", Path(images.__file__).parent / "tools" / "fingerprint_server.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    find = face_finder(calls)
    server = helper.make_server(lambda ps: colour_prints(ps)[1], model="colours-1", dim=3, device="cpu", port=0,
                                faces=lambda ps: find(ps)[1], face_model="faces-1")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_the_helper_finds_faces_only_if_it_can(tmp_path):
    import importlib.util, threading
    from holonomic import images, faces as fc, fingerprints as fp
    server, host = helper_with_faces()
    cfg = dict(CFG, face_learn="named", image_fingerprint_host=host)
    try:
        assert fp.health(cfg)["faces"] == "faces-1" and fc.available(cfg) == "faces-1"
        model, found = fc.make_finder(cfg)([two_tone((230, 60, 40), (40, 200, 60)), flat(GREY)])
        assert model == "faces-1" and [len(f) for f in found] == [2, 0] and found[0][0]["box"][0] < found[0][1]["box"][0]
    finally:
        server.shutdown()
    spec = importlib.util.spec_from_file_location("fingerprint_server", Path(images.__file__).parent / "tools" / "fingerprint_server.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    plain = helper.make_server(lambda ps: colour_prints(ps)[1], model="colours-1", dim=3, device="cpu", port=0)     # no OpenCV there
    threading.Thread(target=plain.serve_forever, daemon=True).start()
    cfg = dict(cfg, image_fingerprint_host=f"http://127.0.0.1:{plain.server_port}")
    try:
        assert fc.available(cfg) is None
        try:
            fc.make_finder(cfg)([flat(GREY)]); assert False
        except fc.FaceError as exc:
            assert "faces are not available" in str(exc)
        m = store(tmp_path)
        images.add_image(m, flat((230, 60, 40)), CFG)
        assert "faces are not available" in fc.scan(m, cfg)["errors"][0] and fc.waiting(m) == 1       # tried again later
    finally:
        plain.shutdown()


def test_she_is_told_who_is_in_a_picture_only_as_the_user_allows(tmp_path):
    if not HAVE_HERMES: return
    from holonomic import images, faces as fc
    kayla, emma = (230, 60, 40), (40, 200, 60)
    server, host = helper_with_faces()
    p = make(tmp_path)
    p._cfg.update(CFG, face_learn="named", image_fingerprint_host=host, face_min=0.9)     # the stand-in's colours are not far apart
    e = p._engine
    keep = images._seer
    seen = []

    def see(step, system, prompt, jpeg, schema, max_tokens):
        seen.append((step, prompt))
        if step == "image":
            return json.dumps({"description": "A photo of a woman with blonde hair and glasses standing in a kitchen by a window.",
                               "labels": ["woman", "kitchen"], "text": "", "people": True})
        if step == "people":
            return json.dumps({"people": [{"name": "", "me": True, "where": "left"}, {"name": "Emma", "me": False, "where": "right"},
                                          {"name": "Zed", "me": False, "where": ""}]})          # Zed was never said
        return json.dumps({"notable": False, "description": "", "labels": [], "names": []})
    images._seer = lambda ic, report: see
    context = lambda: (tmp_path / "home" / "holonomic" / "last_context.txt").read_text(encoding="utf-8")

    def show(name, data, words="Look at this!"):
        f = tmp_path / name
        f.write_bytes(data)
        return p.prefetch(file_message(f, words), session_id="s1")
    try:
        # what she says when showing a picture teaches her who is in it
        first = show("IMG_1.HEIC", two_tone(kayla, emma), "This is me with my daughter Emma.")
        assert [(q["name"], q["is_user"]) for q in fc.people(e)] == [("you", True), ("Emma", False)]
        assert "## People in the attached image" not in first and "she was NOT told" in context()
        # a later picture of her alone: recognised, but she is not told, because nobody asked
        block = show("IMG_2.HEIC", flat((225, 65, 45)))
        new = images.list_images(e)[0]
        assert [q["name"] for q in fc.people_in(e, new["id"])] == ["you"] and "## People" not in block
        assert "people recognised by face: you; she was NOT told" in context()
        assert not any("By their face" in pr for st, pr in seen if st == "image")                # nor is the describer
        assert "people" not in tool(p, action="images", image_id=new["id"])
        # asked who: told
        block = show("IMG_3.HEIC", two_tone((228, 62, 42), (42, 198, 62)), "Who is in this one?")
        assert "## People in the attached image" in block and "do not ask who they are" in block
        assert "- you, on the left: the person you are talking with. This is a picture of them." in block and "- Emma, on the right" in block
        assert "people recognised by face: you, Emma; she was told" in context()
        # the user lets her be told unasked: she is, the describer is, and the tool says who
        p._cfg["face_name_unasked"] = True
        seen.clear()
        block = show("IMG_4.HEIC", flat((42, 202, 58)))
        assert "## People in the attached image" in block and "- Emma" in block
        assert any("By their face, the people in this image include: Emma." in pr for st, pr in seen if st == "image")
        assert tool(p, action="images", image_id=images.list_images(e)[0]["id"])["people"] == ["Emma"]
        # the tool: who is in it, naming, and taking a name back
        who = tool(p, action="images", image_id=new["id"], who=True)
        assert who["faces"] == [{"face": 1, "where": "the only face", "who": "you"}]
        stranger = images.add_image(e, flat((40, 60, 230)), CFG)["id"]
        assert tool(p, action="images", image_id=stranger, who=True)["faces"][0]["who"] == "someone you do not know"
        done = tool(p, action="images", image_id=stranger, person="Sam")
        assert done["name"] == "Sam" and done["new"] and "by their face" in done["note"]
        assert "has 2 faces" in tool(p, action="images", image_id=1, person="Ann")["error"]
        assert "That face will not be taken for them again" in tool(p, action="images", image_id=stranger, person="Sam", wrong=True)["note"]
        assert "Nothing in that image was taken for Sam" in tool(p, action="images", image_id=stranger, person="Sam", wrong=True)["note"]
        # only me: nobody else can be named, by her or from what is said
        p._cfg["face_learn"] = "me"
        assert "Only your own face" in tool(p, action="images", image_id=stranger, person="Sam")["error"]
        # off: nothing at all
        p._cfg["face_learn"] = "none"
        assert "is off" in tool(p, action="images", image_id=new["id"], who=True)["error"]
        assert "## People" not in show("IMG_5.HEIC", flat((226, 64, 44)), "Who is this?")
        # every face kept, and she may ask: leave to ask about someone who keeps appearing, given once
        p._cfg.update(face_learn="often", face_ask_names=True, face_often_images=2, face_name_unasked=False)
        teal = (30, 180, 190)
        assert "Someone you keep seeing" not in show("IMG_6.HEIC", flat(teal))
        block = show("IMG_7.HEIC", flat((34, 176, 186)))
        assert "## Someone you keep seeing" in block and "has been in 2 of the images" in block and "Ask once" in block
        assert "she was given leave to ask who it is" in context()
        assert "Someone you keep seeing" not in show("IMG_8.HEIC", flat((28, 184, 194)))
    finally:
        images._seer = keep
        p.shutdown(); server.shutdown()


def test_faces_from_the_command_line(tmp_path):
    if not HAVE_HERMES: return
    import argparse, contextlib
    from holonomic import images, faces as fc
    server, host = helper_with_faces()
    p = make(tmp_path)
    home = tmp_path / "home"
    (home / "holonomic.json").write_text(json.dumps(dict({"embedder": "hash", "min_score": 0.3, "image_fingerprint_host": host}, **CFG)))
    e = p._engine
    one = images.add_image(e, flat((230, 60, 40)), CFG, origin="me.png")["id"]
    two = images.add_image(e, two_tone((226, 64, 44), (40, 200, 60)), CFG, origin="us.png")["id"]
    p.shutdown()
    sys.modules["hermes_constants"] = types.SimpleNamespace(get_hermes_home=lambda: home)
    import holonomic.cli as cli, holonomic.embed as embed
    keep = embed.OllamaEmbedder
    embed.OllamaEmbedder = lambda *x, **k: HashEmbedder()
    parser = argparse.ArgumentParser(); cli.register_cli(parser)

    def run(*argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            args = parser.parse_args(list(argv)); args.func(args)
        return out.getvalue()
    try:
        out = run("faces")
        assert "Whose faces she may learn: none  (no face is looked for at all)" in out and "faces ready (faces-1)" in out
        assert "she is told who is in an image without being asked: no, only when you ask" in out and "people she knows: nobody" in out
        assert "in images a dream picture is drawn from: none" in out and "named in dreams, going by their faces: none" in out
        assert "Learning faces is off" in run("faces", "scan") and "is off" in run("faces", "name", str(one), "Kayla", "--me")
        assert "Usage:" in run("faces", "learn", "everyone") and "Usage:" in run("faces", "ask", "maybe")
        assert "only your own face is learned" in run("faces", "learn", "me")
        assert "Only your own face may be learned" in run("faces", "name", str(two), "Emma", "--face", "2")
        assert f"image #{one}: face 1 (the only face) is Kayla, which is you." in run("faces", "name", str(one), "Kayla", "--me")
        assert "people you have named are learned" in run("faces", "learn", "named")
        out = run("faces", "name", str(two), "Emma")
        assert "has 2 faces" in out and f"faces name {two} Emma --face 1" in out
        assert "face 2 (on the right) is Emma." in run("faces", "name", str(two), "Emma", "--face", "2")
        assert "Looked for faces in 2 image(s)" in run("faces", "scan")
        out = run("faces", "show", str(two))
        assert "face 1  on the left:  Kayla (recognised)" in out and "face 2  on the right:  Emma (you said so)" in out
        out = run("faces", "people")
        assert "2 person(s) she knows by face." in out and "Kayla  (you)" in out and f"she recognised them in image #{two}" in out
        assert "is never drawn from in a dream" in run("faces", "dream", "Emma", "no") and "in dreams: never" in run("faces", "people")
        assert "noted, that is not Kayla" in run("faces", "not", str(two), "Kayla") and "Nothing in image" in run("faces", "not", str(two), "Kayla")
        assert "every face kept from it has been dropped" in run("faces", "image", str(two), "off") and "Usage:" in run("faces", "image", str(two))
        assert "turned off for image" in run("faces", "show", str(two)) and "faces: not looked for in this image" in run("images", "show", str(two))
        assert "faces are looked for in it again" in run("faces", "image", str(two), "on") and "no such image" in run("faces", "image", "99", "off")
        assert "only noticed when every face is kept" in run("faces", "often")
        assert "every face is kept" in run("faces", "learn", "often") and "She may ask, once" in run("faces", "ask", "on")
        assert "She is told who is in an image whenever" in run("faces", "unasked", "on")
        run("faces", "scan")
        assert "Nobody she does not know is in 3 or more images." in run("faces", "often")
        out = run("dreams", "images", "from_images", "--who", "named", "--name-people", "me")
        assert "are drawn from only if every face in them is someone you named" in out and "people in a dream by name, going by their faces: only you" in out
        assert "Forgot who Emma is" in run("faces", "forget", "Emma") and "nobody called" in run("faces", "forget", "Emma")
        assert "To do it: hermes holonomic faces forget --all --yes" in run("faces", "forget", "--all")
        assert "and everyone she knew by face" in run("faces", "forget", "--all", "--yes") and "people she knows: nobody" in run("faces")
        assert "Faces already kept stay where they are" in run("faces", "learn", "none")
    finally:
        embed.OllamaEmbedder = keep
        server.shutdown()


def test_a_person_in_the_picture_does_not_decide_which_cat_it_is(tmp_path):
    """Kayla holding Theo, Kayla holding Sushi.  The parts of an image with her face in them are left out on both
    sides of the comparison, so the cat is matched by the cat.  And a part of an example that speaks of 'a cat'
    counts as showing the named cat, though the name was not written down for that part."""
    from holonomic import images, faces as fc, fingerprints as fp
    kayla, theo, sushi = (230, 60, 40), (230, 220, 30), (30, 180, 190)
    find = face_finder(not_faces=[theo, sushi])
    cfg = dict(CFG, image_fingerprints=True, face_learn="named", face_min=0.9, image_name_min=0.9)    # the stand-in's colours blend
    assert fp._kind_word("a long-haired tuxedo cat, black with a white chest and paws") == "cat" and fp._kind_word("") == ""
    assert fp._face_in((0.0, 0.25, 0.5, 0.5), [(0.1, 0.3, 0.2, 0.3)]) and not fp._face_in((0.5, 0.25, 0.5, 0.5), [(0.1, 0.3, 0.2, 0.3)])
    m = store(tmp_path)

    def see(step, system, prompt, jpeg, schema, max_tokens):
        if step == "image":
            return json.dumps({"description": "A woman holding an animal against her shoulder.", "labels": ["woman"], "text": "", "people": True})
        place = prompt.split("the ", 1)[1].split(".", 1)[0] if step == "image_part" else ""
        if place.endswith("right"):
            return json.dumps({"notable": True, "description": "A cat's striped fur, seen from behind.", "labels": ["cat"]})
        if place.endswith("left"):
            return json.dumps({"notable": True, "description": "A woman's face, with glasses.", "labels": ["woman"]})
        return json.dumps({"notable": False, "description": "", "labels": [], "names": [], "people": []})

    def add(left, right, origin):
        i = images.add_image(m, two_tone(left, right), CFG, origin=origin)["id"]
        fc.detect(m, cfg, i, find=find)
        fp.fingerprint(m, cfg, embed=colour_prints, only=i)
        return i
    # the only picture of Theo has her in it, and so does the only picture of Sushi
    with_theo = add(kayla, theo, "kayla-and-theo.png")
    with_sushi = add((226, 64, 44), sushi, "kayla-and-sushi.png")
    images.process(m, CFG, see=see)
    assert fc.face_boxes(m, with_theo) == [(0.1, 0.3, 0.2, 0.3)] and fc.face_count(m, with_theo) == 1
    fp.name_thing(m, cfg, with_theo, "Theo", what="a brown tabby cat", redo=False)
    fp.name_thing(m, cfg, with_sushi, "Sushi", what="a tuxedo cat", redo=False)
    # what Theo looks like: the parts that speak of a cat and have no face in them, not the ones with her in them
    clear, peopled = fp._without_people(m, with_theo, fp._vectors(m, with_theo, "colours-1"))
    assert {p for p, _ in peopled} == {"whole", "top left", "middle left"} and "middle right" in {p for p, _ in clear}       # most of her face
    assert len(fp._examples(m, "theo", "Theo", "colours-1", -1, "a brown tabby cat")) == 3              # the three parts on the right
    # a new picture of her with each cat: the cat decides, although her face is alike in all of them
    again_theo = add((228, 62, 42), (226, 224, 34), "kayla-and-theo-again.png")
    again_sushi = add((224, 66, 46), (34, 176, 186), "kayla-and-sushi-again.png")
    assert set(fp.recognise(m, cfg, again_theo)) == {"theo"} and set(fp.recognise(m, cfg, again_sushi)) == {"sushi"}
    assert not any(place.endswith("left") or place == "whole" for place in fp.recognise(m, cfg, again_theo)["theo"]["places"])
    # her alone: no cat
    alone = add((229, 61, 41), GREY, "kayla.png")
    assert fp.recognise(m, cfg, alone) == {}
    # a picture that has only just arrived, with where its faces are handed over
    boxes = []
    fc.recognise_picture(m, cfg, two_tone((227, 63, 43), (228, 222, 32)), find=find, boxes=boxes)
    assert boxes == [(0.1, 0.3, 0.2, 0.3)]
    got = fp.recognise_picture(m, cfg, two_tone((227, 63, 43), (228, 222, 32)), embed=colour_prints, faces=boxes)
    assert set(got) == {"theo"} and not any(place.endswith("left") or place == "whole" for place in got["theo"]["places"])
    # without the faces being known, what is written about a part is gone by instead: the same answer here
    assert set(fp.recognise(m, dict(cfg, face_learn="none"), again_theo)) == {"theo"}


def test_dream_pictures_have_folders_of_their_own_by_sleep(tmp_path):
    """Everything was in one folder: photographs she had been shown and pictures she had dreamt alike."""
    from holonomic import images as im
    m = store(tmp_path)
    night1, night2 = time.time() - 30 * 3600, time.time() - 2 * 3600
    d1 = m.remember("A dream of a lake at dusk with a cat on the shore.", kind="dream", realm="dream", session="dream", whole=True, created_at=night1)[0]
    d2 = m.remember("A dream of a mountain under fireworks.", kind="dream", realm="dream", session="dream", whole=True, created_at=night1 + 1800)[0]
    d3 = m.remember("A dream of a city seen from above.", kind="dream", realm="dream", session="dream", whole=True, created_at=night2)[0]
    stamp = lambda t: time.strftime("%Y-%m-%d_%H%M", time.localtime(t))
    # two dreams of one sleep share its folder; a later sleep has another
    assert im.dream_folder(m, d1) == im.dream_folder(m, d2) == f"dream-images/{stamp(night1)}" and im.dream_folder(m, d3) == f"dream-images/{stamp(night2)}"
    shown = im.add_image(m, picture(colour=(10, 120, 200)), {}, origin="lake.png")
    a = im.add_dream_image(m, picture(colour=(200, 30, 30)), {}, dream_id=d1, scene="a lake at dusk", sources=[])
    b = im.add_dream_image(m, picture(colour=(30, 200, 30)), {}, dream_id=d2, scene="fireworks", sources=[])
    c = im.add_dream_image(m, picture(colour=(30, 30, 200)), {}, dream_id=d3, scene="a city", sources=[])
    rel = lambda img: Path(img["original"]).relative_to(m.path).as_posix()
    assert rel(shown).startswith("images/") and rel(a).startswith(f"dream-images/{stamp(night1)}/dream{d1}_")
    assert rel(b).startswith(f"dream-images/{stamp(night1)}/dream{d2}_") and rel(c).startswith(f"dream-images/{stamp(night2)}/dream{d3}_")
    assert Path(a["file"]).exists() and Path(a["original"]).exists() and "dream-images" in a["file"]
    assert [x.name for x in (m.path / "images").iterdir() if "dream" in x.name] == []
    assert im.dream_pictures(m, d1)[0]["file"] == a["file"] and im.sort_dream_files(m) == {"moved": 0, "folders": [], "missing": 0}
    # pictures from before there were folders are moved, and everything still finds them
    db = im._db(m)
    old = {}
    for img in (a, c):
        row = db.execute("SELECT file, view FROM images WHERE id = ?", (img["id"],)).fetchone()
        back = {col: "images/" + Path(row[col]).name.split("_", 1)[1] for col in ("file", "view")}
        for col in ("file", "view"):
            if (m.path / row[col]).exists():
                (m.path / row[col]).rename(m.path / back[col])
        db.execute("UPDATE images SET file = ?, view = ? WHERE id = ?", (back["file"], back["view"], img["id"]))
        old[img["id"]] = back
    assert Path(im.get_image(m, a["id"])["original"]).parent.name == "images"
    done = im.sort_dream_files(m)
    assert done["moved"] == 2 and sorted(done["folders"]) == sorted({f"dream-images/{stamp(night1)}", f"dream-images/{stamp(night2)}"}) and done["missing"] == 0
    again = im.get_image(m, a["id"])
    assert rel(again).startswith(f"dream-images/{stamp(night1)}/dream{d1}_") and Path(again["original"]).exists() and Path(again["file"]).exists()
    assert not (m.path / old[a["id"]]["file"]).exists() and im.dream_pictures(m, d3)[0]["file"].replace("\\", "/").find(f"dream-images/{stamp(night2)}/") > 0
    assert im.sort_dream_files(m)["moved"] == 0 and Path(im.get_image(m, shown["id"])["original"]).parent.name == "images"
    # a picture whose dream has since been forgotten still gets a folder, by its own date
    m.forget(d3)
    assert im.dream_folder(m, d3, fallback=night2) == f"dream-images/{stamp(night2)}"


def test_a_picture_of_the_whole_dream(tmp_path):
    """Besides its moments a dream gets one picture of all of it: drawn from the dream as she dreamt it, or from a
    single scene she composes of it.  From words alone; a whole dream draws on several images, not one."""
    import random
    from holonomic import images
    from holonomic.sleep import WHOLE_CAPTION, dreams, sleep_config, sleep_once
    assert sleep_config({})["dream_image_whole"] == "text"
    m, ids, now = dream_store(tmp_path)
    cfg = dict(CFG, dream_images="from_images", dream_image_count=1, dream_image_style="soft light", dream_image_whole="text",
               dream_image_candidates=2)
    painted, judged = [], []

    def paint(prompt, start, size=None):
        painted.append((prompt, start))
        return picture(768, 512, colour=(len(painted) * 30, 10, 10))
    keep = images.pick_best
    images.pick_best = lambda cfg, scene, tries, **k: (judged.append(scene), (len(tries) - 1, "the last one"))[1]
    try:
        text = "I am in a greenhouse in winter and the cat on the green couch is asleep among the tomato plants, and the car windows fog over with basil."
        scenes = [{"picture": "A black cat asleep on a green couch inside a winter greenhouse.", "images": [ids["cat"]]}]
        dry = sleep_once(m, cfg, llm=dreamer(scenes), steps=["dream"], rng=random.Random(1), now=now, dry_run=True, paint=paint)
        assert [(p["scene"], p.get("whole")) for p in dry["dreams"][0]["pictures"]] == [(WHOLE_CAPTION, True), (scenes[0]["picture"], None)]
        assert painted == [] and "draw" not in dry["dreams"][0]["pictures"][0]
        report = sleep_once(m, cfg, llm=dreamer(scenes), steps=["dream"], rng=random.Random(1), now=now, paint=paint)
        whole, moment = report["dreams"][0]["pictures"]
        assert not report["errors"] and whole["whole"] is True and whole["scene"] == WHOLE_CAPTION and whole["from"] == [] and "draw" not in whole
        # the dream itself is what is drawn, twice, from words alone; and what her choice is judged against
        assert painted[:2] == [(text + " soft light", None)] * 2 and painted[2][1] is not None and len(painted) == 4
        assert judged == [text, scenes[0]["picture"]] and whole["chosen"] == 2
        # the file name, not the path: pytest names the temporary folder after the test, which has "_whole_" in it
        assert "_whole_" in Path(whole["file"]).name and "_whole_" not in Path(moment["file"]).name and open(whole["file"], "rb").read()
        kept = dreams(m, 1)[0]["pictures"]
        assert [p.get("whole") for p in kept] == [True, None] and kept[0]["file"] == whole["file"]
        # a single scene she composes instead
        painted.clear(); judged.clear()
        report = sleep_once(m, dict(cfg, dream_image_whole="scene"), llm=dreamer(scenes), steps=["dream"], rng=random.Random(2), now=now + 9, paint=paint)
        whole = report["dreams"][0]["pictures"][0]
        assert whole["whole"] and whole["scene"].startswith("A winter greenhouse holding a green couch") and not report["errors"]
        assert painted[0] == (whole["scene"] + ", soft light", None) and judged[0] == whole["scene"]
        # and off: the moments only
        report = sleep_once(m, dict(cfg, dream_image_whole="off"), llm=dreamer(scenes), steps=["dream"], rng=random.Random(3), now=now + 20, paint=paint)
        assert [p.get("whole") for p in report["dreams"][0]["pictures"]] == [None]
    finally:
        images.pick_best = keep


def test_the_whole_picture_for_dreams_she_already_had(tmp_path):
    """A dream from before the picture of the whole dream existed can be given one, and nothing else about it changes."""
    import random
    from holonomic import images
    from holonomic.sleep import WHOLE_CAPTION, draw_whole, dreams, sleep_once
    m, ids, now = dream_store(tmp_path)
    cfg = dict(CFG, dream_images="pictures", dream_image_count=1, dream_image_style="")
    painted = []

    drawn = []

    def paint(prompt, start, size=None):
        painted.append(prompt)
        drawn.append(1)                                                          # every picture different, as real ones are
        return picture(768, 512, colour=(len(drawn) * 30, 10, 10))
    first = sleep_once(m, cfg, llm=dreamer(), steps=["dream"], rng=random.Random(1), now=now, paint=paint)["dreams"][0]
    second = sleep_once(m, cfg, llm=dreamer(), steps=["dream"], rng=random.Random(2), now=now + 9, paint=paint)["dreams"][0]
    assert len(first["pictures"]) == 1 and len(painted) == 2                     # whole is off in CFG: the moment only
    text, folder = m.get(first["id"])["text"], images.dream_folder(m, first["id"])
    painted.clear()
    told = []
    report = draw_whole(m, cfg, [first["id"], 99999, ids["cat"]], paint=paint, tell=told.append)
    assert not report["errors"] and report["missing"] == [99999, ids["cat"]] and report["had"] == []
    assert painted == [text] and len(report["drawn"]) == 1 and told == report["drawn"]       # the dream itself, and no model was asked
    got = report["drawn"][0]
    assert got["dream"] == first["id"] and got["whole"] and got["scene"] == WHOLE_CAPTION and folder.split("/")[-1] in got["file"].replace("\\", "/")
    kept = dreams(m, 2)
    mine = next(d for d in kept if d["id"] == first["id"])
    assert [p.get("whole") for p in mine["pictures"]] == [True, None] and mine["text"] == text
    assert [p.get("whole") for p in next(d for d in kept if d["id"] == second["id"])["pictures"]] == [None]
    # asked again it is passed over, unless told to draw another
    assert draw_whole(m, cfg, [first["id"], second["id"]], paint=paint)["had"] == [first["id"]] and len(painted) == 2
    assert len(draw_whole(m, cfg, [first["id"]], paint=paint, again=True)["drawn"]) == 1 and len(painted) == 3
    # the image server is off: said once, and the rest are not tried

    def broken(prompt, start, size=None):
        raise RuntimeError("Could not reach the image server")
    report = draw_whole(m, cfg, [first["id"], second["id"]], paint=broken, again=True)
    assert report["drawn"] == [] and len(report["errors"]) == 1 and "Could not reach the image server" in report["errors"][0]
    assert "switched off" in draw_whole(m, dict(cfg, dream_images="words"), [first["id"]], paint=paint)["errors"][0]


class _Patch:
    """What pytest's monkeypatch does, for the plain runner too."""

    def __init__(self):
        self.was = []

    def setattr(self, obj, name, value):
        self.was.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self.was):
            setattr(obj, name, value)


def _patched(fn):
    def run(tmp_path):
        mp = _Patch()
        try:
            fn(tmp_path, mp)
        finally:
            mp.undo()
    run.__name__ = fn.__name__
    return run


@_patched
def test_an_image_server_that_falls_over_and_comes_back_does_not_cost_the_pictures(tmp_path, monkeypatch):
    """ComfyUI crashed loading a model and was started again ten seconds later.  The picture being drawn failed as
    it went down, the next was refused while it came back, and a night's pictures were lost."""
    import random
    from holonomic import paint as P, sleep as S
    from holonomic.sleep import sleep_once
    m, ids, now = dream_store(tmp_path)
    cfg = dict(CFG, dream_images="pictures", dream_image_count=1, dream_image_style="", dream_image_candidates=1)
    waits, calls = [], []
    monkeypatch.setattr(S, "_wait", waits.append)

    def paint(prompt, start, size=None):
        calls.append(prompt)
        if len(calls) <= 2:
            raise P.Unreachable("Could not reach the image server at http://x: refused")
        return picture(768, 512, colour=(40, 10, 10))
    report = sleep_once(m, cfg, llm=dreamer(), steps=["dream"], rng=random.Random(1), now=now, paint=paint)
    assert len(report["dreams"][0]["pictures"]) == 1 and len(calls) == 3
    assert waits == [30, 60] and len(report["waited"]) == 2 and not [e for e in report["errors"] if "image server" in e]
    # still down after both waits: said, and the dream stands without pictures
    calls.clear(), waits.clear()

    def down(prompt, start, size=None):
        calls.append(prompt)
        raise P.Unreachable("Could not reach the image server at http://x: refused")
    report = sleep_once(m, cfg, llm=dreamer(), steps=["dream"], rng=random.Random(2), now=now + 9, paint=down)
    assert report["dreams"][0]["pictures"] == [] and len(calls) == 3 and waits == [30, 60]
    assert any("Could not reach" in e for e in report["errors"])
    # told not to wait; and a refusal that is not the server being away is never waited on
    calls.clear(), waits.clear()
    sleep_once(m, dict(cfg, dream_image_retry_wait=0), llm=dreamer(), steps=["dream"], rng=random.Random(3), now=now + 20, paint=down)
    assert len(calls) == 1 and waits == []

    def refuses(prompt, start, size=None):
        calls.append(prompt)
        raise P.PaintError("The image server returned 400")
    calls.clear()
    sleep_once(m, cfg, llm=dreamer(), steps=["dream"], rng=random.Random(4), now=now + 30, paint=refuses)
    assert len(calls) == 1 and waits == []


@_patched
def test_a_restarted_comfyui_is_noticed_and_not_waited_on_for_ten_minutes(tmp_path, monkeypatch):
    from holonomic import paint as P
    asked = []
    clock = [1000.0]
    monkeypatch.setattr(P.time, "time", lambda: clock[0])
    monkeypatch.setattr(P.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + 6))

    def post(url, body, ctype, timeout, raw=False):
        asked.append(url.rsplit("/", 2)[-2] if "/history/" in url else url.rsplit("/", 1)[-1])
        if url.endswith("/prompt"):
            return {"prompt_id": "job1"}
        if url.endswith("/queue"):
            return {"queue_running": [], "queue_pending": []}
        return {}                                             # history: it has never heard of the job
    monkeypatch.setattr(P, "_post", post)
    with pytest.raises(P.Unreachable, match="restarted"):
        P._comfy_run("http://x", {}, 600, 1)
    assert clock[0] - 1000 < 30 and "queue" in asked
    # a job that is still in the queue is waited for
    clock[0] = 1000.0
    done = []

    def busy(url, body, ctype, timeout, raw=False):
        if url.endswith("/prompt"):
            return {"prompt_id": "job1"}
        if url.endswith("/queue"):
            done.append(1)
            return {"queue_running": [[3, "job1", {}]], "queue_pending": []}
        if "/view" in url:
            return b"PICTURE"
        return {"job1": {"outputs": {"9": {"images": [{"filename": "a.png"}]}}}} if len(done) >= 2 else {}
    monkeypatch.setattr(P, "_post", busy)
    assert P._comfy_run("http://x", {}, 600, 1) == b"PICTURE"


def test_the_pictures_of_a_dream_that_lost_them_can_be_drawn_afterwards(tmp_path):
    import random
    from holonomic.sleep import draw_whole, dreams, sleep_once
    m, ids, now = dream_store(tmp_path)
    cfg = dict(CFG, dream_images="pictures", dream_image_count=1, dream_image_style="", dream_image_candidates=1,
               dream_image_whole="text", dream_image_retry_wait=0)

    def broken(prompt, start, size=None):
        raise RuntimeError("Could not reach the image server")
    lost = sleep_once(m, cfg, llm=dreamer(), steps=["dream"], rng=random.Random(1), now=now, paint=broken)["dreams"][0]
    assert lost["pictures"] == []
    text, n = m.get(lost["id"])["text"], [0]

    def paint(prompt, start, size=None):
        n[0] += 1
        return picture(768, 512, colour=(n[0] * 30, 10, 10))
    report = draw_whole(m, cfg, [lost["id"]], llm=dreamer(), paint=paint, moments=True)
    assert not report["errors"] and sorted(bool(p.get("whole")) for p in report["drawn"]) == [False, True]
    assert m.get(lost["id"])["text"] == text
    assert len(next(d for d in dreams(m, 3) if d["id"] == lost["id"])["pictures"]) == 2
    # it has them now: passed over, unless asked again, and then the picture of the whole dream is not repeated
    assert draw_whole(m, cfg, [lost["id"]], llm=dreamer(), paint=paint, moments=True)["had"] == [lost["id"]] and n[0] == 2
    more = draw_whole(m, cfg, [lost["id"]], llm=dreamer(), paint=paint, moments=True, again=True)
    assert [bool(p.get("whole")) for p in more["drawn"]] == [False] and n[0] == 3
