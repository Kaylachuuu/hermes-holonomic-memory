"""Self-test against a real Ollama embedding model.

    python3 selftest.py                      # nomic-embed-text on localhost:11434
    python3 selftest.py --model NAME --host http://HOST:11434

Writes only to a temporary folder, which it deletes afterwards.
Paste the whole output back to Claude.
"""
import argparse, os, platform, shutil, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _bootstrap  # noqa: F401
import numpy as np
from holonomic import HolonomicMemory, OllamaEmbedder, HashEmbedder, EmbeddingError

# (topic turns 0-1, an off-topic aside, topic turns 3-4, a paraphrase of turn 1)
CONVS = [
    (["I want to repot the monstera this weekend because its roots are circling the pot.",
      "The potting mix should drain fast, so I'm adding perlite and orchid bark to it.",
      "By the way, my cousin Dario finally passed his driving test on the fourth try.",
      "After repotting I'll hold off watering for a few days so the roots can settle.",
      "The old pot can go to the fiddle leaf fig once I've scrubbed it out."],
     "what goes into the soil blend so the plant drains quickly?"),
    (["The backup job on the file server failed again last night with a timeout.",
      "It turns out the snapshot was taking forty minutes because the disk is nearly full.",
      "Unrelated, but the bakery on Elm Street started selling cardamom buns.",
      "I'm going to prune the old snapshots and move the archive to the second array.",
      "Once that's done the nightly job should finish in under ten minutes."],
     "why was the server snapshot so slow?"),
    (["I've been learning the cello for about six months now.",
      "My teacher says my bow hold is too tense and it's making the tone scratchy.",
      "Oh, and I found my grandfather's old pocket watch in the attic yesterday.",
      "She gave me an exercise with long open strings to relax my right hand.",
      "I'm hoping to play a simple Bach minuet by the spring recital."],
     "what did the music teacher say is wrong with how I hold the bow?"),
    (["We're planning a hiking trip to the Wind River Range in August.",
      "The route crosses two passes above eleven thousand feet, so we need to acclimatise first.",
      "Side note, the landlord is replacing all the windows in the building next month.",
      "I'm packing a bear canister because food storage rules are strict there.",
      "We'll resupply at the trailhead on day four before the second half."],
     "how high are the mountain passes on the trek?"),
    (["I'm trying to cut down on caffeine because I haven't been sleeping well.",
      "I switched my afternoon coffee to rooibos tea and the headaches lasted three days.",
      "Random thought, I still owe Priya forty dollars for the concert tickets.",
      "Falling asleep is easier now, though I still wake up around four.",
      "Next step is no screens for an hour before bed."],
     "what did I replace my afternoon coffee with?"),
    (["The sourdough starter finally doubled in six hours after a week of feeding.",
      "I'm using a mix of rye and bread flour because the rye ferments faster.",
      "Incidentally, the neighbours' dog is called Biscuit and he hates the mail carrier.",
      "The first loaf came out dense, so I'll proof it longer next time.",
      "I also need a hotter oven and a lidded pot to get a proper crust."],
     "which flours am I feeding the starter with?"),
    (["I've started sketching the plot for a short story set on a tidal island.",
      "The main character is a lighthouse keeper who records every ship that passes.",
      "Before I forget, the car's inspection sticker expires at the end of the month.",
      "One night a ship appears in her log that she has no memory of writing down.",
      "I think the ending should leave it unclear whether the ship was real."],
     "who is the protagonist of the story I'm writing?"),
    (["My budget spreadsheet shows we overspent on groceries by a lot in September.",
      "Most of the overage came from buying lunch out instead of packing it.",
      "Also, Aunt Rosalind is visiting from Halifax the week after next.",
      "I set a weekly cap and I'm tracking it in a separate column.",
      "If we stay under the cap for two months, the difference goes to the travel fund."],
     "what caused most of the extra food spending?"),
]
UNRELATED = [c[0][i] for c in CONVS for i in (0,)] + [c[0][2] for c in CONVS[:4]]


class Timed:
    """Wraps an embedder and adds up the time spent inside it."""
    def __init__(self, inner): self.inner, self.seconds, self.calls = inner, 0.0, 0
    def __getattr__(self, name): return getattr(self.inner, name)
    def embed(self, texts, kind="document"):
        t0 = time.perf_counter()
        try:
            return self.inner.embed(texts, kind)
        finally:
            self.seconds += time.perf_counter() - t0; self.calls += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--hash", action="store_true", help="use the built-in test embedder instead of Ollama")
    a = ap.parse_args()
    print(f"holonomic self-test v3 | python {platform.python_version()} | numpy {np.__version__} | {platform.system()}")
    emb = Timed(HashEmbedder() if a.hash else OllamaEmbedder(a.model, a.host))
    problems = []

    # 1. embedder
    try:
        t0 = time.perf_counter(); v = emb.embed(["hello there"], "query"); first = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); emb.embed(["hello there"], "query"); single = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter(); emb.embed(UNRELATED, "document"); batch = (time.perf_counter() - t0) * 1000
    except EmbeddingError as exc:
        print(f"\nFAIL: could not get embeddings.\n  {exc}\n  Check that Ollama is running and that `ollama list` shows the model name you passed.")
        return 1
    print(f"\n[1] embedder {emb.signature}: {v.shape[1]} dimensions, prefixes {getattr(emb, 'prefixes', None)}")
    print(f"    first call {first:.0f} ms, warm single {single:.0f} ms, batch of {len(UNRELATED)} {batch:.0f} ms")

    tmp = tempfile.mkdtemp(prefix="holonomic-selftest-")
    try:
        mem = HolonomicMemory(tmp, emb)
        # 2. how similar are unrelated texts, before and after centring?
        raw = emb.embed(UNRELATED, "document"); raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        cen = mem.proj.prep(emb.embed(UNRELATED, "document"))
        off = lambda m: float((m @ m.T)[~np.eye(len(m), dtype=bool)].mean())
        print(f"\n[2] similarity between unrelated sentences: raw {off(raw):.3f}, after centring {off(cen):.3f}")
        if off(cen) > 0.25:
            problems.append("unrelated texts still look similar after centring")

        # 3. store the conversations, then query
        emb.seconds = 0.0; t0 = time.perf_counter(); ids = []
        for n, (turns, _) in enumerate(CONVS):
            ids.append([mem.remember(t, session=f"conv{n}")[0] for t in turns])
        n_mem = sum(len(c[0]) for c in CONVS)
        write = (time.perf_counter() - t0) * 1000 / n_mem
        write_embed = emb.seconds * 1000 / n_mem
        print(f"\n[3] stored {mem.stats()['memories']} memories in {mem.stats()['plates']} plate(s), {write:.0f} ms each ({write_embed:.0f} ms of that embedding) "
              f"(plate load {mem.stats()['mean_plate_load']:.0f} of {mem.stats()['plate_capacity']:.0f})")
        print("    conv | top hit correct | query~turn (rank) | aside: direct  assoc | found | false assoc")
        ok_direct = ok_aside = false_total = 0; times = []; emb.seconds = 0.0
        for n, (turns, query) in enumerate(CONVS):
            t0 = time.perf_counter(); hits = mem.recall(query, k=40, min_score=0.0); times.append((time.perf_counter() - t0) * 1000)
            by = {h.id: h for h in hits}
            top_ok = hits[0].id == ids[n][1]
            aside = by.get(ids[n][2])
            found = bool(aside and aside.assoc > 0.15)
            false = sum(1 for h in hits if h.id not in ids[n] and h.assoc > 0.15)
            ok_direct += top_ok; ok_aside += found; false_total += false
            rank = 1 + sum(1 for h in hits if h.direct > by[ids[n][1]].direct)
            print(f"    {n:>4} | {str(top_ok):>15} | {by[ids[n][1]].direct:>10.2f} ({rank:>4}) | {aside.direct if aside else 0:>13.2f} "
                  f"{aside.assoc if aside else 0:>6.2f} | {str(found):>5} | {false}")
        print(f"    top hit correct {ok_direct}/{len(CONVS)}, off-topic aside recovered {ok_aside}/{len(CONVS)}, "
              f"false associations {false_total}")
        if ok_direct < len(CONVS) - 1: problems.append("direct recall missed more than one query")
        if ok_aside < len(CONVS) - 2: problems.append("associative recall missed more than two asides")
        if false_total > 4: problems.append("too many false associations")

        # 4. default recall as Hermes would see it, and timings
        hits = mem.recall(CONVS[1][1], k=4)
        print(f"\n[4] default recall for: {CONVS[1][1]!r}")
        for h in hits:
            print(f"    score {h.score:.2f} (direct {h.direct:.2f}, assoc {h.assoc:.2f})  {h.text[:70]}")
        print(f"    recall incl. embedding: median {np.median(times):.0f} ms, max {max(times):.0f} ms "
              f"({emb.seconds * 1000 / len(CONVS):.0f} ms of that embedding)")

        # timing breakdown of the pieces that involve no model
        def clock(fn, n=20):
            fn(); t0 = time.perf_counter()
            for _ in range(n): fn()
            return (time.perf_counter() - t0) * 1000 / n
        vec = mem._X.a[0].copy()
        plate = mem._trace.a[:1]
        print(f"\n[4b] cpu count {os.cpu_count()}; projection {clock(lambda: mem.proj.phasor(vec[None])):.2f} ms, "
              f"back-projection {clock(lambda: mem.proj.back(plate)):.2f} ms, "
              f"recall without embedding {clock(lambda: mem.recall(vector=vec, k=8)):.2f} ms")
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
            print("     math libraries:", [(i.get("internal_api"), i.get("num_threads")) for i in threadpool_info()])
            with threadpool_limits(limits=1):
                print(f"     single-threaded: projection {clock(lambda: mem.proj.phasor(vec[None])):.2f} ms, "
                      f"recall without embedding {clock(lambda: mem.recall(vector=vec, k=8)):.2f} ms")
        except ImportError:
            print("     (threadpoolctl not installed; `pip install threadpoolctl` adds a threading check)")

        # 5. restart
        before = [(h.id, round(h.score, 3)) for h in hits]
        mem.close(); mem = HolonomicMemory(tmp, emb)
        same = [(h.id, round(h.score, 3)) for h in mem.recall(CONVS[1][1], k=4)] == before
        print(f"\n[5] identical results after closing and reopening: {same}")
        if not same: problems.append("results changed after reopening")
        mem.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT: " + ("PASS" if not problems else "NEEDS TUNING: " + "; ".join(problems)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
