"""Measure what the plates recover, against plate load, plus recall latency.

Scenario: conversations of 6 turns.  Turns in a conversation are topically
similar to each other.  One turn per conversation is an off-topic aside that
only its neighbours can lead to.  Queries are paraphrases of a turn.
"""
import sys, tempfile, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
from conftest import StructEmbedder
from holonomic import HolonomicMemory

TURNS = 6


def build(convs, capacity, dim=4096, keys=False):
    emb = StructEmbedder()
    m = HolonomicMemory(tempfile.mkdtemp(), emb, dim=dim, plate_capacity=capacity)
    ids = {}
    t0 = time.time()
    for c in range(convs):
        d = c % 40
        for t in range(TURNS):
            path = f"aside{c}" if t == 3 else f"d{d}/c{c}/t{t}"
            ids[(c, t)] = m.remember(f"{path} original wording", session=f"s{c}",
                                     keys=[f"person{c % 200}"] if keys else [])[0]
    return m, emb, ids, (time.time() - t0) / (convs * TURNS) * 1000


def accuracy(capacity, convs=600, **kw):
    m, emb, ids, write_ms = build(convs, capacity)
    rng = np.random.default_rng(1)
    sample = rng.choice(convs, 200, replace=False)
    aside_hit = 0
    nb, far, other = [], [], []
    for c in sample:
        d = c % 40
        # paraphrase of turn 2, whose neighbours are turn 1 (on-topic) and turn 3 (the aside)
        res = {h.id: h for h in m.recall(f"d{d}/c{c}/t2 asked again differently", k=50, min_score=0.0, **kw)}
        aside = res.get(ids[(c, 3)])
        aside_hit += bool(aside and aside.assoc > 0.15)
        nb.append(res[ids[(c, 1)]].assoc if ids[(c, 1)] in res else 0.0)
        far.append(np.mean([res[ids[(c, t)]].assoc if ids[(c, t)] in res else 0.0 for t in (0, 5)]))
        mine = {ids[(c, t)] for t in range(TURNS)}
        other.append(sum(1 for h in res.values() if h.id not in mine and h.assoc > 0.15))
    s = m.stats(); m.close()
    return dict(capacity=capacity, mem_per_plate=round(s["memories"] / s["plates"], 1),
                aside_recovered=round(aside_hit / len(sample), 3),
                neighbour_assoc=round(float(np.mean(nb)), 3), same_conv_nonneighbour=round(float(np.mean(far)), 3),
                false_assoc_per_query=round(float(np.mean(other)), 3), write_ms=round(write_ms, 2))


def latency(convs, capacity):
    m, emb, ids, _ = build(convs, capacity, keys=True)
    qs = m.proj.prep(emb.embed([f"d{c % 40}/c{c}/t2 asked again" for c in range(0, convs, max(1, convs // 100))], "query"))
    times = []
    for q in qs:
        t0 = time.perf_counter(); m.recall(vector=q, k=8); times.append((time.perf_counter() - t0) * 1000)
    s = m.stats(); m.close()
    return dict(memories=s["memories"], plates=s["plates"], median_ms=round(float(np.median(times)), 1),
                p95_ms=round(float(np.percentile(times, 95)), 1),
                plate_MB=round(s["plate_bytes"] / 1e6, 1), cleanup_MB=round(s["cleanup_bytes"] / 1e6, 1))


if __name__ == "__main__":
    caps = [float(a) for a in sys.argv[1:]] or [48, 96, 192, 384]
    for cap in caps:
        print(accuracy(cap))
