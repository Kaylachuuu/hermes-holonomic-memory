"""Looking into the plates: is the write log a true account of them, and what are they doing at recall?

Nothing here changes a store.

The write log (engine.py, table `bindings`) has a row for every term added to a plate.  `verify` adds a plate's
rows up again from nothing and compares the result with the plate as stored: if they agree, the log is enough to
rebuild that plate, and one of its terms could be taken out again exactly.

`diagnose` measures three things that were argued about before anyone had numbers:

- how many plates answer a cue, and whether the limit on how many are read ever turns any away;
- how much of what the plates carry belongs to memories that recall can no longer return;
- whether those memories, still counted among a plate's members when a read-out is attributed, change what is
  recalled.  More results would not by itself mean better ones, so the measure is whether the associates a cue is
  *known* to have are found: known from the write log, or from the sources a conclusion was stored with.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Set

OUT_OF_RECALL = 0.15          # the trust below which everyday recall leaves a memory out (provider.recall_options)


def _np():
    import numpy as np
    return np


def log_status(engine) -> dict:
    """How much of the store the write log covers."""
    db = engine._db
    with engine._lock:
        plates = db.execute("SELECT COUNT(*) AS n, COALESCE(SUM(logged), 0) AS logged FROM plates").fetchone()
        rows = db.execute("SELECT COUNT(*) AS n, MIN(created_at) AS first FROM bindings").fetchone()
        encodings = db.execute("SELECT COUNT(*) AS n FROM encodings").fetchone()["n"]
        started = engine._meta_get("log_started")
    return {"plates": int(plates["n"]), "plates_logged": int(plates["logged"]), "bindings": int(rows["n"]),
            "log_started": float(started) if started else None, "encodings": int(encodings), "encoding": int(engine._encoding)}


def verify(engine, tolerance: float = 1e-3) -> dict:
    """Rebuild every logged plate from the log alone and compare it with what is stored.

    A plate cannot be checked if one of its terms involves a memory that has since been forgotten (its vector is
    gone: that is what forgetting means) or was written under another encoding."""
    np = _np()
    db = engine._db
    report: Dict[str, Any] = {"checked": 0, "agree": 0, "disagree": [], "unlogged": 0, "cannot": [], "worst": 0.0, "tolerance": tolerance}
    with engine._lock:
        engine._sync()
        plates = db.execute("SELECT id, logged, trace, gate FROM plates ORDER BY id").fetchall()
        for plate in plates:
            if not plate["logged"]:
                report["unlogged"] += 1
                continue
            rows = db.execute("SELECT cue_memory, cue_key, target, weight, encoding FROM bindings WHERE plate_id = ? ORDER BY id",
                              (plate["id"],)).fetchall()
            trace = np.zeros(engine.dim, dtype=np.complex64)
            gate = np.zeros(engine.dim, dtype=np.complex64)
            missing = other = 0
            for b in rows:
                if int(b["encoding"]) != engine._encoding:
                    other += 1
                    continue
                target = engine._row.get(int(b["target"]))
                if b["cue_key"] is not None:
                    cue = engine._key_atom(b["cue_key"])
                else:
                    source = engine._row.get(int(b["cue_memory"]))
                    cue = engine._cue(engine._phasor_of(source)) if source is not None else None
                if target is None or cue is None:
                    missing += 1
                    continue
                w = np.complex64(b["weight"])
                trace += w * cue * engine._phasor_of(target)
                gate += w * cue
            if missing or other:
                report["cannot"].append({"plate": int(plate["id"]), "bindings": len(rows), "forgotten": missing, "other_encoding": other})
                continue
            stored_trace = np.frombuffer(plate["trace"], dtype=np.complex64)
            stored_gate = np.frombuffer(plate["gate"], dtype=np.complex64)

            def off(a, b) -> float:
                size = float(np.linalg.norm(b))
                return float(np.linalg.norm(a - b)) / size if size > 0 else float(np.linalg.norm(a))
            worst = max(off(trace, stored_trace), off(gate, stored_gate))
            report["checked"] += 1
            report["worst"] = max(report["worst"], worst)
            if worst <= tolerance:
                report["agree"] += 1
            else:
                report["disagree"].append({"plate": int(plate["id"]), "bindings": len(rows), "difference": worst})
    return report


def _known_links(engine, live: Set[int]) -> Dict[int, Set[int]]:
    """memory id -> the memories it is on record as having been bound to, among those recall can return.

    From the write log (exact), and from the sources a memory was stored with (every such link was written both
    ways).  Neighbours in a conversation are bound too, but nothing recorded it before the log, so they are not
    counted as known; `_probably_linked` has them, for explaining a result and not for expecting one."""
    db = engine._db
    known: Dict[int, Set[int]] = {}

    def add(a: int, b: int) -> None:
        if a in live and b in live and a != b:
            known.setdefault(a, set()).add(b)
    for b in db.execute("SELECT cue_memory, target FROM bindings WHERE cue_memory IS NOT NULL"):
        add(int(b["cue_memory"]), int(b["target"]))
    # Only what reflection stored: it binds a conclusion to every source it names.  An account of a conversation
    # names up to forty sources and is bound to two of them, so its list says nothing about what was written.
    for m in db.execute("SELECT id, meta FROM memories WHERE forgotten = 0 AND kind IN ('fact', 'project_fact', 'self_note', "
                        "'bond_note', 'insight') AND meta LIKE '%\"sources\"%'"):
        try:
            sources = json.loads(m["meta"] or "{}").get("sources") or []
        except ValueError:
            continue
        for s in sources:
            if isinstance(s, (int, float)):
                add(int(m["id"]), int(s))
                add(int(s), int(m["id"]))
    return known


def _probably_linked(engine) -> Dict[int, Set[int]]:
    """Links nothing recorded exactly: the line before and after in a conversation, and whatever a memory lists
    as its sources (an account of a conversation is bound to some of those it lists)."""
    out: Dict[int, Set[int]] = {}
    for m in engine._db.execute("SELECT id, meta FROM memories WHERE forgotten = 0 AND meta LIKE '%\"sources\"%'"):
        try:
            sources = [int(x) for x in (json.loads(m["meta"] or "{}").get("sources") or []) if isinstance(x, (int, float))]
        except ValueError:
            continue
        for x in sources:
            out.setdefault(int(m["id"]), set()).add(x)
            out.setdefault(x, set()).add(int(m["id"]))
    # An image and its parts, and the lines an image arrived with.
    by_image: Dict[int, List[int]] = {}
    for m in engine._db.execute("SELECT id, meta FROM memories WHERE forgotten = 0 AND meta LIKE '%\"image_id\"%'"):
        try:
            by_image.setdefault(int(json.loads(m["meta"] or "{}").get("image_id")), []).append(int(m["id"]))
        except (ValueError, TypeError):
            continue
    try:
        arrived = {int(r["id"]): [int(x) for x in (json.loads(r["meta"] or "{}").get("links") or []) if isinstance(x, (int, float))]
                   for r in engine._db.execute("SELECT id, meta FROM images")}
    except Exception:                                   # a store with no images in it
        arrived = {}
    for image_id, mids in by_image.items():
        for a in mids:
            for b in mids + arrived.get(image_id, []):
                if a != b:
                    out.setdefault(a, set()).add(b)
                    out.setdefault(b, set()).add(a)
    previous: Dict[tuple, int] = {}
    for m in engine._db.execute("SELECT id, realm, session FROM memories WHERE forgotten = 0 ORDER BY id"):
        key = (m["realm"], m["session"])
        if key in previous:
            out.setdefault(int(m["id"]), set()).add(previous[key])
            out.setdefault(previous[key], set()).add(int(m["id"]))
        previous[key] = int(m["id"])
    return out


def _median(values: List[float]) -> Optional[float]:
    values = sorted(values)
    return values[len(values) // 2] if values else None


def diagnose(engine, sample: int = 300, seed: int = 1, realm: str = "waking", cfg: Optional[Dict[str, Any]] = None) -> dict:
    np = _np()
    db = engine._db
    cfg = cfg or {}
    with engine._lock:
        engine._sync()
        code = engine._realm_codes.get(realm)
        if code is None:
            return {"cues": 0, "note": f"There is nothing in the '{realm}' realm."}
        codes = [code]
        rows_all = [r for r in range(engine._X.n) if engine._realm.a[r] == code and engine._trust.a[r] >= 0]
        live_rows = {r for r in rows_all if engine._trust.a[r] >= OUT_OF_RECALL}
        retired_rows = {r for r in rows_all if engine._trust.a[r] < OUT_OF_RECALL}
        ids = {r: int(engine._ids.a[r]) for r in rows_all}
        live_ids = {ids[r] for r in live_rows}
        logged = {int(p["id"]): bool(p["logged"]) for p in db.execute("SELECT id, logged FROM plates")}

        # ---- what the plates carry
        plates = []
        gone = {int(r["plate_id"]): int(r["n"]) for r in db.execute(
            "SELECT plate_id, COUNT(*) AS n FROM plate_members WHERE gone = 1 GROUP BY plate_id")}
        for i in range(engine._trace.n):
            if engine._prealm.a[i] != code:
                continue
            members = engine._members[i]
            plates.append({"plate": int(engine._pid.a[i]), "members": len(members), "logged": logged.get(int(engine._pid.a[i]), False),
                           "out_of_recall": len([r for r in members if r in retired_rows]),
                           "forgotten": gone.get(int(engine._pid.a[i]), 0), "load": float(engine._pload.a[i])})
        forgotten_total = int(db.execute("SELECT COUNT(*) AS n FROM memories WHERE forgotten != 0 AND realm = ?", (realm,)).fetchone()["n"])
        forgotten_placed = int(db.execute("SELECT COUNT(DISTINCT memory_id) AS n FROM plate_members WHERE gone = 1").fetchone()["n"])

        known = _known_links(engine, live_ids)
        probable = _probably_linked(engine)

        def sort_out(back: Set[int], cue_ids: List[int]) -> Dict[str, Set[int]]:
            """What came back, by what is on record about it: known to be bound to a cue, explained by something
            else on record, or explained by nothing."""
            expected: Set[int] = set()
            explained: Set[int] = set()
            for c in cue_ids:
                expected |= known.get(c, set())
                explained |= {i for i in probable.get(c, set()) if i in live_ids}
            return {"known": back & expected, "explained": (back & explained) - expected, "unexplained": back - expected - explained,
                    "expected": expected}

        def where(plate_ids: Set[int]) -> List[str]:
            """A return read from a plate with every write on record is one thing; from a plate written before
            the log, another.  Read from both, it counts under each."""
            return sorted({"logged" if logged.get(p) else "legacy" for p in plate_ids}) or ["legacy"]

        # ---- what a cue brings back off the plates, three ways
        cues = sorted(live_rows)
        random.Random(seed).shuffle(cues)
        cues = cues[:max(1, int(sample))]
        ways = {"as it is": {}, "without them": {"leave_out": retired_rows}, "no limit": {"max_plates": 10 ** 9}}
        totals = {w: {"expected": 0, "found": 0, "returned": 0, "explained": 0, "unexplained": 0} for w in ways}
        unexplained_by = {"legacy": 0, "logged": 0}
        returned_by = {"legacy": 0, "logged": 0}
        scores: Dict[str, List[float]] = {"known": [], "explained": [], "unexplained": []}
        resonated: List[int] = []
        turned_away = differ = 0
        changed: List[dict] = []
        for r in cues:
            cue = engine._cue(engine._phasor_of(r))
            got: Dict[str, Set[int]] = {}
            for way, options in ways.items():
                info: Dict[str, Any] = {}
                found = engine._probe(cue, codes, info=info, **options)
                # As read, not capped at 1 the way recall caps it: capped, a strong return and a very strong one look alike.
                strength = {ids[row]: float(sc) for row, sc in found.items() if row in live_rows and row != r}
                back = set(strength)
                got[way] = back
                sorted_out = sort_out(back, [ids[r]])
                t = totals[way]
                t["expected"] += len(sorted_out["expected"])
                t["found"] += len(sorted_out["known"])
                t["returned"] += len(back)
                t["explained"] += len(sorted_out["explained"])
                t["unexplained"] += len(sorted_out["unexplained"])
                if way == "as it is":
                    resonated.append(info.get("resonated", 0))
                    turned_away += 1 if info.get("turned_away", 0) else 0
                    read_from = {ids[row]: plate_ids for row, plate_ids in (info.get("plates") or {}).items() if row in ids}
                    for group in scores:
                        scores[group] += [strength[i] for i in sorted_out[group]]
                    for i in back:
                        for kind in where(read_from.get(i, set())):
                            returned_by[kind] += 1
                            if i in sorted_out["unexplained"]:
                                unexplained_by[kind] += 1
            if got["without them"] != got["as it is"]:
                differ += 1
                if len(changed) < 8:
                    gained, lost = got["without them"] - got["as it is"], got["as it is"] - got["without them"]
                    expected = known.get(ids[r], set())
                    changed.append({"cue": ids[r], "gained": sorted(gained), "lost": sorted(lost),
                                    "gained_expected": len(gained & expected), "lost_expected": len(lost & expected)})

        # ---- what she would be given: recall as a message does it, the memory standing in for the message
        k = int(cfg.get("recall_k", 6))
        options = {"k": k, "min_score": float(cfg.get("min_score", 0.2)), "min_trust": OUT_OF_RECALL,
                   "min_strength": float(cfg.get("fade_threshold", 0.35)), "skip_kinds": ("asked_user",),
                   "kind_weights": {"said_assistant": float(cfg.get("assistant_weight", 0.75)), "dreamtalk_assistant": 0.5,
                                    "dreamtalk_user": 0.75, "image_part": 0.9}}
        given = {"cues": 0, "given": 0, "by_likeness": 0, "by_association": 0, "known": 0, "explained": 0, "unexplained": 0,
                 "unexplained_legacy": 0, "unexplained_logged": 0, "examples": []}
        for r in cues:
            info = {}
            hits = [h for h in engine.recall(vector=engine._X.a[r].copy(), realms=(realm,), info=info, **options) if h.id != ids[r]]
            given["cues"] += 1
            given["given"] += len(hits)
            # It is there for its likeness to the message, or because the plates brought it: whichever counted for more.
            brought = [h for h in hits if 0.85 * h.assoc > h.direct]
            given["by_likeness"] += len(hits) - len(brought)
            given["by_association"] += len(brought)
            sorted_out = sort_out({h.id for h in brought}, info.get("hops") or [ids[r]])
            read_from = {ids[row]: plate_ids for row, plate_ids in (info.get("plates") or {}).items() if row in ids}
            for group in ("known", "explained", "unexplained"):
                given[group] += len(sorted_out[group])
            for i in sorted_out["unexplained"]:
                for kind in where(read_from.get(i, set())):
                    given["unexplained_" + kind] += 1
                if len(given["examples"]) < 6:
                    given["examples"].append({"cue": ids[r], "given": i, "read_from": where(read_from.get(i, set()))})
    resonated.sort()
    return {
        "cues": len(cues), "live": len(live_rows), "out_of_recall": len(retired_rows),
        "plates": {"count": len(plates), "logged": len([p for p in plates if p["logged"]]),
                   "with_out_of_recall": len([p for p in plates if p["out_of_recall"]]),
                   "members": sum(p["members"] for p in plates), "members_out_of_recall": sum(p["out_of_recall"] for p in plates),
                   "worst": sorted(plates, key=lambda p: -(p["out_of_recall"] + p["forgotten"]))[:5]},
        "forgotten": {"total": forgotten_total, "on_record": forgotten_placed, "plates_unknown": max(0, forgotten_total - forgotten_placed)},
        "resonance": {"median": resonated[len(resonated) // 2] if resonated else 0, "most": resonated[-1] if resonated else 0,
                      "limit": 24, "cues_with_plates_turned_away": turned_away},
        "known_links": {"cues_with_any": len([r for r in cues if known.get(ids[r])]), "from_log": int(
            db.execute("SELECT COUNT(*) AS n FROM bindings WHERE cue_memory IS NOT NULL").fetchone()["n"])},
        "recovery": totals,
        "unexplained_by_plate": unexplained_by, "returned_by_plate": returned_by,
        "strength": {group: {"count": len(v), "median": _median(v)} for group, v in scores.items()},
        "given": given,
        "cues_where_leaving_them_out_changed_the_result": differ, "examples": changed,
    }
