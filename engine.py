"""Holonomic memory engine.

Storage model
-------------
* A **plate** is one complex vector holding many associations superposed on top
  of each other: for each association, `plate += weight * cue * target`.
  Which cue leads to which target is recorded nowhere else.
* Each plate also keeps a **gate**: the plain sum of its cues.  Comparing a cue
  with every gate is a cheap way to see which plates will resonate.
* The **cleanup memory** holds one embedding per memory plus its text.  Plates
  return a noisy blend; cleanup turns that blend back into actual memories.

Associations written for every memory
-------------------------------------
* it <-> the previous memory in the same session (both directions),
* it <-> any memory it is explicitly linked to (e.g. a reflection and its sources),
* each discrete key (an entity, a topic) -> it.

A memory with no neighbour, link or key has nothing to associate with; it lives
in the cleanup memory only and is found by direct similarity.

Recall
------
1. Direct: exact similarity between the query and the cleanup memory.
2. Associative: the query (plus its best direct matches) is used as a cue.
   Resonating plates are unbound with it, the result is projected back into
   embedding space, and cleaned up against that plate's members.  This surfaces
   memories that are *linked to* things like the query without being similar
   to the query themselves.

Realms keep kinds of experience apart ("waking", "dream"): recall only ever
probes the realms it is asked for, so dreams cannot leak into factual recall.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import vsa

SCHEMA_VERSION = 1

# On a many-core machine the math library's worker threads fight with whatever
# else is busy (measured: 109 ms per recall with 16 threads, 21 ms with one), and
# the arrays here are too small to benefit from them.  Hold it to one thread
# while the engine runs.
try:
    from threadpoolctl import ThreadpoolController
    _BLAS = ThreadpoolController()

    def _single_threaded():
        return _BLAS.limit(limits=1, user_api="blas")
except Exception:                                    # threadpoolctl missing or unusable
    import contextlib

    def _single_threaded():
        return contextlib.nullcontext()


def _locked(method):
    """Run an engine method under the lock, single-threaded, on fresh state."""
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock, _single_threaded():
            self._sync()
            return method(self, *args, **kwargs)
    return wrapper

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB);
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'episodic',
    realm      TEXT NOT NULL DEFAULT 'waking',
    session    TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    strength   REAL NOT NULL DEFAULT 1.0,
    trust      REAL NOT NULL DEFAULT 0.5,
    recalls    INTEGER NOT NULL DEFAULT 0,
    meta       TEXT NOT NULL DEFAULT '{}',
    vec        BLOB,
    forgotten  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(realm, session, id);
CREATE INDEX IF NOT EXISTS idx_mem_kind ON memories(kind, id);
CREATE TABLE IF NOT EXISTS plates (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    realm      TEXT NOT NULL,
    load       REAL NOT NULL DEFAULT 0,
    sealed     INTEGER NOT NULL DEFAULT 0,
    gain       REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    trace      BLOB,
    gate       BLOB
);
CREATE TABLE IF NOT EXISTS profiles (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    who        TEXT NOT NULL,
    text       TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS plate_members (
    plate_id  INTEGER NOT NULL,
    memory_id INTEGER NOT NULL,
    PRIMARY KEY (plate_id, memory_id)
) WITHOUT ROWID;
-- How phasors and cues are built: the projection, the centre, the cue's permutations and role, and the formula.
-- A write can be reproduced only by the encoding it was made with, so each binding names its own.
CREATE TABLE IF NOT EXISTS encodings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT NOT NULL UNIQUE,
    created_at  REAL NOT NULL
);
-- The write log: one row for every term ever added to a plate.  trace += weight * cue * phasor(target), and
-- gate += weight * cue, where the cue is built from a memory (cue_memory) or from a key's text (cue_key).
-- Written in the same transaction as the plate it describes, so the two cannot disagree.
CREATE TABLE IF NOT EXISTS bindings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_id   INTEGER NOT NULL,
    cue_memory INTEGER,
    cue_key    TEXT,
    target     INTEGER NOT NULL,
    weight     REAL NOT NULL,
    encoding   INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bind_plate ON bindings(plate_id, id);
CREATE INDEX IF NOT EXISTS idx_bind_target ON bindings(target);
CREATE INDEX IF NOT EXISTS idx_bind_cue ON bindings(cue_memory);
"""

# The cue formula in _cue().  Change the formula, change this: it is part of an encoding's fingerprint.
CUE_FORMULA = "selfbind(perm2)-perm-role/1"

# Generic sentences used once, at creation, to estimate the component every
# embedding shares.  Fixed forever afterwards so stored phasors stay valid.
_CALIBRATION = [
    "The weather was cold and it rained most of the morning.",
    "She fixed the bug in the database migration script.",
    "We talked about what to cook for dinner tonight.",
    "The stock market fell sharply after the announcement.",
    "He plays guitar in a small band on weekends.",
    "I need to renew my passport before the trip.",
    "The cat knocked a glass off the kitchen table.",
    "Quantum computers use qubits instead of classical bits.",
    "My grandmother grew tomatoes in her garden every summer.",
    "The meeting was moved to Thursday at three o'clock.",
    "Install the package and restart the server.",
    "They felt anxious about the upcoming exam.",
    "The novel follows three generations of one family.",
    "What do you think happens after we die?",
    "The GPU ran out of memory during training.",
    "He apologised for forgetting her birthday.",
    "The river floods the valley every spring.",
    "I prefer tea to coffee in the afternoon.",
    "The function returns null when the list is empty.",
    "We watched the sunset from the top of the hill.",
    "The doctor recommended more sleep and less caffeine.",
    "Ancient Rome built roads across its whole empire.",
    "She laughed so hard she started crying.",
    "The invoice is due at the end of the month.",
    "I had a strange dream about flying over the ocean.",
    "The team shipped the new feature last night.",
    "Please remember to water the plants while I'm away.",
    "Music helps me concentrate when I work.",
    "The children built a fort out of blankets.",
    "Who are you, and what do you want to become?",
    "The train was delayed by forty minutes.",
    "I'm proud of how much you've grown this year.",
]

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(text, content=memories, content_rowid=id, tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
"""

# Words that carry no topic.  What remains of a query is matched against the text
# of the memories, so "what is my name" can find "my name is Kayla" through the
# word they share even though, as vectors, a question and its answer are far apart.
_STOPWORDS = frozenset("""a about after again all also am an and any are as at be because been before being but by can
could did do does doing don down each few for from further had has have having he her here hers him his how i if in
into is it its just me more most my no nor not now of off on once only or other our out over own same she should so
some such than that the their them then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours tell said say know remember recall think
please thing things something anything really got get like want going one ever us im ive id""".split())


def content_terms(text: str, limit: int = 12) -> list[str]:
    seen: dict[str, None] = {}
    for tok in re.findall(r"[a-z0-9]+", text.lower().replace("'", "")):
        if len(tok) > 1 and tok not in _STOPWORDS:
            seen.setdefault(tok)
    return list(seen)[:limit]


ROLE_ASSOC = "role:assoc"
_CUE_PERM = "cue"


@dataclass
class Recollection:
    id: int
    text: str
    kind: str
    realm: str
    score: float           # final ranking score
    direct: float          # similarity to the query itself
    assoc: float           # strength recovered from the plates
    strength: float
    trust: float
    created_at: float
    session: str = ""
    meta: dict = field(default_factory=dict)
    lexical: float = 0.0       # share of the score that came from matching words
    as_query: float = 0.0      # similarity when the query is embedded as a search query
    as_statement: float = 0.0  # similarity when the query is embedded as a statement

    def to_dict(self) -> dict:
        return asdict(self)


class _Grow:
    """Append-only numpy array with amortised doubling."""

    def __init__(self, width: int, dtype, cap: int = 64):
        self.a = np.zeros((cap, width) if width else (cap,), dtype=dtype)
        self.n = 0

    def add(self, row=None) -> int:
        if self.n == len(self.a):
            self.a = np.concatenate([self.a, np.zeros_like(self.a)])
        if row is not None:
            self.a[self.n] = row
        self.n += 1
        return self.n - 1

    @property
    def v(self) -> np.ndarray:
        return self.a[:self.n]


def split_text(text: str, max_chars: int) -> list[str]:
    """Split long text at paragraph, then sentence, boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > max_chars:
                cut = sent.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                pieces.append(sent[:cut].strip())
                sent = sent[cut:].strip()
            if sent:
                pieces.append(sent)
    chunks, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + 1 + len(piece) > max_chars:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def split_sentences(text: str, max_chars: int = 320, min_chars: int = 12) -> list[str]:
    """One unit per sentence, so each fact gets its own vector.  A message that
    states a name, an age and a job in one breath is otherwise stored as a blend
    that matches a question about any one of them only weakly.  Fragments shorter
    than `min_chars` ("Okay!") are attached to the sentence that follows."""
    units: list[str] = []
    carry = ""
    for para in re.split(r"\n+", text.strip()):
        for sent in re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", para.strip()):
            sent = sent.strip()
            if not sent:
                continue
            sent = f"{carry} {sent}".strip()
            carry = ""
            if len(sent) < min_chars:
                carry = sent
                continue
            units.extend(split_text(sent, max_chars))
    if carry:
        if units and len(units[-1]) + 1 + len(carry) <= max_chars:
            units[-1] = f"{units[-1]} {carry}"
        else:
            units.append(carry)
    return units


def normalize_key(key: str) -> str:
    return re.sub(r"\s+", " ", key.strip().lower())


MIN_WEIGHT, MAX_WEIGHT = 0.25, 2.0          # how faintly and how brightly a memory may be written
# One association written at weight w puts w squared of energy on a plate (cue and target are unit phasors).
# A plate must hold at least one at full weight.
MIN_PLATE_CAPACITY = MAX_WEIGHT ** 2


class HolonomicMemory:
    def __init__(self, path: str | Path, embedder, *, dim: int = 4096, plate_capacity: float = 128.0,
                 max_chars: int = 1200):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.max_chars = max_chars
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path / "holonomic.db", check_same_thread=False, timeout=10.0,
                                   isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._closed = False
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")      # safe with WAL; avoids a disk flush per write
            self._db.executescript(_SCHEMA)
            try:
                had_fts = bool(self._db.execute("SELECT 1 FROM sqlite_master WHERE name = 'memories_fts'").fetchone())
                self._db.executescript(_FTS_SCHEMA)
                if not had_fts:                              # store created before word matching existed
                    self._db.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
                self._fts = True
            except sqlite3.OperationalError:                 # SQLite built without FTS5
                self._fts = False
            self._init_meta(dim, plate_capacity)
            self._migrate()
            self._load()
        except BaseException:
            # e.g. the embedding server is down during first-time calibration.  Without
            # this the open handle leaks, and on Windows the file stays locked.
            self._closed = True
            self._db.close()
            raise

    # ------------------------------------------------------------------ setup

    def _meta_get(self, key: str):
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _meta_set(self, key: str, value) -> None:
        self._db.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                         (key, value))

    def _init_meta(self, dim: int, plate_capacity: float) -> None:
        proj_path = self.path / "projection.npy"
        stored = self._meta_get("embedder")
        if stored is None:
            if not float(plate_capacity) >= MIN_PLATE_CAPACITY:      # before anything is created
                raise ValueError(f"plate_capacity is {float(plate_capacity):g}; it must be at least {MIN_PLATE_CAPACITY:g}, "
                                 "so that a single association written at full weight fits on an empty plate. The default is 128.")
            calib = np.asarray(self.embedder.embed(_CALIBRATION, "document"), dtype=np.float32)
            embed_dim = int(calib.shape[1])
            center = calib.mean(axis=0)
            np.save(proj_path, vsa.make_projection(embed_dim, dim))
            self._db.execute("BEGIN")
            for key, value in (("schema", str(SCHEMA_VERSION)), ("embedder", self.embedder.signature),
                               ("embed_dim", str(embed_dim)), ("dim", str(dim)),
                               ("plate_capacity", str(plate_capacity)), ("center", center.tobytes())):
                self._meta_set(key, value)
            self._db.execute("COMMIT")
        elif stored != self.embedder.signature:
            raise ValueError(f"This memory was built with embedder '{stored}' but is being opened with "
                             f"'{self.embedder.signature}'. Vectors from different models are not comparable.")
        self.dim = int(self._meta_get("dim"))
        self.embed_dim = int(self._meta_get("embed_dim"))
        self.plate_capacity = float(self._meta_get("plate_capacity"))
        if not self.plate_capacity >= MIN_PLATE_CAPACITY:        # also turns away a value that is not a number
            raise ValueError(f"This store's plate capacity is {self.plate_capacity:g}, below the least a plate can have "
                             f"({MIN_PLATE_CAPACITY:g}): a single association written at full weight would not fit on an empty plate.")
        center = np.frombuffer(self._meta_get("center"), dtype=np.float32)
        if not proj_path.exists():
            raise FileNotFoundError(f"{proj_path} is missing; the plates cannot be read without it.")
        self.proj = vsa.Projector(np.load(proj_path), center)
        self._perm = vsa.permutation(_CUE_PERM, self.dim)
        self._perm2 = vsa.permutation(_CUE_PERM + "2", self.dim)
        self._role = vsa.atom(ROLE_ASSOC, self.dim)

    def _fingerprint(self) -> str:
        """What a write depends on besides the two memories: change any of it and old writes cannot be reproduced."""
        import hashlib
        h = hashlib.sha256()
        for part in (CUE_FORMULA.encode(), str(self.dim).encode(), np.ascontiguousarray(self.proj.P).tobytes(),
                     np.ascontiguousarray(self.proj.center).tobytes(), np.ascontiguousarray(self._perm).tobytes(),
                     np.ascontiguousarray(self._perm2).tobytes(), np.ascontiguousarray(self._role).tobytes()):
            h.update(part)
        return h.hexdigest()

    def _migrate(self) -> None:
        """Bring a store made by an earlier version up to date, and find this store's encoding."""
        columns = lambda table: {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}
        self._db.execute("BEGIN IMMEDIATE")
        try:
            if "logged" not in columns("plates"):
                # 1 = every write to this plate is in the write log.
                self._db.execute("ALTER TABLE plates ADD COLUMN logged INTEGER NOT NULL DEFAULT 0")
            if "gone" not in columns("plate_members"):
                # 1 = the memory has been forgotten; kept so that what a plate still carries can be counted.
                self._db.execute("ALTER TABLE plate_members ADD COLUMN gone INTEGER NOT NULL DEFAULT 0")
            if self._meta_get("log_started") is None:
                # Plates written before there was a log are closed as they stand: no plate then holds some
                # writes that are recorded and some that are not.
                self._db.execute("UPDATE plates SET sealed = 1 WHERE logged = 0 AND sealed = 0")
                self._meta_set("log_started", str(time.time()))
            fingerprint = self._fingerprint()
            row = self._db.execute("SELECT id FROM encodings WHERE fingerprint = ?", (fingerprint,)).fetchone()
            if row is None:
                row = {"id": self._db.execute("INSERT INTO encodings (fingerprint, created_at) VALUES (?, ?)",
                                              (fingerprint, time.time())).lastrowid}
            self._encoding = int(row["id"])
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def _load(self) -> None:
        self._X = _Grow(self.embed_dim, np.float32)      # cleanup memory
        self._ids = _Grow(0, np.int64)
        self._realm = _Grow(0, np.int16)
        self._strength = _Grow(0, np.float32)
        self._trust = _Grow(0, np.float32)
        self._kind = _Grow(0, np.int16)
        self._row: dict[int, int] = {}                   # memory id -> row
        self._realm_codes: dict[str, int] = {}
        self._kind_codes: dict[str, int] = {}
        for r in self._db.execute("SELECT id, realm, kind, strength, trust, vec FROM memories WHERE forgotten = 0 ORDER BY id"):
            self._add_row(r["id"], r["realm"], r["kind"], r["strength"], r["trust"],
                          np.frombuffer(r["vec"], dtype=np.float16).astype(np.float32))

        self._trace = _Grow(self.dim, np.complex64, cap=8)
        self._gate = _Grow(self.dim, np.complex64, cap=8)
        self._pid = _Grow(0, np.int64, cap=8)
        self._prealm = _Grow(0, np.int16, cap=8)
        self._pload = _Grow(0, np.float32, cap=8)
        self._pgain = _Grow(0, np.float32, cap=8)
        self._pgate = _Grow(0, np.float32, cap=8)        # energy of each gate vector
        self._members: list[set[int]] = []               # plate index -> rows
        self._open: dict[int, int] = {}                  # realm code -> open plate index
        pidx: dict[int, int] = {}
        for p in self._db.execute("SELECT * FROM plates ORDER BY id"):
            i = self._trace.add(np.frombuffer(p["trace"], dtype=np.complex64))
            self._gate.add(np.frombuffer(p["gate"], dtype=np.complex64))
            self._pid.add(p["id"])
            code = self._realm_code(p["realm"])
            self._prealm.add(code)
            self._pload.add(p["load"])
            self._pgain.add(p["gain"])
            self._pgate.add(float(np.mean(np.abs(self._gate.a[i]) ** 2)))
            self._members.append(set())
            pidx[p["id"]] = i
            if not p["sealed"]:
                self._open[code] = i
        for m in self._db.execute("SELECT plate_id, memory_id FROM plate_members WHERE gone = 0"):
            row = self._row.get(m["memory_id"])
            if row is not None and m["plate_id"] in pidx:
                self._members[pidx[m["plate_id"]]].add(row)
        self._last: dict[tuple[str, str], int | None] = {}
        self._dv = self._data_version()

    def _data_version(self) -> int:
        return int(self._db.execute("PRAGMA data_version").fetchone()[0])

    def _sync(self) -> None:
        """Reload if another process has written to the store since we last looked.
        Plates are read-modify-write, so working from a stale copy would silently
        drop the other process's associations."""
        if self._data_version() != self._dv:
            self._load()

    def _realm_code(self, realm: str) -> int:
        return self._realm_codes.setdefault(realm, len(self._realm_codes))

    def _add_row(self, mid: int, realm: str, kind: str, strength: float, trust: float, x: np.ndarray) -> int:
        row = self._X.add(x)
        self._ids.add(mid)
        self._realm.add(self._realm_code(realm))
        self._kind.add(self._kind_codes.setdefault(kind, len(self._kind_codes)))
        self._strength.add(strength)
        self._trust.add(trust)
        self._row[mid] = row
        return row

    # ---------------------------------------------------------------- writing

    def _cue(self, phasor: np.ndarray) -> np.ndarray:
        """Content cue: the phasor bound to a shuffled copy of itself, then
        permuted and tagged.  Self-binding squares the similarity between cues,
        so an exact cue still matches fully (1 -> 1) while the faint resemblance
        every text has to every other text is suppressed (0.1 -> 0.01).  Without
        this, a loaded plate answers every cue with a blur of all its members."""
        return (phasor * phasor[self._perm2])[self._perm] * self._role

    def _key_atom(self, key: str) -> np.ndarray:
        return vsa.atom("key:" + normalize_key(key), self.dim)

    def _phasor_of(self, row: int) -> np.ndarray:
        return self.proj.phasor(self._X.a[row:row + 1])[0]

    def _last_in_session(self, realm: str, session: str) -> int | None:
        k = (realm, session)
        if k not in self._last:
            r = self._db.execute("SELECT MAX(id) AS m FROM memories WHERE realm = ? AND session = ? AND forgotten = 0",
                                 (realm, session)).fetchone()
            self._last[k] = r["m"]
        return self._last[k]

    def _plate_for(self, realm: str, incoming_load: float) -> int:
        code = self._realm_code(realm)
        i = self._open.get(code)
        if i is not None and self._pload.a[i] > 0 and self._pload.a[i] + incoming_load > self.plate_capacity:
            self._db.execute("UPDATE plates SET sealed = 1 WHERE id = ?", (int(self._pid.a[i]),))
            i = None
        if i is None:
            zeros = np.zeros(self.dim, dtype=np.complex64)
            cur = self._db.execute("INSERT INTO plates (realm, created_at, trace, gate, logged) VALUES (?, ?, ?, ?, 1)",
                                   (realm, time.time(), zeros.tobytes(), zeros.tobytes()))
            i = self._trace.add()
            self._gate.add()
            self._pid.add(cur.lastrowid)
            self._prealm.add(code)
            self._pload.add(0.0)
            self._pgain.add(1.0)
            self._pgate.add(0.0)
            self._members.append(set())
            self._open[code] = i
        return i

    def _write(self, realm: str, bindings: list[tuple]) -> None:
        """bindings: (cue, target phasor, target row, weight, cue memory id or None, cue key or None).

        Every term added to a plate is also written to the log, in the transaction the caller holds open:
        either both are kept or neither is.

        A plate's capacity is kept to, term by term.  It used to be a soft line: the room a batch needed was
        estimated from its weights before writing, as though the terms were unrelated, and the true energy
        was measured only afterwards.  Terms that agree add up to more than that estimate, so a plate could
        end above its capacity.  Now each term is tried against the plate as it stands; one that would take
        the plate past capacity closes that plate and goes on the next."""
        if not bindings:
            return
        i = self._plate_for(realm, 0.0)
        now = time.time()
        log, touched = [], []
        for cue, target, row, w, cue_memory, cue_key in bindings:
            term = np.complex64(w) * cue * target
            trace = self._trace.a[i] + term
            load = float(np.mean(np.abs(trace) ** 2))
            if load > self.plate_capacity and self._pload.a[i] > 0:      # full: this term starts the next plate
                self._db.execute("UPDATE plates SET sealed = 1 WHERE id = ?", (int(self._pid.a[i]),))
                self._open.pop(int(self._prealm.a[i]), None)
                i = self._plate_for(realm, 0.0)
                trace = self._trace.a[i] + term
                load = float(np.mean(np.abs(trace) ** 2))
            if load > self.plate_capacity * (1 + 1e-6):
                # Only a term too large for an empty plate gets here.  Stores are refused a capacity that small and
                # weights are held to MAX_WEIGHT, so this is a fault in the caller: refuse it, and the caller's
                # transaction is rolled back, sooner than keep a plate above its capacity.
                raise ValueError(f"one association of weight {float(w):g} has energy {load:.3g}, more than a whole plate "
                                 f"holds ({self.plate_capacity:g})")
            self._trace.a[i] = trace
            self._gate.a[i] += np.complex64(w) * cue
            self._pload.a[i] = load
            if i not in touched:
                touched.append(i)
            pid = int(self._pid.a[i])
            log.append((pid, cue_memory, cue_key, int(self._ids.a[row]), float(w), self._encoding, now))
            if row not in self._members[i]:
                self._members[i].add(row)
                self._db.execute("INSERT OR IGNORE INTO plate_members (plate_id, memory_id) VALUES (?, ?)",
                                 (pid, int(self._ids.a[row])))
        self._db.executemany("INSERT INTO bindings (plate_id, cue_memory, cue_key, target, weight, encoding, created_at) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?)", log)
        # Load is the plate's measured energy, not a count of bindings: many similar
        # targets under one key add up coherently and use far more of the plate's
        # capacity than the same number of unrelated associations.
        for i in touched:
            self._pgate.a[i] = float(np.mean(np.abs(self._gate.a[i]) ** 2))
            self._db.execute("UPDATE plates SET trace = ?, gate = ?, load = ? WHERE id = ?",
                             (self._trace.a[i].tobytes(), self._gate.a[i].tobytes(), float(self._pload.a[i]), int(self._pid.a[i])))

    def find_by_text(self, text: str, *, kind: str | None = None, realm: str = "waking") -> list[int]:
        sql, params = "SELECT id FROM memories WHERE forgotten = 0 AND realm = ? AND text = ?", [realm, text.strip()]
        if kind:
            sql, params = sql + " AND kind = ?", params + [kind]
        with self._lock:
            return [r["id"] for r in self._db.execute(sql, params)]

    def remember(self, text: str, *, kind: str = "episodic", realm: str = "waking", session: str = "",
                 keys: tuple[str, ...] | list[str] = (), links: tuple[int, ...] | list[int] = (),
                 salience: float = 1.0, trust: float = 0.5, meta: dict | None = None, chain: bool = True,
                 created_at: float | None = None, sentences: bool = False, whole: bool = False) -> list[int]:
        """Store text.  Long text is split into chunks that are chained together.
        `salience` is how brightly the memory is written into the plate.
        `links` are ids of existing memories this one is associated with.
        Returns the new memory ids."""
        # whole=True keeps the text as one memory however long it is (a dream must not be stored in pieces)
        chunks = [text.strip()] if whole and text.strip() else split_sentences(text) if sentences else split_text(text, self.max_chars)
        if not chunks:
            return []
        raw = self.embedder.embed(chunks, "document")         # network call: outside the lock
        with _single_threaded():
            # A memory is written with the vector that is kept for it, at the precision it is kept at.  The
            # phasor used to be made from the full-precision embedding, which the store cannot give back.
            X = self.proj.prep(raw).astype(np.float16).astype(np.float32)
            phasors = self.proj.phasor(X)
        now = time.time() if created_at is None else created_at
        all_keys = list(dict.fromkeys(normalize_key(k) for k in keys if k and k.strip()))
        w = float(min(max(salience, MIN_WEIGHT), MAX_WEIGHT))
        ids: list[int] = []
        with self._lock, _single_threaded():
            self._db.execute("BEGIN IMMEDIATE")      # take the write lock first, then check for foreign writes
            try:
                self._sync()
                for chunk, x, p in zip(chunks, X, phasors):
                    previous = self._last_in_session(realm, session)     # must be read before the insert
                    cur = self._db.execute(
                        "INSERT INTO memories (text, kind, realm, session, created_at, strength, trust, meta, vec) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (chunk, kind, realm, session, now, w, trust, json.dumps(meta or {}), x.astype(np.float16).tobytes()))
                    mid = int(cur.lastrowid)
                    row = self._add_row(mid, realm, kind, w, trust, x.astype(np.float16).astype(np.float32))
                    partners = [previous] if chain else []
                    partners += list(links) if not ids else []      # links attach to the first chunk
                    bindings = []
                    for other in dict.fromkeys(o for o in partners if o is not None and o != mid):
                        orow = self._row.get(int(other))
                        if orow is None:
                            continue
                        po = self._phasor_of(orow)
                        bindings.append((self._cue(po), p, row, w, int(other), None))     # the other one recalls this one
                        bindings.append((self._cue(p), po, orow, w, mid, None))           # and this one recalls the other
                    bindings += [(self._key_atom(k), p, row, w, None, k) for k in all_keys]
                    self._write(realm, bindings)
                    self._last[(realm, session)] = mid
                    ids.append(mid)
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                self._load()            # discard partial in-memory state
                raise
        return ids

    # ---------------------------------------------------------------- reading

    def _probe(self, cue: np.ndarray, realm_codes: list[int], *, aperture: float = 1.0,
               max_plates: int = 24, z_min: float = 3.5, ridge: float = 0.1,
               leave_out: set | None = None, info: dict | None = None) -> dict[int, float]:
        """Unbind `cue` from every resonating plate; return {row: recovered strength}.

        `leave_out` and `info` are for measuring, not for recall: rows to keep out of the attribution among a
        plate's members, and a dict that is told how many plates resonated and how many the limit turned away."""
        n = self._trace.n
        if n == 0:
            return {}
        d = self.dim if aperture >= 1.0 else max(64, int(self.dim * aperture))
        cc = np.conj(cue[:d])
        cue_weight2 = float(np.mean(np.abs(cc) ** 2))        # the cue's actual power
        load = self._pload.v
        resonance = (self._gate.a[:n, :d] @ cc).real / d
        gate_sigma = np.sqrt(np.maximum(self._pgate.v, 1e-9) * cue_weight2 / (2.0 * d))
        eligible = np.isin(self._prealm.v, realm_codes) & (load > 0)
        if eligible.sum() >= 4:
            # Every plate resonates a little with every cue, in proportion to its
            # load.  Subtract that shared baseline so only real matches stand out.
            resonance = resonance - np.median(resonance[eligible] / load[eligible]) * load
        ok = eligible & (resonance > 2.5 * gate_sigma)
        idx = np.nonzero(ok)[0]
        if info is not None:
            info["resonated"] = int(idx.size)
            info["turned_away"] = max(0, int(idx.size) - int(max_plates))
        if idx.size == 0:
            return {}
        if idx.size > max_plates:
            idx = idx[np.argsort(-resonance[idx])[:max_plates]]
        estimates = self.proj.back(self._trace.a[idx, :d] * cc, aperture_dim=d)
        out: dict[int, float] = {}
        for j, i in enumerate(idx):
            rows = np.fromiter(self._members[i], dtype=np.int64, count=len(self._members[i]))
            if rows.size == 0:
                continue
            rows = rows[np.isin(self._realm.a[rows], realm_codes)]
            if leave_out:
                rows = rows[~np.isin(rows, list(leave_out))]
                if rows.size == 0:
                    continue
            M = self._X.a[rows]
            # Ridge deconvolution: members of a plate resemble each other, so a
            # plain matched filter credits every member for its neighbours'
            # signal.  Solving (M M^T + lambda I) a = M e instead attributes the
            # estimate to the members that actually explain it.
            G = (M @ M.T).astype(np.float64)
            K = np.linalg.inv(G + ridge * np.eye(rows.size))
            scores = (1.0 + ridge) * (K @ (M @ estimates[j]).astype(np.float64))
            sigma = (1.0 + ridge) * vsa.noise_sigma(float(load[i]), cue_weight2, d) \
                * np.sqrt(np.maximum(np.einsum("ij,jk,ki->i", K, G, K), 1e-12))
            keep = scores > z_min * sigma
            gain = float(self._pgain.a[i])
            for row, sc in zip(rows[keep], scores[keep]):
                out[int(row)] = max(out.get(int(row), 0.0), float(sc) * gain)
                if info is not None:                     # which plate each memory was read from
                    info.setdefault("plates", {}).setdefault(int(row), set()).add(int(self._pid.a[i]))
        return out

    def _lexical(self, query: str, mask: np.ndarray) -> dict[int, float]:
        """{row: 0..1} for memories sharing content words with the query, best match = 1.
        Only memories that can be returned count, so an excluded one cannot be the yardstick."""
        terms = content_terms(query) if self._fts else []
        if not terms:
            return {}
        match = " OR ".join(f'"{t}"' for t in terms)
        try:
            found = self._db.execute("SELECT rowid, bm25(memories_fts) AS b FROM memories_fts WHERE memories_fts MATCH ? "
                                     "ORDER BY b LIMIT 40", (match,)).fetchall()
        except sqlite3.OperationalError:
            return {}
        scores = {self._row[r["rowid"]]: r["b"] for r in found if r["rowid"] in self._row and mask[self._row[r["rowid"]]]}
        if not scores:
            return {}
        best = min(scores.values()) or -1e-9
        return {row: max(0.0, min(1.0, b / best)) for row, b in scores.items()}

    def _alive_mask(self, realm_codes: list[int]) -> np.ndarray:
        return np.isin(self._realm.v, realm_codes) & (self._trust.v >= 0)   # trust < 0 marks forgotten rows

    def recall(self, query: str | None = None, k: int = 8, *, realms: tuple[str, ...] = ("waking",),
               keys: tuple[str, ...] | list[str] = (), vector: np.ndarray | None = None, hops: int = 3,
               hop_min: float = 0.3, hop_ratio: float = 0.75, gamma: float = 0.85, min_score: float = 0.2,
               aperture: float = 1.0, blur: float = 0.0, fuzzy: bool = False, reinforce: bool = False,
               exclude: tuple[int, ...] | list[int] = (), rng: np.random.Generator | None = None,
               dual: bool = False, skip_kinds: tuple[str, ...] | list[str] = (), lexical: float = 0.0,
               kind_weights: dict[str, float] | None = None,
               only_kinds: tuple[str, ...] | list[str] = (), min_trust: float = 0.0,
               min_strength: float = 0.0, reach: int = 1, info: dict | None = None) -> list[Recollection]:
        """Recall memories for a text query, a prepared vector, discrete keys, or any mix.

        The best direct matches ("hops") are each used as an exact cue on the
        plates.  A memory recovered that way gets `assoc` = how strongly it is
        tied to the hop, scaled by how well the hop matched relative to the best
        match.  An associate never outranks the memory that led to it.

        aperture < 1 reads only that fraction of each plate (graceful degradation).
        blur > 0 adds phase noise to the cues, loosening the associations.
        fuzzy=True also cues the plates with the query itself, which returns a
        looser blend of things linked to anything resembling it (used for dreaming).
        dual=True embeds a text query twice, as a question and as a statement, and
        takes the better match: "what is my name" resembles "my name is Kayla" far
        more as a statement than as a search query.
        skip_kinds leaves memories of those kinds out entirely.
        lexical adds up to that much to the similarity of memories sharing content
        words with a text query.  kind_weights scales the final score by kind.
        min_strength leaves faded memories out (everyday recall); 0 includes them (deep recall).
        reach is how many links to follow outward from the direct matches: with 2, what a
        linked memory is itself linked to can surface as well.
        """
        x = x_alt = None
        if vector is not None:
            x = np.asarray(vector, dtype=np.float32)
        elif query:
            raw = self.embedder.embed([query], "query")       # network call: outside the lock
            x = self.proj.prep(raw)[0]
            if dual:
                x_alt = self.proj.prep(self.embedder.embed([query], "document"))[0]
        with self._lock, _single_threaded():
            self._sync()
            if self._X.n == 0:
                return []
            codes = [self._realm_codes[r] for r in realms if r in self._realm_codes]
            if not codes:
                return []
            mask = self._alive_mask(codes)
            for mid in exclude:
                if mid in self._row:
                    mask[self._row[mid]] = False
            skip_codes = [self._kind_codes[kind] for kind in skip_kinds if kind in self._kind_codes]
            if skip_codes:
                mask &= ~np.isin(self._kind.v, skip_codes)
            if only_kinds:
                mask &= np.isin(self._kind.v, [self._kind_codes[kind] for kind in only_kinds if kind in self._kind_codes])
            if min_trust > 0:
                mask &= self._trust.v >= min_trust       # superseded and repeatedly-wrong memories stay out
            if min_strength > 0:
                mask &= self._strength.v >= min_strength  # faded memories stay out of everyday recall
            self._last_parts, self._last_lex, self._last_lex_weight = None, {}, lexical

            def dither(cue: np.ndarray) -> np.ndarray:
                if blur <= 0:
                    return cue
                gen = rng or np.random.default_rng()
                return cue * np.exp(1j * gen.normal(0.0, blur, self.dim)).astype(np.complex64)

            direct = np.zeros(self._X.n, dtype=np.float32)
            assoc: dict[int, float] = {}
            anchor = 1.0
            if x is not None:
                direct = self._X.v @ x
                if x_alt is not None:
                    alt = self._X.v @ x_alt
                    self._last_parts = (direct.copy(), alt)
                    direct = np.maximum(direct, alt)
                if lexical > 0 and query:
                    self._last_lex = self._lexical(query, mask)
                    for row, share in self._last_lex.items():
                        direct[row] += lexical * share
                direct[~mask] = -1.0
                order = [int(r) for r in np.argsort(-direct)[:max(hops, 0)]]
                d_top = float(direct[order[0]]) if order else 0.0
                anchor = max(d_top, 0.0)
                for r in order:
                    if direct[r] < max(hop_min, hop_ratio * d_top):
                        continue
                    relevance = float(direct[r]) / d_top
                    if info is not None:                 # for measuring: which memories were used as cues
                        info.setdefault("hops", []).append(int(self._ids.a[r]))
                    for row, sc in self._probe(dither(self._cue(self._phasor_of(r))), codes, aperture=aperture, info=info).items():
                        if row != r:
                            assoc[row] = max(assoc.get(row, 0.0), min(1.0, sc) * relevance)
                # Follow links outward from what was just found.  Each further step is weaker,
                # and only memories that could be returned are used as stepping stones.
                frontier = dict(assoc)
                for _ in range(max(reach, 1) - 1):
                    found: dict[int, float] = {}
                    for r, weight in sorted(frontier.items(), key=lambda kv: -kv[1])[:4]:
                        if not mask[r] or weight < 0.3:
                            continue
                        for row, sc in self._probe(self._cue(self._phasor_of(r)), codes, aperture=aperture).items():
                            value = min(1.0, sc) * weight * 0.7
                            if row != r and row not in order and value > assoc.get(row, 0.0):
                                found[row] = max(found.get(row, 0.0), value)
                    assoc.update(found)
                    frontier = found
                if fuzzy:
                    cue = dither(self._cue(self.proj.phasor(x[None, :])[0]))
                    for row, sc in self._probe(cue, codes, aperture=aperture).items():
                        assoc[row] = max(assoc.get(row, 0.0), min(1.0, sc))
            norm_keys = [normalize_key(key) for key in keys if key and key.strip()]
            if norm_keys:
                anchor = max(anchor, 0.5)
                kcue = np.sum([self._key_atom(key) for key in norm_keys], axis=0)
                for row, sc in self._probe(kcue, codes, aperture=aperture).items():
                    assoc[row] = max(assoc.get(row, 0.0), min(1.0, sc / len(norm_keys)))
            return self._rank(direct, assoc, mask, anchor, gamma, k, min_score, reinforce, kind_weights)

    def _rank(self, direct: np.ndarray, assoc: dict[int, float], mask: np.ndarray, anchor: float, gamma: float,
              k: int, min_score: float, reinforce: bool, kind_weights: dict[str, float] | None = None) -> list[Recollection]:
        parts = getattr(self, "_last_parts", None)
        lex, lex_weight = getattr(self, "_last_lex", {}), getattr(self, "_last_lex_weight", 0.0)
        d = np.maximum(direct, 0.0)
        via = np.zeros_like(d)
        for row, sc in assoc.items():
            via[row] = gamma * anchor * sc
        base = np.maximum(d, via) + 0.15 * np.minimum(d, via)
        base[~mask] = 0.0
        final = base * (0.7 + 0.3 * np.tanh(self._strength.v)) * (0.75 + 0.5 * np.clip(self._trust.v, 0, 1))
        for kind, weight in (kind_weights or {}).items():
            if kind in self._kind_codes:
                final[self._kind.v == self._kind_codes[kind]] *= weight
        # A strongly linked memory passes whenever the match that led to it would pass.
        strong = np.zeros(len(d), dtype=bool)
        if anchor >= min_score:
            for row, sc in assoc.items():
                strong[row] = sc >= 0.6
        top = [int(r) for r in np.argsort(-final)[:k] if mask[r] and (base[r] >= min_score or strong[r])]
        if not top:
            return []
        ids = [int(self._ids.a[r]) for r in top]
        rows = {r["id"]: r for r in self._db.execute(
            f"SELECT id, text, kind, realm, session, created_at, meta FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids)}
        out = [Recollection(id=mid, text=rows[mid]["text"], kind=rows[mid]["kind"], realm=rows[mid]["realm"],
                            score=float(final[r]), direct=float(d[r]), assoc=float(assoc.get(r, 0.0)),
                            strength=float(self._strength.a[r]), trust=float(self._trust.a[r]),
                            created_at=rows[mid]["created_at"], session=rows[mid]["session"],
                            meta=json.loads(rows[mid]["meta"] or "{}"),
                            lexical=float(lex_weight * lex.get(r, 0.0)),
                            as_query=float(parts[0][r]) if parts else float(d[r] - lex_weight * lex.get(r, 0.0)),
                            as_statement=float(parts[1][r]) if parts else 0.0)
               for r, mid in zip(top, ids) if mid in rows]
        if reinforce:
            self.reinforce(ids, 0.05)
        return out

    def associates(self, memory_id: int, k: int = 8, *, realms: tuple[str, ...] = ("waking",),
                   aperture: float = 1.0, min_score: float = 0.2) -> list[Recollection]:
        """What does this memory lead to?  Cue the plates with the memory itself."""
        with self._lock, _single_threaded():
            self._sync()
            row = self._row.get(memory_id)
            codes = [self._realm_codes[r] for r in realms if r in self._realm_codes]
            if row is None or not codes:
                return []
            mask = self._alive_mask(codes)
            mask[row] = False
            self._last_parts, self._last_lex, self._last_lex_weight = None, {}, 0.0
            found = self._probe(self._cue(self._phasor_of(row)), codes, aperture=aperture)
            assoc = {r: min(1.0, sc) for r, sc in found.items() if r != row}
            return self._rank(np.zeros(self._X.n, dtype=np.float32), assoc, mask, 1.0, 1.0, k, min_score, False)

    # ------------------------------------------------- reflection support

    def kv_get(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            value = self._meta_get("kv:" + key)
        return default if value is None else (value.decode() if isinstance(value, bytes) else str(value))

    def kv_set(self, key: str, value: str) -> None:
        with self._lock:
            self._meta_set("kv:" + key, str(value))

    def memories_after(self, after_id: int, limit: int = 60, *, realm: str = "waking",
                       exclude_kinds: tuple[str, ...] | list[str] = ()) -> list[dict]:
        """Memories with id greater than `after_id`, oldest first."""
        sql = "SELECT id, text, kind, session, created_at FROM memories WHERE forgotten = 0 AND realm = ? AND id > ?"
        params: list = [realm, after_id]
        if exclude_kinds:
            sql += f" AND kind NOT IN ({','.join('?' * len(exclude_kinds))})"
            params += list(exclude_kinds)
        with self._lock:
            return [dict(r) for r in self._db.execute(sql + " ORDER BY id LIMIT ?", params + [limit])]

    def first_id_since(self, when: float, *, realm: str = "waking") -> int | None:
        """The id of the oldest memory made at or after a time; None if there is none."""
        with self._lock:
            r = self._db.execute("SELECT MIN(id) AS m FROM memories WHERE forgotten = 0 AND realm = ? AND created_at >= ?",
                                 (realm, float(when))).fetchone()
        return int(r["m"]) if r and r["m"] is not None else None

    def user_wrote(self, word: str) -> bool:
        """Whether the user has ever written this word themselves (whatever its capitals)."""
        word = str(word or "").strip()
        if not word:
            return False
        pattern = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", re.IGNORECASE)
        with self._lock:
            rows = self._db.execute("SELECT text FROM memories WHERE forgotten = 0 AND kind LIKE '%!_user' ESCAPE '!' "
                                    "AND text LIKE ? LIMIT 200", (f"%{word}%",)).fetchall()
        return any(pattern.search(r["text"] or "") for r in rows)

    def count_after(self, after_id: int, *, realm: str = "waking", exclude_kinds: tuple[str, ...] | list[str] = ()) -> int:
        sql = "SELECT COUNT(*) AS n FROM memories WHERE forgotten = 0 AND realm = ? AND id > ?"
        params: list = [realm, after_id]
        if exclude_kinds:
            sql += f" AND kind NOT IN ({','.join('?' * len(exclude_kinds))})"
            params += list(exclude_kinds)
        with self._lock:
            return int(self._db.execute(sql, params).fetchone()["n"])

    def related_of_kind(self, memory_ids: list[int], kind: str, *, per_memory: int = 3, min_similarity: float = 0.2,
                        limit: int = 20, min_trust: float = 0.15) -> list[dict]:
        """Existing memories of `kind` that resemble any of the given memories, using the
        vectors already stored (no embedding calls).  Used to show reflection the facts a
        new statement might contradict."""
        with self._lock, _single_threaded():
            self._sync()
            code = self._kind_codes.get(kind)
            rows = [self._row[m] for m in memory_ids if m in self._row]
            if code is None or not rows:
                return []
            pool = np.nonzero((self._kind.v == code) & (self._trust.v >= min_trust))[0]
            if pool.size == 0:
                return []
            sims = self._X.a[rows] @ self._X.a[pool].T
            best: dict[int, float] = {}
            for i in range(len(rows)):
                for j in np.argsort(-sims[i])[:per_memory]:
                    if sims[i, j] >= min_similarity:
                        best[int(pool[j])] = max(best.get(int(pool[j]), 0.0), float(sims[i, j]))
            ids = [int(self._ids.a[r]) for r, _ in sorted(best.items(), key=lambda kv: -kv[1])[:limit]]
            if not ids:
                return []
            found = {r["id"]: dict(r) for r in self._db.execute(
                f"SELECT id, text, kind FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids)}
            return [found[i] for i in ids if i in found]

    def near_pairs(self, kinds: tuple[str, ...] | list[str], *, min_similarity: float = 0.8, min_trust: float = 0.15,
                   limit: int = 2000) -> list[tuple[int, int, float]]:
        """Pairs of memories of these kinds that are close to one another, closest first, from the vectors already
        stored.  (older id, newer id, similarity).  Retired memories are left out."""
        with self._lock, _single_threaded():
            self._sync()
            codes = [self._kind_codes[k] for k in kinds if k in self._kind_codes]
            if not codes:
                return []
            pool = np.nonzero(np.isin(self._kind.v, codes) & (self._trust.v >= min_trust))[0]
            if pool.size < 2:
                return []
            X = self._X.a[pool]
            out: list[tuple[int, int, float]] = []
            for start in range(0, pool.size, 512):                     # in blocks: a large store stays within memory
                sims = X[start:start + 512] @ X.T
                for i, j in zip(*np.nonzero(sims >= min_similarity)):
                    a, b = start + int(i), int(j)
                    if a < b:
                        ia, ib = int(self._ids.a[pool[a]]), int(self._ids.a[pool[b]])
                        out.append((min(ia, ib), max(ia, ib), float(sims[i, j])))
            out.sort(key=lambda t: -t[2])
            return out[:limit]

    def unsupersede(self, memory_id: int, trust: float = 0.6) -> bool:
        """Bring a retired memory back into recall."""
        with self._lock:
            row = self._row.get(memory_id)
            current = self._db.execute("SELECT meta FROM memories WHERE id = ? AND forgotten = 0", (memory_id,)).fetchone()
            if row is None or current is None:
                return False
            meta = json.loads(current["meta"] or "{}")
            if "superseded_at" not in meta:
                return False
            for key in ("superseded_by", "superseded_at", "superseded_reason"):
                meta.pop(key, None)
            self._trust.a[row] = trust
            self._db.execute("UPDATE memories SET trust = ?, meta = ? WHERE id = ?", (trust, json.dumps(meta), memory_id))
            return True

    def supersede(self, old_id: int, new_id: int | None = None, reason: str = "") -> bool:
        """Retire a memory without deleting it: it leaves recall but stays on record,
        so the agent can still know that something used to be true."""
        with self._lock:
            row = self._row.get(old_id)
            if row is None:
                return False
            current = self._db.execute("SELECT meta FROM memories WHERE id = ?", (old_id,)).fetchone()
            meta = json.loads(current["meta"] or "{}")
            meta.update({"superseded_by": new_id, "superseded_at": time.time(), "superseded_reason": reason[:300]})
            self._trust.a[row] = 0.0
            self._db.execute("UPDATE memories SET trust = 0, meta = ? WHERE id = ?", (json.dumps(meta), old_id))
            return True

    def profile(self, who: str) -> str:
        with self._lock:
            row = self._db.execute("SELECT text FROM profiles WHERE who = ? ORDER BY id DESC LIMIT 1", (who,)).fetchone()
        return row["text"] if row else ""

    def set_profile(self, who: str, text: str) -> None:
        """Profiles are append-only so that drift can be inspected and undone."""
        with self._lock:
            self._db.execute("INSERT INTO profiles (who, text, created_at) VALUES (?, ?, ?)", (who, text.strip(), time.time()))

    def profile_history(self, who: str, limit: int = 10) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(
                "SELECT id, text, created_at FROM profiles WHERE who = ? ORDER BY id DESC LIMIT ?", (who, limit))]

    def get(self, memory_id: int) -> dict | None:
        r = self._db.execute("SELECT id, text, kind, realm, session, created_at, strength, trust, recalls, meta "
                             "FROM memories WHERE id = ? AND forgotten = 0", (memory_id,)).fetchone()
        return dict(r, meta=json.loads(r["meta"] or "{}")) if r else None

    def recent(self, n: int = 20, *, realm: str = "waking", kind: str | None = None, since: float | None = None) -> list[dict]:
        sql, params = "SELECT id, text, kind, realm, session, created_at, strength, trust FROM memories WHERE forgotten = 0 AND realm = ?", [realm]
        if kind:
            sql, params = sql + " AND kind = ?", params + [kind]
        if since is not None:
            sql, params = sql + " AND created_at >= ?", params + [since]
        return [dict(r) for r in self._db.execute(sql + " ORDER BY id DESC LIMIT ?", params + [n])]

    # ------------------------------------------------------------ maintenance

    @_locked
    def reinforce(self, ids: list[int], amount: float = 0.1) -> None:
        with self._lock:
            for mid in ids:
                row = self._row.get(mid)
                if row is not None:
                    self._strength.a[row] = min(3.0, self._strength.a[row] + amount)
                    self._db.execute("UPDATE memories SET strength = ?, recalls = recalls + 1 WHERE id = ?",
                                     (float(self._strength.a[row]), mid))

    @_locked
    def set_kind(self, memory_id: int, kind: str) -> bool:
        row = self._row.get(memory_id)
        if row is None:
            return False
        self._kind.a[row] = self._kind_codes.setdefault(kind, len(self._kind_codes))
        self._db.execute("UPDATE memories SET kind = ? WHERE id = ?", (kind, memory_id))
        return True

    @_locked
    def similarity_to_realm(self, memory_id: int, realm: str, kinds: tuple[str, ...] | list[str] = ()) -> float:
        """How closely this memory resembles the nearest memory in another realm (0 if that realm is empty)."""
        row, code = self._row.get(memory_id), self._realm_codes.get(realm)
        if row is None or code is None:
            return 0.0
        pool = (self._realm.v == code) & (self._trust.v >= 0)
        if kinds:
            pool &= np.isin(self._kind.v, [self._kind_codes[k] for k in kinds if k in self._kind_codes])
        if not pool.any():
            return 0.0
        return float(np.max(self._X.v[pool] @ self._X.a[row]))

    @_locked
    def fade(self, ids: list[int], factor: float, *, floor: float = 0.1) -> int:
        """Lower the strength of these memories.  Nothing is deleted: a faded memory drops
        out of everyday recall and stays reachable by deep recall, which strengthens it again."""
        changed = 0
        for mid in ids:
            row = self._row.get(mid)
            if row is None:
                continue
            new = max(floor, float(self._strength.a[row]) * factor)
            if new < self._strength.a[row]:
                self._strength.a[row] = new
                self._db.execute("UPDATE memories SET strength = ? WHERE id = ?", (new, mid))
                changed += 1
        return changed

    @_locked
    def sessions(self, kinds: tuple[str, ...] | list[str], *, realm: str = "waking") -> list[dict]:
        """One row per conversation: its id, first and last memory, size and time span."""
        marks = ",".join("?" * len(kinds))
        return [dict(r) for r in self._db.execute(
            f"SELECT session, MIN(id) AS first_id, MAX(id) AS last_id, COUNT(*) AS n, MIN(created_at) AS started, "
            f"MAX(created_at) AS ended FROM memories WHERE forgotten = 0 AND realm = ? AND kind IN ({marks}) "
            f"GROUP BY session ORDER BY MIN(id)", [realm] + list(kinds))]

    @_locked
    def session_memories(self, session: str, *, after_id: int = 0, kinds: tuple[str, ...] | list[str] = (),
                         realm: str = "waking", limit: int = 200) -> list[dict]:
        sql = ("SELECT id, text, kind, session, created_at, strength FROM memories WHERE forgotten = 0 AND realm = ? "
               "AND session = ? AND id > ?")
        params: list = [realm, session, after_id]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params += list(kinds)
        return [dict(r) for r in self._db.execute(sql + " ORDER BY id LIMIT ?", params + [limit])]

    @_locked
    def echoes(self, memory_id: int, *, older_than: float, low: float = 0.25, high: float = 0.7, k: int = 3,
               kinds: tuple[str, ...] | list[str] = (), realm: str = "waking",
               skip_kinds: tuple[str, ...] | list[str] = (), skip_ids=()) -> list[dict]:
        """Older memories that resemble this one somewhat: related, but not the same thing
        said again.  Faded memories are included; this is how a dream reaches back.  Works
        from the stored vectors, so it costs no embedding calls and strengthens nothing.

        Age, kind and `skip_ids` (memories the caller already has) are decided before the best `k` are taken.  They used to be decided after: the likeliest
        few candidates were taken first, and if those were all recent, or all of a kind the caller would not
        use, nothing came back although older memories that fitted were a little further down."""
        row = self._row.get(memory_id)
        code = self._realm_codes.get(realm)
        if row is None or code is None:
            return []
        sims = self._X.v @ self._X.a[row]
        ok = (self._realm.v == code) & (self._trust.v >= 0.15) & (sims >= low) & (sims <= high)
        if kinds:
            ok &= np.isin(self._kind.v, [self._kind_codes[x] for x in kinds if x in self._kind_codes])
        if skip_kinds:
            ok &= ~np.isin(self._kind.v, [self._kind_codes[x] for x in skip_kinds if x in self._kind_codes])
        ok[row] = False
        for mid in skip_ids:
            at = self._row.get(mid)
            if at is not None:
                ok[at] = False
        if not ok.any():
            return []
        old = np.zeros(self._X.n, dtype=bool)
        with self._lock:
            for r in self._db.execute("SELECT id FROM memories WHERE forgotten = 0 AND realm = ? AND created_at < ?", (realm, older_than)):
                at = self._row.get(r["id"])
                if at is not None:
                    old[at] = True
        ok &= old
        candidates = [int(r) for r in np.argsort(-sims) if ok[r]][:k]
        if not candidates:
            return []
        ids = [int(self._ids.a[r]) for r in candidates]
        with self._lock:
            rows = {r["id"]: dict(r) for r in self._db.execute(
                f"SELECT id, text, kind, session, created_at, strength FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids)}
        return [dict(rows[mid], similarity=float(sims[r])) for r, mid in zip(candidates, ids) if mid in rows]

    def linked(self, memory_id: int, *, older_than: float | None = None, k: int = 2, realm: str = "waking",
               skip_kinds: tuple[str, ...] | list[str] = (), skip_ids=(), min_chars: int = 0) -> list[dict]:
        """What the plates tie to this memory, strongest first: read off the plates with the memory as the cue.
        With `older_than`, only memories from before then.  Faded memories are included and nothing is
        strengthened: this is for dreaming, which reaches back without disturbing what it finds.

        Everything the caller would turn down (`skip_kinds`, `skip_ids`, text shorter than `min_chars`, too
        recent) is left out before the strongest `k` are taken, so what is turned down never takes the place
        of something usable further down."""
        with self._lock, _single_threaded():
            self._sync()
            row = self._row.get(memory_id)
            code = self._realm_codes.get(realm)
            if row is None or code is None:
                return []
            found = self._probe(self._cue(self._phasor_of(row)), [code])
            skip = {self._kind_codes[x] for x in skip_kinds if x in self._kind_codes}
            have = set(skip_ids)
            rows = [(r, sc) for r, sc in found.items()
                    if r != row and self._trust.a[r] >= 0.15 and int(self._kind.a[r]) not in skip
                    and int(self._ids.a[r]) not in have]
            if not rows:
                return []
            ids = [int(self._ids.a[r]) for r, _ in rows]
            sql = f"SELECT id, text, kind, session, created_at, strength FROM memories WHERE forgotten = 0 AND id IN ({','.join('?' * len(ids))})"
            params: list = list(ids)
            if older_than is not None:
                sql, params = sql + " AND created_at < ?", params + [older_than]
            kept = {r["id"]: dict(r) for r in self._db.execute(sql, params)}
            out = [dict(kept[int(self._ids.a[r])], link=float(min(1.0, sc))) for r, sc in sorted(rows, key=lambda t: -t[1])
                   if int(self._ids.a[r]) in kept and len(kept[int(self._ids.a[r])]["text"]) >= min_chars]
            return out[:k]

    @_locked
    def set_trust(self, memory_id: int, trust: float) -> bool:
        with self._lock:
            row = self._row.get(memory_id)
            if row is None:
                return False
            self._trust.a[row] = min(max(trust, 0.0), 1.0)
            self._db.execute("UPDATE memories SET trust = ? WHERE id = ?", (float(self._trust.a[row]), memory_id))
            return True

    @_locked
    def decay(self, factor: float = 0.98, *, floor: float = 0.1, exempt_kinds: tuple[str, ...] | list[str] = ()) -> None:
        """Let memories fade a little, except the exempt kinds.  Reinforced memories fade
        from a higher start.  Nothing calls this yet; when fading is switched on, what the
        agent knows about the user is exempt: those facts change by contradiction, not by time."""
        with self._lock:
            keep = np.isin(self._kind.v, [self._kind_codes[k] for k in exempt_kinds if k in self._kind_codes])
            self._strength.v[~keep] = np.maximum(self._strength.v[~keep] * factor, floor)
            sql, params = "UPDATE memories SET strength = MAX(strength * ?, ?) WHERE forgotten = 0", [factor, floor]
            if exempt_kinds:
                sql += f" AND kind NOT IN ({','.join('?' * len(exempt_kinds))})"
                params += list(exempt_kinds)
            self._db.execute(sql, params)

    @_locked
    def forget(self, memory_id: int) -> bool:
        """Remove a memory from cleanup.  Its traces stay in the plates as faint
        noise that can no longer be resolved into anything."""
        with self._lock:
            row = self._row.pop(memory_id, None)
            if row is None:
                return False
            self._trust.a[row] = -1.0
            self._X.a[row] = 0.0
            for members in self._members:
                members.discard(row)
            self._db.execute("UPDATE memories SET forgotten = 1, text = '', vec = NULL WHERE id = ?", (memory_id,))
            # The record of which plates carried it is kept, marked: its traces are still there, and counted.
            self._db.execute("UPDATE plate_members SET gone = 1 WHERE memory_id = ?", (memory_id,))
            self._last.clear()
            return True

    @_locked
    def stats(self) -> dict:
        with self._lock:
            n_plates = self._trace.n
            by_realm = {realm: int(np.sum((self._realm.v == code) & (self._trust.v >= 0)))
                        for realm, code in self._realm_codes.items()}
            return {
                "memories": len(self._row),
                "by_realm": by_realm,
                "plates": n_plates,
                "dim": self.dim,
                "embed_dim": self.embed_dim,
                "plate_capacity": self.plate_capacity,
                "mean_plate_load": float(self._pload.v.mean()) if n_plates else 0.0,
                "plate_bytes": int(n_plates * self.dim * 16),
                "cleanup_bytes": int(len(self._row) * self.embed_dim * 2),
                "embedder": self.embedder.signature,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                self._db.close()

    def __enter__(self) -> "HolonomicMemory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
