"""Hermes Agent memory provider backed by the holonomic engine.

What it does each turn:
  prefetch()   recalls memories for the incoming message and returns them as context
  sync_turn()  stores what was just said (runs on Hermes' background worker)

Config lives in $HERMES_HOME/holonomic.json; data in $HERMES_HOME/holonomic/.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt, spawn_context_thread

from .reflect import REFLECT_DEFAULTS, IdleReflector

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "ollama_host": "http://localhost:11434",
    "embed_model": "nomic-embed-text",
    "embed_timeout": 10.0,        # seconds; a hung embedding call must not stall a turn
    "recall_k": 6,                # memories injected per turn
    "min_score": 0.2,             # floor: below this a memory is never injected
    "score_band": 0.25,           # and it must also score within this much of the best match
    "lexical_weight": 0.2,        # how much sharing content words with the message adds to similarity
    "assistant_weight": 0.75,     # the agent's own past statements rank below the user's
    "max_context_chars": 2400,    # budget for the injected block
    "max_item_chars": 420,        # each recalled memory is trimmed to this
    "dual_query": False,          # also match the message as a statement (a second embedding call; rarely helps)
    "store_assistant": True,      # also remember the agent's own replies
    "max_turn_chars": 4000,       # longer messages are cut before storing
    "dim": 4096,                  # plate dimension (fixed once the store exists)
    "plate_capacity": 128.0,      # plate energy at which a plate is sealed (fixed once the store exists)
}

DEFAULTS.update(REFLECT_DEFAULTS)

TOOL_SCHEMA = {
    "name": "holonomic_memory",
    "description": (
        "Long-term associative memory. Relevant memories are recalled automatically every turn; use this tool "
        "to dig deeper or to manage memories.\n"
        "ACTIONS:\n"
        "- recall: search memory for `query`. Returns memories similar to it and memories linked to those.\n"
        "- remember: store `content` as a durable fact, preference or note. Use for things worth keeping that "
        "might not be obvious from the conversation alone. Optional `about` (names/topics) and `importance` 1-3.\n"
        "- related: what is linked to a memory (`memory_id`) or filed under a name/topic (`entity`).\n"
        "- feedback: rate `memory_id` as `helpful` or `wrong`. Wrong memories sink, helpful ones rise.\n"
        "- forget: permanently remove `memory_id`. Only when asked to, or when a memory is clearly false.\n"
        "- stats: size of the memory store."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["recall", "remember", "related", "feedback", "forget", "stats"]},
            "query": {"type": "string", "description": "What to search for (recall)."},
            "content": {"type": "string", "description": "What to store (remember)."},
            "about": {"type": "array", "items": {"type": "string"}, "description": "Names or topics this concerns (remember)."},
            "importance": {"type": "integer", "minimum": 1, "maximum": 3, "description": "1 minor, 2 normal, 3 vital (remember)."},
            "memory_id": {"type": "integer", "description": "A memory id as shown in brackets, e.g. [#42]."},
            "entity": {"type": "string", "description": "A name or topic (related)."},
            "rating": {"type": "string", "enum": ["helpful", "wrong"], "description": "For feedback."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Max results (default 8)."},
        },
        "required": ["action"],
    },
}

_KIND_LABEL = {"fact": "learned about the user", "self_note": "your own note", "insight": "insight",
               "said_user": "user said", "asked_user": "user asked", "said_assistant": "you said", "note": "noted", "core": "core memory",
               "reflection": "reflection", "dream": "dream"}

_STOP = {"I", "I'm", "I've", "I'll", "I'd", "The", "A", "An", "It", "It's", "This", "That", "These", "Those", "We",
         "You", "He", "She", "They", "My", "Your", "Our", "And", "But", "Or", "So", "If", "When", "What", "Why", "How",
         "Who", "Where", "Yes", "No", "Ok", "Okay", "Thanks", "Please", "Also", "There", "Here", "Is", "Are", "Do",
         "Does", "Can", "Could", "Would", "Should", "Will", "Let", "Let's", "Not", "Now", "Then", "Just", "Maybe",
         "Yesterday", "Today", "Tomorrow", "After", "Before", "Once", "Since", "While", "Because", "However",
         "Although", "Still", "Well", "Oh", "Hey", "Hi", "Hello", "For", "In", "On", "At", "To", "From", "With", "Of",
         "By", "As", "One", "Two", "Some", "Any", "All", "Most", "Many", "Every", "Each", "Both", "Its", "Their",
         "His", "Her", "Got", "Noted", "Understood", "Sure", "Great", "Right", "Sorry", "Here's", "That's", "There's"}
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9'\-]+(?:\s+[A-Z][A-Za-z0-9'\-]+){0,2}\b")
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_CONTEXT_RE = re.compile(r"<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>", re.IGNORECASE)


def extract_keys(text: str, limit: int = 4) -> List[str]:
    """Names and topics worth filing a memory under: capitalised terms that are
    not just the first word of a sentence, plus `backticked` terms."""
    found: Dict[str, int] = {}
    for m in _ENTITY_RE.finditer(text):
        term = m.group(0)
        words = term.split()
        at_sentence_start = m.start() == 0 or bool(re.search(r"[.!?:\n]\s*$", text[max(0, m.start() - 3):m.start()]))
        if at_sentence_start and len(words) == 1:
            continue
        while words and words[0] in _STOP:
            words = words[1:]
        if not words or (len(words) == 1 and (words[0] in _STOP or len(words[0]) < 3)):
            continue
        key = " ".join(words)
        found[key] = found.get(key, 0) + 1
    for m in re.finditer(r"`([^`\n]{2,40})`", text):
        found[m.group(1)] = found.get(m.group(1), 0) + 1
    # Fold a shorter term into a longer one that contains it ("Mara" into "Mara Oliveira").
    for short in sorted(found, key=len):
        longer = next((k for k in found if k != short and re.search(rf"\b{re.escape(short)}\b", k)), None)
        if longer:
            found[longer] += found.pop(short)
    return [k for k, _ in sorted(found.items(), key=lambda kv: -kv[1])[:limit]]


def clean_for_storage(text: str, max_chars: int) -> str:
    text = _CONTEXT_RE.sub("", text or "")
    text = _CODE_BLOCK_RE.sub("[code omitted]", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text[:max_chars]


# Questions are stored (they keep the conversation chain intact) but never recalled:
# a past question tells the agent nothing, and it is always the best match for the
# same question asked again, crowding out the memory that answers it.
QUESTION_KIND = "asked_user"

# Replies in which the model says it has no memory or does not know something.  Storing
# these teaches the agent, through its own recall, that it cannot remember: "I don't know
# your name" is the best possible match for "what is my name?" the next time it is asked.
_DENIAL_RE = re.compile(
    r"\bI(?: am|'m)? (?:do not|don't|cannot|can't|unable to|not able to) (?:\w+ly )?(?:know|have|recall|remember|retain|see|find|access)\b"
    r"|\b(?:do not|don't|does not|doesn't|cannot|can't|unable to|no)\b[^.!?\n]{0,60}"
    r"\b(?:memory|memories|remember|recall|retain|access to your|persistent|previous (?:conversations?|sessions?)|"
    r"past (?:conversations?|sessions?)|personal identity)\b"
    r"|\beach (?:session|conversation) is independent\b",
    re.IGNORECASE)


def is_question(text: str) -> bool:
    return text.rstrip().endswith("?")


def strip_memory_denials(text: str) -> str:
    from .engine import split_sentences
    return " ".join(s for s in split_sentences(text, max_chars=2000) if not _DENIAL_RE.search(s))


def select_for_injection(hits: list, cfg: Dict[str, Any]) -> list:
    """Hits worth putting in front of the model: above the floor and within a band of
    the best one.  A fixed threshold alone does not work: a question and the statement
    that answers it can score 0.25 while an unrelated memory scores 0.28, so what
    matters is how a memory compares with the best match for this particular message."""
    if not hits:
        return []
    cut = max(float(cfg["min_score"]), max(h.score for h in hits) - float(cfg["score_band"]))
    return [h for h in hits if h.score >= cut]


def recall_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {"dual": bool(cfg.get("dual_query", False)), "skip_kinds": (QUESTION_KIND,),
            "lexical": float(cfg.get("lexical_weight", 0.2)),
            "kind_weights": {"said_assistant": float(cfg.get("assistant_weight", 0.75))}}


def _error(message: str) -> str:
    return json.dumps({"error": str(message)[:500]})


def load_config(hermes_home: str | Path) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    try:
        data = json.loads((Path(hermes_home) / "holonomic.json").read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cfg.update(data)
    except (OSError, ValueError):
        pass
    return cfg


def write_config(hermes_home: str | Path, values: Dict[str, Any]) -> None:
    """Merge `values` into holonomic.json atomically."""
    path = Path(hermes_home) / "holonomic.json"
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
        current = current if isinstance(current, dict) else {}
    except (OSError, ValueError):
        current = {}
    current.update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".holonomic-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(current, fh, indent=2)
    os.replace(tmp, path)


# One engine per data directory per process, shared by every agent instance
# (primary agent, subagents, gateway sessions) so they all see the same plates.
_ENGINES: Dict[str, list] = {}
_ENGINES_LOCK = threading.Lock()


def _acquire_engine(data_dir: Path, cfg: Dict[str, Any]):
    from .embed import HashEmbedder, OllamaEmbedder
    from .engine import HolonomicMemory
    key = str(data_dir.resolve())
    with _ENGINES_LOCK:
        entry = _ENGINES.get(key)
        if entry is None:
            if cfg.get("embedder") == "hash":           # offline testing only
                embedder = HashEmbedder()
            else:
                embedder = OllamaEmbedder(cfg["embed_model"], cfg["ollama_host"], timeout=float(cfg["embed_timeout"]))
            engine = HolonomicMemory(data_dir, embedder, dim=int(cfg["dim"]), plate_capacity=float(cfg["plate_capacity"]))
            home = data_dir.parent
            reflector = IdleReflector(engine, lambda: load_config(home), key_fn=extract_keys,
                                      spawn=lambda target, name: spawn_context_thread(target, name=name))
            entry = _ENGINES[key] = [engine, 0, reflector]
        entry[1] += 1
        return entry[0]


def reflector_for(engine) -> Optional[IdleReflector]:
    with _ENGINES_LOCK:
        return next((entry[2] for entry in _ENGINES.values() if entry[0] is engine), None)


def _release_engine(engine) -> None:
    with _ENGINES_LOCK:
        for key, entry in list(_ENGINES.items()):
            if entry[0] is engine:
                entry[1] -= 1
                if entry[1] <= 0:
                    del _ENGINES[key]
                    entry[2].stop()
                    try:
                        engine.close()
                    except Exception as exc:
                        logger.debug("holonomic: close failed: %s", exc)


class HolonomicMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = dict(DEFAULTS)
        self._engine = None
        self._home: Optional[Path] = None
        self._session_id = ""
        self._writes_enabled = True
        self._retry_at = 0.0
        self._last_error = ""
        self._last_count: Optional[int] = None
        self._compressed_at: Dict[str, float] = {}      # session -> time of last context compression
        self._backlog: List[tuple] = []                 # turns that failed to store, retried later
        self._last_user: Dict[str, int] = {}            # session -> id of the user's previous message

    # ------------------------------------------------------------- lifecycle

    @property
    def name(self) -> str:
        return "holonomic"

    def is_available(self) -> bool:
        try:
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        return ("numpy is not installed in Hermes' Python environment. Run `hermes memory setup` and choose "
                "holonomic to install the provider's dependencies (numpy, threadpoolctl), then restart Hermes.")

    def initialize(self, session_id: str, **kwargs) -> None:
        home = kwargs.get("hermes_home")
        if not home:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        self._home = Path(str(home))
        self._cfg = load_config(self._home)
        self._session_id = session_id or ""
        self._writes_enabled = kwargs.get("agent_context", "primary") in ("primary", None, "")
        self._ensure_engine()

    def _ensure_engine(self):
        """Open the engine, or return None if the embedding server can't be reached.
        Creating a new store needs one embedding call, so this is retried lazily."""
        if self._engine is not None:
            return self._engine
        if self._home is None or time.time() < self._retry_at:
            return None
        try:
            self._engine = _acquire_engine(self._home / "holonomic", self._cfg)
            self._last_error = ""
        except Exception as exc:
            self._last_error = str(exc)
            self._retry_at = time.time() + 30.0
            logger.warning("holonomic: memory unavailable (%s); retrying in 30 s", exc)
        return self._engine

    def shutdown(self) -> None:
        if self._engine is not None:
            self._flush_backlog()
            _release_engine(self._engine)
            self._engine = None

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        self._session_id = new_session_id or ""

    # --------------------------------------------------------- system prompt

    def system_prompt_block(self) -> str:
        engine = self._ensure_engine()
        if engine is None:
            return ("# Holonomic Memory\nLong-term memory is currently unreachable"
                    + (f" ({self._last_error[:120]})." if self._last_error else "."))
        try:
            n = engine.stats()["memories"]
        except Exception:
            n = 0
        return ("# Holonomic Memory\n"
                f"Long-term associative memory is active ({n} memories). Memories relevant to each message are recalled "
                "automatically and shown with ids like [#42]; items marked 'linked' were not similar to the message but "
                "are associated with something that was. Recalled memories can be incomplete or out of date: weigh them, "
                "don't recite them. Use the holonomic_memory tool to search deeper, to store something important "
                "(action 'remember'), or to mark a recalled memory 'helpful' or 'wrong'."
                + self._profile_block(engine))

    @staticmethod
    def _profile_block(engine) -> str:
        """Profiles written by reflection.  Always in view, so who the user is never depends on a search."""
        try:
            user, me = engine.profile("user"), engine.profile("self")
        except Exception:
            return ""
        out = ""
        if user:
            out += f"\n\n## What you know about the user (from your own reflection on past conversations)\n{user}"
        if me:
            out += f"\n\n## How you understand yourself (from your own reflection)\n{me}"
        return out

    def _touch(self) -> None:
        reflector = reflector_for(self._engine) if self._engine is not None else None
        if reflector is not None:
            reflector.touch()

    # ---------------------------------------------------------------- recall

    def _visible(self, hit, session_id: str) -> bool:
        """Skip memories from this session that are still in the context window."""
        if not session_id or hit.session != session_id:
            return True
        return hit.created_at < self._compressed_at.get(session_id, 0.0)

    def _line(self, hit, max_chars: int) -> str:
        when = time.strftime("%Y-%m-%d", time.localtime(hit.created_at))
        label = _KIND_LABEL.get(hit.kind, hit.kind)
        # "linked" = it came through an association, not because it resembles the message.
        linked = ", linked" if hit.assoc > 0 and hit.direct < 0.3 else ""
        text = " ".join(hit.text.split())
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "…"
        return f"- [#{hit.id}] ({when}, {label}{linked}) {text}"

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_count = None
        if is_trivial_prompt(query):
            return ""
        engine = self._ensure_engine()
        if engine is None:
            return ""
        sid = session_id or self._session_id
        self._touch()
        try:
            k = int(self._cfg["recall_k"])
            hits = engine.recall(query[:2000], k=k * 3, min_score=float(self._cfg["min_score"]), **recall_options(self._cfg))
            hits = select_for_injection([h for h in hits if self._visible(h, sid)], self._cfg)
        except Exception as exc:
            logger.warning("holonomic: recall failed: %s", exc)
            return ""
        lines, used = [], 0
        for hit in hits:
            if not self._visible(hit, sid):
                continue
            line = self._line(hit, int(self._cfg["max_item_chars"]))
            if used + len(line) > int(self._cfg["max_context_chars"]) or len(lines) >= k:
                break
            lines.append(line)
            used += len(line) + 1
        if not lines:
            return ""
        self._last_count = len(lines)
        return "## Holonomic Memory (recalled; may be incomplete or outdated)\n" + "\n".join(lines)

    def recall_status(self) -> Optional[RecallStatus]:
        return RecallStatus("Holonomic", self._last_count) if self._last_count else None

    # --------------------------------------------------------------- storing

    def _store_turn(self, engine, user: str, assistant: str, sid: str) -> None:
        max_chars = int(self._cfg["max_turn_chars"])
        from .engine import split_sentences
        user = clean_for_storage(user, max_chars)
        only_questions = False
        if user and not is_trivial_prompt(user):
            # One memory per sentence, so each fact is separately findable; the plates
            # chain the sentences back together.  Besides following the previous reply,
            # the first sentence is linked to the user's previous statement, so their
            # train of thought stays connected across replies.
            previous = self._last_user.get(sid)
            units = split_sentences(user)
            only_questions = all(is_question(u) for u in units)
            for unit in units:
                question = is_question(unit)
                ids = engine.remember(unit, kind=QUESTION_KIND if question else "said_user", session=sid,
                                      keys=[] if question else extract_keys(unit), salience=0.5 if question else 1.0,
                                      links=[previous] if previous and not question else [])
                if ids and not question:
                    previous = self._last_user[sid] = ids[-1]
        if self._cfg.get("store_assistant", True):
            assistant = strip_memory_denials(clean_for_storage(assistant, max_chars))
            # A short reply to a bare question is an answer read off the context, not new knowledge.
            if len(assistant) >= (80 if only_questions else 20):
                engine.remember(assistant, kind="said_assistant", session=sid, keys=extract_keys(assistant), salience=0.8)

    def _flush_backlog(self) -> None:
        engine = self._engine
        while engine is not None and self._backlog:
            user, assistant, sid = self._backlog[0]
            try:
                self._store_turn(engine, user, assistant, sid)
            except Exception:
                return
            self._backlog.pop(0)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", **_: Any) -> None:
        # Hermes already calls this on its background worker, one turn at a time.
        if not self._writes_enabled:
            return
        sid = session_id or self._session_id
        engine = self._ensure_engine()
        self._touch()
        try:
            if engine is None:
                raise RuntimeError(self._last_error or "memory unavailable")
            self._flush_backlog()
            self._store_turn(engine, user_content or "", assistant_content or "", sid)
        except Exception as exc:
            if len(self._backlog) < 50:
                self._backlog.append((user_content or "", assistant_content or "", sid))
            logger.warning("holonomic: could not store turn (%s); %d waiting to retry", exc, len(self._backlog))

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        # After compression, older turns of this session leave the context window,
        # so they become eligible for recall again.
        if self._session_id:
            self._compressed_at[self._session_id] = time.time()
        return ""

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror writes to Hermes' built-in memory so they are recallable associatively too."""
        engine = self._ensure_engine()
        if engine is None or not self._writes_enabled:
            return
        try:
            previous = (metadata or {}).get("previous_content")
            if action in ("replace", "remove") and previous:
                for mid in engine.find_by_text(clean_for_storage(previous, 4000), kind="core"):
                    engine.forget(mid)
            if action in ("add", "replace") and content:
                text = clean_for_storage(content, 4000)
                if text and not engine.find_by_text(text, kind="core"):
                    engine.remember(text, kind="core", session="builtin-memory", chain=False, keys=extract_keys(text),
                                    salience=1.5, trust=0.9, meta={"target": target})
        except Exception as exc:
            logger.debug("holonomic: mirroring built-in memory write failed: %s", exc)

    # ------------------------------------------------------------------ tool

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [TOOL_SCHEMA]

    @staticmethod
    def _hit_json(hit) -> Dict[str, Any]:
        return {"id": hit.id, "text": hit.text, "kind": hit.kind,
                "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(hit.created_at)),
                "score": round(hit.score, 3), "similar": round(hit.direct, 3), "linked": round(hit.assoc, 3),
                "trust": round(hit.trust, 2)}

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != TOOL_SCHEMA["name"]:
            return _error(f"Unknown tool: {tool_name}")
        engine = self._ensure_engine()
        if engine is None:
            return _error(f"Memory is unavailable: {self._last_error or 'embedding server not reachable'}")
        action = (args or {}).get("action")
        limit = max(1, min(int(args.get("limit") or 8), 25))
        try:
            if action == "recall":
                query = (args.get("query") or "").strip()
                if not query:
                    return _error("recall needs 'query'")
                hits = engine.recall(query, k=limit, min_score=0.15, **recall_options(self._cfg))
                return json.dumps({"results": [self._hit_json(h) for h in hits], "count": len(hits)})
            if action == "remember":
                content = clean_for_storage(args.get("content") or "", 4000)
                if not content:
                    return _error("remember needs 'content'")
                importance = max(1, min(int(args.get("importance") or 2), 3))
                keys = list(dict.fromkeys([str(k) for k in (args.get("about") or [])] + extract_keys(content)))[:6]
                ids = engine.remember(content, kind="note", session=self._session_id, keys=keys, chain=False,
                                      salience=(0.8, 1.2, 1.8)[importance - 1], trust=0.8)
                return json.dumps({"stored": ids, "filed_under": keys})
            if action == "related":
                if args.get("memory_id") is not None:
                    hits = engine.associates(int(args["memory_id"]), k=limit)
                elif (args.get("entity") or "").strip():
                    hits = engine.recall(keys=[args["entity"]], k=limit, min_score=0.2)
                else:
                    return _error("related needs 'memory_id' or 'entity'")
                return json.dumps({"results": [self._hit_json(h) for h in hits], "count": len(hits)})
            if action in ("feedback", "forget"):
                if args.get("memory_id") is None:
                    return _error(f"{action} needs 'memory_id'")
                mid = int(args["memory_id"])
                current = engine.get(mid)
                if current is None:
                    return _error(f"No memory with id {mid}")
                if action == "forget":
                    return json.dumps({"forgotten": engine.forget(mid)})
                rating = args.get("rating")
                if rating == "helpful":
                    engine.set_trust(mid, current["trust"] + 0.1)
                    engine.reinforce([mid], 0.2)
                elif rating == "wrong":
                    engine.set_trust(mid, current["trust"] - 0.2)
                else:
                    return _error("feedback needs 'rating': helpful or wrong")
                return json.dumps({"id": mid, "trust": round(engine.get(mid)["trust"], 2)})
            if action == "stats":
                return json.dumps(engine.stats())
            return _error(f"Unknown action: {action}")
        except Exception as exc:
            logger.warning("holonomic: tool call failed: %s", exc)
            return _error(exc)

    # ---------------------------------------------------------------- config

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "ollama_host", "description": "Ollama server address", "default": DEFAULTS["ollama_host"]},
            {"key": "embed_model", "description": "Ollama embedding model", "default": DEFAULTS["embed_model"]},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        write_config(hermes_home, {k: v for k, v in (values or {}).items() if v not in (None, "")})

    def get_status_config(self, provider_config: Any = None) -> Dict[str, Any]:
        """Shown by `hermes memory status`, which calls this on an uninitialised instance."""
        cfg = self._cfg
        if self._home is None:
            try:
                from hermes_constants import get_hermes_home
                cfg = load_config(get_hermes_home())
            except Exception:
                pass
        return {k: cfg.get(k) for k in ("ollama_host", "embed_model", "recall_k", "min_score")}
