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

from . import images as _images
from .images import IMAGE_DEFAULTS
from .reflect import (DREAM_TALK_ASSISTANT, DREAM_TALK_KINDS, DREAM_TALK_USER, IMAGE, IMAGE_KINDS, IMAGE_PART, REFLECT_DEFAULTS,
                      SUBJECTS, IdleReflector, read_foundation)
from .sleep import DREAM, DREAM_INSIGHT, DREAM_REALM, EPISODE, SLEEP_DEFAULTS, dreams as list_dreams, latest_dream

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
DEFAULTS.update(SLEEP_DEFAULTS)
DEFAULTS.update(IMAGE_DEFAULTS)

_DREAM_WORD_RE = re.compile(r"\bdream(?:s|t|ed|ing)?\b", re.IGNORECASE)

TOOL_SCHEMA = {
    "name": "holonomic_memory",
    "description": (
        "Long-term associative memory. Relevant memories are recalled automatically every turn; use this tool "
        "to dig deeper or to manage memories.\n"
        "ACTIONS:\n"
        "- recall: search memory for `query`. Returns memories similar to it and memories linked to those. If an "
        "ordinary recall does not find what you are looking for, set `deep` to true: that also searches memories "
        "that have faded with time and follows links further. Set "
        "`subject` to search only what you have concluded about the user ('user'), about yourself ('self') or "
        "about the two of you ('us').\n"
        "- remember: store `content` as a durable fact, preference or note. Use for things worth keeping that "
        "might not be obvious from the conversation alone. Optional `about` (names/topics) and `importance` 1-3.\n"
        "- related: what is linked to a memory (`memory_id`) or filed under a name/topic (`entity`).\n"
        "- feedback: rate `memory_id` as `helpful` or `wrong`. Wrong memories sink, helpful ones rise.\n"
        "- forget: permanently remove `memory_id`. Only when asked to, or when a memory is clearly false.\n"
        "- dreams: your recent dreams and what you made of them. Dreams are not things that happened.\n"
        "- images: images you have been shown. With `label`, every image in which that thing was noticed (e.g. 'cat'); "
        "with `query`, images whose description matches; with `image_id`, one image and what is in each part of it; "
        "with none of these, the most recent. If the user tells you a description is wrong, pass `image_id` and "
        "`correction` (what they said, in their words): the image is looked at again with that taken as true. Each result has a `file`: to show the image to the user, write "
        "MEDIA: followed by that file path on a line of its own in your reply.\n"
        "- look: look at a stored image again (`image_id`) to answer a `question` its description does not cover. "
        "Optional `section` (a part number from 'images') to look closely at one part.\n"
        "- stats: size of the memory store."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["recall", "remember", "related", "feedback", "forget", "dreams", "images",
                                                  "look", "stats"]},
            "label": {"type": "string", "description": "A thing to find in images, e.g. 'cat' (images)."},
            "image_id": {"type": "integer", "description": "An image id as shown, e.g. image #3 (images, look)."},
            "question": {"type": "string", "description": "What you want to know about the image (look)."},
            "correction": {"type": "string", "description": "What the user said is wrong about an image's description (images)."},
            "section": {"type": "integer", "description": "A part of the image to look at closely (look)."},
            "deep": {"type": "boolean", "description": "Recall only: also search faded memories and follow links further."},
            "query": {"type": "string", "description": "What to search for (recall)."},
            "subject": {"type": "string", "enum": ["user", "self", "us"],
                        "description": "Limit recall to your own conclusions about this subject."},
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

_KIND_LABEL = {DREAM_TALK_USER: "user said, talking about a dream of yours",
               DREAM_TALK_ASSISTANT: "you said, describing a dream you had; not something that happened",
               EPISODE: "your account of a past conversation", DREAM: "a dream you had",
               DREAM_INSIGHT: "what you made of a dream", "fact": "learned about the user", "self_note": "your own note", "bond_note": "about the two of you",
               "insight": "insight", IMAGE: "an image you were shown", IMAGE_PART: "part of an image you were shown",
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
    text = _images.strip_image_markers(text)          # Hermes' notes about an attached image are not the user's words
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


# Words that mark a sentence as being about how the memory system works, as opposed to using it.
# "I can't reach faded memories without deep recall" is a true statement about the design and worth
# keeping; "I don't have a memory of our past conversations" is the denial the filter exists for.
_MECHANICS_RE = re.compile(
    r"\b(?:plugin|feature|settings?|config\w*|prompt|realms?|plates?|database|memory (?:system|provider|plugin|store|engine)|"
    r"sleep cycle|command|version|implement\w*|algorithm|embeddings?|threshold|parameter|label(?:s|led|ed|ling)?|detector|"
    r"storage|fad(?:e|es|ed|ing)|reflection (?:step|pass|process)|consolidat\w*|deep recall|holonomic|holograph\w*)\b",
    re.IGNORECASE)


def strip_memory_denials(text: str) -> str:
    from .engine import split_sentences
    return " ".join(s for s in split_sentences(text, max_chars=2000)
                    if not _DENIAL_RE.search(s) or _MECHANICS_RE.search(s))


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
    return {"dual": bool(cfg.get("dual_query", False)), "skip_kinds": (QUESTION_KIND,), "min_trust": 0.15,
            "min_strength": float(cfg.get("fade_threshold", 0.35)),      # faded memories need deep recall
            "lexical": float(cfg.get("lexical_weight", 0.2)),
            "kind_weights": {"said_assistant": float(cfg.get("assistant_weight", 0.75)),
                             DREAM_TALK_ASSISTANT: 0.5, DREAM_TALK_USER: 0.75, IMAGE_PART: 0.9}}


# The user referring to a dream of the agent's ("your dream", "did you dream"), as opposed to "my dream job".
_HER_DREAM_RE = re.compile(r"\b(?:your|that|this|the|last night'?s?)\s+(?:\w+\s+){0,2}dreams?\b|\bdid you dream|"
                           r"\byou (?:dreamt|dreamed|were dreaming)\b|\bany dreams\b|\bin (?:your|the|that) dream\b", re.IGNORECASE)
# The user asking what was dreamed.  Only this makes the whole reply dream talk; a passing mention
# ("your dreams sound like ours") labels that sentence and lets the reply be judged on its own.
_ASKS_DREAM_RE = re.compile(
    r"\bdid you (?:have )?(?:any |a )?dream|\bhave you (?:had )?(?:any |a )?dream|\bwhat (?:did|do) you dream|\bany dreams\b|"
    r"\bwhat (?:was|were|happened in) (?:your|the|that|last night'?s?) dreams?\b|"
    r"\b(?:tell me|talk to me|more) (?:more )?about (?:your|the|that|those|last night'?s?) (?:\w+ ){0,2}dreams?\b|"
    r"\b(?:describe|share|recount|remember) (?:your|the|that|those|last night'?s?) (?:\w+ ){0,2}dreams?\b|"
    r"\bwhat (?:do|did) you (?:make|think) of (?:your|the|that|those) dreams?\b|\bdream(?:t|ed)? (?:of|about) (?:anything|something)\b",
    re.IGNORECASE)
# The agent recounting one.
_MY_DREAM_RE = re.compile(r"\b(?:my|the|that|this|a|last night'?s?)\s+(?:\w+\s+){0,2}dreams?\b|\bI (?:dreamt|dreamed|was dreaming)\b|"
                          r"\bin (?:my|the|that) dream\b", re.IGNORECASE)


def is_dream_talk(engine, memory_id: int, text: str, speaker: str, cfg: Dict[str, Any]) -> bool:
    """Is this piece of conversation about a dream the agent had?  Either it says so, or it
    resembles a stored dream closely (a follow-up about the dream's imagery need not use the word)."""
    try:
        if not engine.recent(1, realm=DREAM_REALM, kind=DREAM):
            return False                                    # no dreams yet: "dream" can only be meant in the everyday sense
        # Mentioning a dream is enough, unless the sentence is about how dreaming works ("your dreams are
        # kept in a separate realm").  Then, as for anything that does not mention one, it has to resemble a dream.
        if (_HER_DREAM_RE if speaker == "user" else _MY_DREAM_RE).search(text) and not _MECHANICS_RE.search(text):
            return True
        return engine.similarity_to_realm(memory_id, DREAM_REALM, kinds=(DREAM,)) >= float(cfg.get("dream_talk_similarity", 0.55))
    except Exception:
        return False


def asks_about_dreams(engine, text: str) -> bool:
    """Did the user just ask about the agent's dreams?  The reply to that is dream talk even when it
    never uses the word: a real reply began "I did, actually. They were quite vivid" and went on to
    describe a hall with a glass floor."""
    try:
        return (bool(_ASKS_DREAM_RE.search(text or "")) and not _MECHANICS_RE.search(text or "")
                and bool(engine.recent(1, realm=DREAM_REALM, kind=DREAM)))
    except Exception:
        return False


def mark_dream_talk(engine, memory_id: int, text: str, kind: str, cfg: Dict[str, Any], force: bool = False) -> Optional[str]:
    """Relabel a stored piece of conversation if it is dream talk.  Returns the new kind, or None."""
    if kind not in ("said_user", "said_assistant"):
        return None
    speaker = "user" if kind == "said_user" else "assistant"
    if not force and not is_dream_talk(engine, memory_id, text, speaker, cfg):
        return None
    new = DREAM_TALK_USER if speaker == "user" else DREAM_TALK_ASSISTANT
    engine.set_kind(memory_id, new)
    return new


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
                                      spawn=lambda target, name: spawn_context_thread(target, name=name),
                                      foundation_fn=lambda: read_foundation(home))
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
        self._looked_at: set = set()                    # images kept while preparing a reply, not to be counted twice

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
                + self._images_block(engine) + self._profile_block(engine) + self._latest_dream_block(engine))

    def _images_block(self, engine) -> str:
        if not self._cfg.get("image_enabled"):
            return ""
        try:
            n = _images.count_images(engine)
        except Exception:
            return ""
        return (f"\nImages you are shown are kept ({n} so far) along with a description of what you saw in each, so you "
                "can recall them like anything else. A recalled image comes with its file: to show it to the user, write "
                "MEDIA: followed by that file path on a line of its own. Use the tool's 'images' action to find images by "
                "what is in them, and 'look' to look at one again when its description does not answer the question.")

    @staticmethod
    def _profile_block(engine) -> str:
        """Profiles written by reflection.  Always in view, so who the user is never depends on a search."""
        try:
            user, me, us = engine.profile("user"), engine.profile("self"), engine.profile("us")
        except Exception:
            return ""
        out = ""
        if user:
            out += f"\n\n## What you know about the user (from your own reflection on past conversations)\n{user}"
        if me:
            out += f"\n\n## Who you have become (from your own reflection; your core identity is defined elsewhere)\n{me}"
        if us:
            out += f"\n\n## Your relationship with the user (from your own reflection)\n{us}"
        return out

    def _latest_dream_block(self, engine) -> str:
        try:
            d = latest_dream(engine, self._cfg)
        except Exception:
            return ""
        if not d:
            return ""
        when = time.strftime("%A %d %B", time.localtime(d["at"]))
        thoughts = (d.get("meta") or {}).get("thoughts", "")
        return (f"\n\n## Your most recent dream ({when}; a dream, not something that happened)\n{d['text']}"
                + (f"\nWhat you made of it on waking: {thoughts}" if thoughts else "")
                + self._dream_pictures_text(d.get("pictures"), "\n")
                + "\nYou can mention or discuss it if it comes up or seems worth sharing. Do not treat it as fact.")

    @staticmethod
    def _dream_pictures_text(pictures: Optional[list], lead: str) -> str:
        """Pictures of a dream, with their files so she can show them.  They are pictures of a dream, not of anything real."""
        if not pictures:
            return ""
        return (f"{lead}Pictures of moments in this dream (made from the dream; not photographs of anything real). To show one, "
                "write MEDIA: followed by its file path on a line of its own:"
                + "".join(f"{lead}  - {p['scene']} (file: {p['file']})" for p in pictures))

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
        words = _images.strip_image_markers(query)       # recall on what was said, not on Hermes' note about a file
        k = int(self._cfg["recall_k"])
        hits = []
        try:
            if words and not is_trivial_prompt(words):
                hits = engine.recall(words[:2000], k=k * 3, min_score=float(self._cfg["min_score"]), **recall_options(self._cfg))
                hits = select_for_injection([h for h in hits if self._visible(h, sid)], self._cfg)
        except Exception as exc:
            logger.warning("holonomic: recall failed: %s", exc)
            return ""
        lines, used, shown = [], 0, []
        for hit, image_id, parts in _images.group_hits(hits):      # an image and its parts make one entry
            line = (self._image_line(engine, hit, image_id, parts, int(self._cfg["max_item_chars"])) if image_id
                    else self._line(hit, int(self._cfg["max_item_chars"])))
            if used + len(line) > int(self._cfg["max_context_chars"]) or len(lines) >= k:
                break
            lines.append(line)
            shown += [hit.id] + [p.id for p in parts]
            used += len(line) + 1
        extra = [b for b in (self._seen_before_block(engine, query), self._unseen_block(engine, query, sid),
                             self._dream_block(engine, query) if _DREAM_WORD_RE.search(query) else "") if b]
        if not lines:
            return "\n\n".join(extra)
        try:                                             # what gets used stays strong; what is never recalled fades
            engine.reinforce(shown, 0.03)
        except Exception:
            pass
        self._last_count = len(lines)
        return "\n\n".join(["## Holonomic Memory (recalled; may be incomplete or outdated)\n" + "\n".join(lines)] + extra)

    def _image_line(self, engine, hit, image_id: int, parts: list, max_chars: int) -> str:
        """One entry for an image: where its file is, what it shows, and which parts of it matched."""
        when = time.strftime("%Y-%m-%d", time.localtime(hit.created_at))
        clip = lambda t: (lambda t: t if len(t) <= max_chars else t[:max_chars].rsplit(" ", 1)[0] + "…")(" ".join(t.split()))  # noqa: E731
        try:
            img = _images.get_image(engine, image_id)
        except Exception:
            img = None
        if not img:
            return self._line(hit, max_chars)
        whole = hit.text if hit.kind == IMAGE else img["description"]
        line = f"- [#{img['memory_id'] or hit.id}] ({when}, image #{image_id} you were shown; file: {img['file']}) {clip(whole)}"
        for p in parts:
            line += f"\n    in the {(p.meta or {}).get('place', 'image')} [#{p.id}]: {clip(p.text)}"
        return line

    def _seen_before_block(self, engine, query: str) -> str:
        """If the image attached to this message is one she has already been shown, say so: the file is
        identical, so this is certain."""
        if not self._cfg.get("image_enabled"):
            return ""
        out = []
        try:
            for data, _ in _images.images_in_turn(query, None, int(self._cfg.get("image_max_bytes", 30_000_000)))[:4]:
                img = _images.known(engine, data)
                if img and img["realm"] == "dream":
                    out.append(f"- this is a picture from one of your own dreams (not a photograph of anything real): {img['caption']}")
                elif img:
                    when = time.strftime("%Y-%m-%d", time.localtime(img["created_at"]))
                    out.append(f"- image #{img['id']}, first shown {when}, seen {img['seen']} time(s) before now"
                               + (f": {' '.join(img['description'].split())}" if img["description"] else ""))
        except Exception as exc:
            logger.debug("holonomic: checking for a known image failed: %s", exc)
        return ("## An image you have seen before (the attached file is identical to one already in your memory)\n"
                + "\n".join(out)) if out else ""

    def _unseen_block(self, engine, query: str, sid: str) -> str:
        """A picture attached as a plain file (the desktop app does this with a phone's HEIC photos) is never
        shown to the model: it is only told a file exists.  So it is looked at here, before she replies, and
        she is given what was seen, instead of answering about a picture she has not seen."""
        if not self._cfg.get("image_enabled") or not self._writes_enabled:
            return ""
        out = []
        caption = clean_for_storage(query, 600)
        for path in _images.attached_as_files(query)[:3]:
            try:
                p = Path(path)
                if not p.is_file() or p.stat().st_size > int(self._cfg.get("image_max_bytes", 30_000_000)):
                    continue
                data = p.read_bytes()
                if _images.known(engine, data):
                    continue                             # the seen-before block covers it
                img = _images.add_image(engine, data, self._cfg, origin=path, caption=caption, session=sid)
                self._looked_at.add(img["id"])
                try:
                    img = _images.describe(engine, self._cfg, img["id"], key_fn=extract_keys) or img
                except Exception as exc:
                    logger.warning("holonomic: image #%s kept; describing it failed and will be retried (%s)", img["id"], exc)
                out.append(f"- image #{img['id']} ({p.name}; file: {img['file']}): "
                           + (" ".join(img["description"].split()) if img.get("description")
                              else "it could not be looked at just now. Say so; do not describe it from the user's words."))
            except Exception as exc:
                logger.warning("holonomic: could not look at an attached picture (%s)", exc)
        return ("## The picture attached to this message (you were not shown it directly; this is what you saw when it was "
                "looked at for you just now)\n" + "\n".join(out)) if out else ""

    def _dream_block(self, engine, query: str) -> str:
        """Dreams are only brought up when the conversation turns to them, and always labelled."""
        try:
            found = list_dreams(engine, 3)
        except Exception:
            return ""
        if not found:
            return ""
        out = ["## Your dreams (dreams you had while idle; not things that happened. If you talk about them, "
               "speak of them as dreams.)"]
        for d in found:
            when = time.strftime("%Y-%m-%d", time.localtime(d["created_at"]))
            out.append(f"- ({when}) {' '.join(d['text'].split())}"
                       + (f"\n  What you made of it: {d['thoughts']}" if d.get("thoughts") else "")
                       + self._dream_pictures_text(d.get("pictures"), "\n  ")
                       + "".join(f"\n  A connection you noticed: {c}" for c in d.get("connections", [])))
        return "\n".join(out)

    def recall_status(self) -> Optional[RecallStatus]:
        return RecallStatus("Holonomic", self._last_count) if self._last_count else None

    # --------------------------------------------------------------- storing

    def _store_turn(self, engine, user: str, assistant: str, sid: str, pictures: Optional[list] = None) -> None:
        said = self._store_words(engine, user, assistant, sid)
        if pictures:
            self._store_images(engine, pictures, user, sid, said)

    def _store_images(self, engine, pictures: list, user: str, sid: str, said: List[int]) -> None:
        """Keep the images attached to this message, linked to what was said about them, and describe each
        one now.  If the vision model cannot be reached the image is kept and described later."""
        caption = clean_for_storage(user, 600)
        for data, origin in pictures:
            try:
                before = _images.known(engine, data)
                img = _images.add_image(engine, data, self._cfg, origin=origin, caption=caption, session=sid, links=said[:2],
                                        count=not (before and before["id"] in self._looked_at))
                self._looked_at.discard(img["id"])       # it was kept while preparing this reply: one showing, not two
            except Exception as exc:
                logger.warning("holonomic: could not keep an image (%s)", exc)
                continue
            if not img["new"] or not _images.image_config(self._cfg)["image_model"]:
                continue
            try:
                _images.describe(engine, self._cfg, img["id"], key_fn=extract_keys)
            except Exception as exc:
                logger.warning("holonomic: image #%s kept; describing it failed and will be retried (%s)", img["id"], exc)

    def _store_words(self, engine, user: str, assistant: str, sid: str) -> List[int]:
        """Store what was said.  Returns the ids: the user's first statement, then the reply's first part."""
        said: List[int] = []
        first_reply: List[int] = []
        max_chars = int(self._cfg["max_turn_chars"])
        from .engine import split_sentences
        user = clean_for_storage(user, max_chars)
        only_questions = False
        dream_turn = asks_about_dreams(engine, user)
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
                    said += ids[:1] if not said else []
                    previous = self._last_user[sid] = ids[-1]
                    mark_dream_talk(engine, ids[-1], unit, "said_user", self._cfg)
        if self._cfg.get("store_assistant", True):
            # Stored a paragraph at a time, so a reply that recounts a dream and then turns to something
            # real is labelled part by part.
            parts = [strip_memory_denials(p) for p in re.split(r"\n\s*\n", clean_for_storage(assistant, max_chars))]
            parts = [p for p in parts if len(p) >= 20]
            # A short reply to a bare question is an answer read off the context, not new knowledge.
            if sum(len(p) for p in parts) >= (80 if only_questions else 20):
                for part in parts:
                    for mid in engine.remember(part, kind="said_assistant", session=sid, keys=extract_keys(part), salience=0.8):
                        first_reply += [mid] if not first_reply else []
                        stored = engine.get(mid)
                        if stored:
                            mark_dream_talk(engine, mid, stored["text"], "said_assistant", self._cfg, force=dream_turn)
        return said + first_reply

    def _flush_backlog(self) -> None:
        engine = self._engine
        while engine is not None and self._backlog:
            user, assistant, sid, pictures = self._backlog[0]
            try:
                self._store_turn(engine, user, assistant, sid, pictures)
            except Exception:
                return
            self._backlog.pop(0)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: Optional[List[Dict[str, Any]]] = None, **_: Any) -> None:
        # Hermes already calls this on its background worker, one turn at a time.
        if not self._writes_enabled:
            return
        sid = session_id or self._session_id
        pictures: list = []
        if self._cfg.get("image_enabled"):
            try:                                         # read now: Hermes may clear its upload folder later
                pictures = _images.images_in_turn(user_content or "", messages, int(self._cfg.get("image_max_bytes", 30_000_000)))
            except Exception as exc:
                logger.warning("holonomic: could not read the attached image (%s)", exc)
        engine = self._ensure_engine()
        self._touch()
        try:
            if engine is None:
                raise RuntimeError(self._last_error or "memory unavailable")
            self._flush_backlog()
            self._store_turn(engine, user_content or "", assistant_content or "", sid, pictures)
        except Exception as exc:
            if len(self._backlog) < 50:
                self._backlog.append((user_content or "", assistant_content or "", sid, pictures))
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

    def _hit_json(self, hit) -> Dict[str, Any]:
        out = {"id": hit.id, "text": hit.text, "kind": hit.kind,
               "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(hit.created_at)),
               "score": round(hit.score, 3), "similar": round(hit.direct, 3), "linked": round(hit.assoc, 3),
               "trust": round(hit.trust, 2)}
        if hit.kind in IMAGE_KINDS and (hit.meta or {}).get("image_id") and self._engine is not None:
            img = _images.get_image(self._engine, int(hit.meta["image_id"]))
            if img:
                out.update(image_id=img["id"], file=img["file"])
                if hit.kind == IMAGE_PART:
                    out["part_of_image"] = hit.meta.get("place", "")
        return out

    @staticmethod
    def _image_json(img: dict, **extra: Any) -> Dict[str, Any]:
        out = {"image_id": img["id"], "shown": time.strftime("%Y-%m-%d %H:%M", time.localtime(img["created_at"])),
               "file": img["file"], "size": f"{img['width']}x{img['height']}",
               "description": img["description"] or "(not described yet)", "things_in_it": img["labels"]}
        if img.get("caption"):
            out["said_when_shown"] = img["caption"]
        if img.get("seen", 1) > 1:
            out["times_shown"] = img["seen"]
        if img.get("sections_waiting"):
            out["parts_not_yet_looked_at"] = img["sections_waiting"]
        if "sections" in img:
            out["parts"] = [{"section": s["section"], "place": s["place"], "description": s["description"]}
                            for s in img["sections"] if s["description"]]
        out.update({k: v for k, v in extra.items() if v})
        return out

    def _images_action(self, engine, args: Dict[str, Any], limit: int) -> str:
        how = "To show an image to the user, write MEDIA: followed by its file path on a line of its own."
        if args.get("image_id") is not None and (args.get("correction") or "").strip():
            img = _images.redescribe(engine, self._cfg, int(args["image_id"]), correction=args["correction"], key_fn=extract_keys)
            if not img:
                return _error(f"No image with id {args['image_id']}")
            return json.dumps(dict(self._image_json(img), note="Described again with the correction taken as true. Its parts will be "
                                   "looked at again when the conversation is quiet."))
        if args.get("image_id") is not None:
            img = _images.get_image(engine, int(args["image_id"]), sections=True)
            return json.dumps(dict(self._image_json(img), note=how)) if img else _error(f"No image with id {args['image_id']}")
        if (args.get("label") or "").strip():
            found = _images.find_by_label(engine, args["label"])
            return json.dumps({"label": args["label"], "count": len(found), "note": how + " This is every image in which it was "
                               "noticed; an image where it went unnoticed is not listed.",
                               "images": [self._image_json(i, matched=i["matched"], noticed_in=i["matched_in"]) for i in found[:limit]],
                               "more": max(0, len(found) - limit)})
        if (args.get("query") or "").strip():
            options = recall_options(self._cfg)
            if args.get("deep"):
                options.update(min_strength=0.0, reach=2)
            hits = engine.recall(args["query"].strip(), k=limit * 3, min_score=0.1, only_kinds=IMAGE_KINDS, **options)
            out = []
            for hit, image_id, parts in _images.group_hits(hits)[:limit]:
                img = _images.get_image(engine, image_id) if image_id else None
                if img:
                    out.append(self._image_json(img, score=round(hit.score, 3), matching_parts=[
                        {"place": (p.meta or {}).get("place", ""), "description": p.text} for p in parts]))
            return json.dumps({"count": len(out), "note": how, "images": out})
        found = _images.list_images(engine, limit)
        return json.dumps({"count": len(found), "total": _images.count_images(engine), "note": how,
                           "images": [self._image_json(i) for i in found]})

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
                subject = args.get("subject")
                if subject and subject not in SUBJECTS:
                    return _error("subject must be 'user', 'self' or 'us'")
                only = {"only_kinds": (SUBJECTS[subject],)} if subject else {}
                options = recall_options(self._cfg)
                threshold = options["min_strength"]
                if args.get("deep"):
                    options.update(min_strength=0.0, reach=2)
                hits = engine.recall(query, k=limit, min_score=0.0 if subject else (0.1 if args.get("deep") else 0.15),
                                     **options, **only)
                if args.get("deep"):
                    # Recovering a faded memory brings it back within everyday reach.
                    faded = [h.id for h in hits if h.strength < threshold]
                    if faded:
                        engine.reinforce(faded, 0.3)
                if subject:
                    return json.dumps({"profile": engine.profile(subject), "results": [self._hit_json(h) for h in hits],
                                       "count": len(hits)})
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
                    image_id = (current.get("meta") or {}).get("image_id") if current["kind"] == IMAGE else None
                    if image_id:            # forgetting what an image showed forgets the image; its file stays on disk
                        return json.dumps({"forgotten": _images.forget_image(engine, int(image_id)), "image_id": image_id})
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
            if action == "dreams":
                found = list_dreams(engine, limit)
                return json.dumps({"note": "These are dreams, not things that happened.", "count": len(found), "dreams": [
                    {"id": d["id"], "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(d["created_at"])),
                     "dream": d["text"], "what_you_made_of_it": d.get("thoughts", ""),
                     "connections_noticed": d.get("connections", []),
                     **({"pictures_of_it": [{"scene": p["scene"], "file": p["file"]} for p in d["pictures"]],
                         "to_show_a_picture": "write MEDIA: followed by its file path on a line of its own"}
                        if d.get("pictures") else {})} for d in found]})
            if action == "images":
                return self._images_action(engine, args, limit)
            if action == "look":
                if args.get("image_id") is None or not (args.get("question") or "").strip():
                    return _error("look needs 'image_id' and 'question'")
                answer = _images.look(engine, self._cfg, int(args["image_id"]), args["question"],
                                      section=args.get("section") if args.get("section") is not None else None)
                return json.dumps({"image_id": int(args["image_id"]), "answer": answer})
            if action == "stats":
                return json.dumps(dict(engine.stats(), images=_images.count_images(engine)))
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
