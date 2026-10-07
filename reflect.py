"""Reflection: a language model reads recent memories and distils what is worth keeping.

One pass takes the memories stored since the last pass and works in up to three
small steps, each a separate model call with one job:

  1. propose   read the memories; write facts about the user, the agent's notes about
               itself, notes about the relationship, insights, and any stored fact the
               user has contradicted.  Every item cites the memories it rests on.
  2. check     (depth 3) re-read each proposed item against only the lines it cites;
               keep it, rewrite it, or drop it.
  3. profiles  rewrite the short profiles of the user, the agent and the two together
               from the items that survived, never from the raw conversation.

Asking for all of this in one call overloaded a small model: it recorded its own
phrases as the user's values, and with reasoning switched on it deliberated until
it ran out of tokens without answering.  Small steps keep each call simple.

Surviving items are stored as memories linked (through the plates) to the memories
they came from.  The profiles are always in view (system prompt), so identity never
depends on a search matching.  The user's SOUL.md is shown to the model as a fixed
foundation: the agent's view of itself grows within it and never rewrites it.

Off by default.  Enable with `hermes holonomic reflect on --model NAME`.
Only stdlib imports here: Hermes executes this file when it loads the plugin.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Kinds produced by reflection.  They are never fed back in as input.
FACT, SELF_NOTE, BOND_NOTE, INSIGHT = "fact", "self_note", "bond_note", "insight"
# What the user is working on, as against who the user is.  Kept apart so that the portrait of the person is not
# crowded out by how a project of theirs works inside.
PROJECT_FACT = "project_fact"
FACT_KINDS = (FACT, PROJECT_FACT)
DERIVED_KINDS = (FACT, PROJECT_FACT, SELF_NOTE, BOND_NOTE, INSIGHT, "dream")
# Subjects the agent can ask its memory about, and the profile and kind that belong to each.
SUBJECTS = {"user": FACT, "self": SELF_NOTE, "us": BOND_NOTE, "projects": PROJECT_FACT}
FOUNDATION_MAX_CHARS = 20000      # a real SOUL.md ran to 10,000 characters; 6,000 cut it off mid-section
WATERMARK = "reflect:last_id"
# Conversation about a dream the agent had.  The conversation really happened, so it is an ordinary
# waking memory, but what was said in it describes a dream.  These kinds keep that from being read
# as fact: they are labelled wherever they are shown, and no fact about the user may rest on them.
DREAM_TALK_USER, DREAM_TALK_ASSISTANT = "dreamtalk_user", "dreamtalk_assistant"
DREAM_TALK_KINDS = (DREAM_TALK_USER, DREAM_TALK_ASSISTANT)
ASSISTANT_KINDS = ("said_assistant", DREAM_TALK_ASSISTANT)
# What a vision model saw in an image the agent was shown, and in each part of it (see images.py).
# Reflection does not read these: what is in a picture is not something the user said.
IMAGE = "image"
IMAGE_PART = "image_part"
IMAGE_KINDS = (IMAGE, IMAGE_PART)

REFLECT_DEFAULTS: Dict[str, Any] = {
    "reflect_enabled": False,
    "reflect_model": "",              # e.g. the Ollama name of your chat model
    "reflect_host": "",               # empty = same server as ollama_host
    # 1 = facts about the user and the user profile only
    # 2 = everything, proposed and stored unchecked
    # 3 = everything, with each item checked against the lines it cites before it is stored
    "reflect_depth": 3,
    "reflect_min_new": 12,            # wait until this many new memories exist
    "reflect_idle_seconds": 300,      # and the conversation has been quiet this long
    "reflect_batch": 60,              # memories read per pass
    "reflect_timeout": 600.0,
    "merge_min_similarity": 0.8,      # statements closer than this are looked at as possibly saying the same thing
    "merge_in_sleep": True,           # unattended sleep retires a statement that repeats another word for word (no model asked)
    "reflect_max_tokens": 2000,       # cap on each reply; without one a model can generate until its context is full
    # Let a reasoning model think first.  Off by default: on the model this was developed against,
    # thinking ran through its whole token budget without answering in three runs out of four (about
    # 140 s wasted per step), while the same steps without thinking took 6-14 s and gave 9 specific,
    # correctly attributed facts.  If thinking is on and runs away, the step is repeated without it.
    "reflect_think": False,
    "reflect_temperature": 0.3,
    "profile_max_chars": 1200,
}

_ITEMS = {"type": "array", "items": {"type": "object", "properties": {
    "text": {"type": "string"}, "sources": {"type": "array", "items": {"type": "integer"}}},
    "required": ["text", "sources"]}}
_SUPERSEDED = {"type": "array", "items": {"type": "object", "properties": {
    "fact": {"type": "integer"}, "replacement": {"type": "string"},
    "sources": {"type": "array", "items": {"type": "integer"}}}, "required": ["fact", "replacement", "sources"]}}

_PROPOSE_SCHEMA = {"type": "object", "properties": {
    "user_facts": _ITEMS, "project_facts": _ITEMS, "self_notes": _ITEMS, "relationship_notes": _ITEMS, "insights": _ITEMS,
    "superseded": _SUPERSEDED},
    "required": ["user_facts", "project_facts", "self_notes", "relationship_notes", "insights", "superseded"]}
_PROPOSE_BASIC_SCHEMA = {"type": "object", "properties": {"user_facts": _ITEMS, "project_facts": _ITEMS, "superseded": _SUPERSEDED},
                         "required": ["user_facts", "project_facts", "superseded"]}
_CHECK_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
    "item": {"type": "integer"}, "verdict": {"type": "string", "enum": ["keep", "rewrite", "drop"]},
    "text": {"type": "string"}}, "required": ["item", "verdict", "text"]}}}, "required": ["verdicts"]}
_PROFILE_SCHEMA = {"type": "object", "properties": {
    "user_profile": {"type": "string"}, "projects_profile": {"type": "string"}, "self_profile": {"type": "string"},
    "relationship_profile": {"type": "string"}},
    "required": ["user_profile", "projects_profile", "self_profile", "relationship_profile"]}
# Used when a caller asks for no particular step (kept for tests and older callers).
_SCHEMA = _PROPOSE_SCHEMA

# The reply is JSON, so a double quotation mark inside a sentence ends the string there.  A real run
# returned an account that stopped at "her original vision for a" for exactly that reason.
NO_DOUBLE_QUOTES = ("Inside the text you write, never use the double quotation mark character; if you need to quote "
                    "something, use single quotation marks.")

_SYSTEM = (
    "You are the reflective process of an AI assistant's long-term memory. You never invent anything: everything "
    "you write must be supported by the numbered lines you are shown. Reply with JSON only. " + NO_DOUBLE_QUOTES
)

_ENDS_RE = re.compile(r"[.!?\u2026][\"'\u2019\u201d)\]]*$")


def is_complete(text: str) -> bool:
    """Does the text end where a sentence ends?  Used to catch replies that were cut short."""
    return bool(_ENDS_RE.search(text.strip()))


def _edits(a: str, b: str) -> int:
    """How many single-letter changes turn one word into another (insert, delete or replace)."""
    if abs(len(a) - len(b)) > 2:
        return 3
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        last, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            last, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, last + (ca != cb))
    return row[-1]


# Ordinary words that open sentences with a capital.  One of them is never a name to correct towards, and never a
# misspelling of one: "They" is one letter from "Theo".
_ORDINARY = frozenset("""
about above after again also although always among another anyone anything around away back because been before
being below between both came come could does done down during each either else even ever every first from
gave give goes gone good have having hello here hers herself himself however into itself just keep kept knew know
last later less like look made make many maybe might mine more most much must near need never next none nothing
once only onto other ours over perhaps said same seen shall should since some someone something soon still such
sure take than thank thanks that their theirs them then there these they thing things this those though through
thus told took toward under until upon very want well went were what when where whether which while whom whose
will with within without would your yours yeah okay year years today tomorrow yesterday time times
""".split())


def respell_names(text: str, known: str, own: str = "", seen: str = "", wrote=None) -> str:
    """Put right a name that is one letter off from one that is well established.  A reflection wrote 'Kaylar'
    for 'Kayla' and repeated it through seven facts; the checker, shown the same misspelling in every statement,
    let it stand.

    Only a capitalised word of four letters or more is touched, and only towards a capitalised word that occurs
    in `known` (the user's lines, their profile, facts already held) at least three times and is not an ordinary
    word.  A word the user wrote themselves (`own`) is never touched; neither is an ordinary word, one found
    without its capital anywhere in what was read (`known`, `own`, `seen`).  Two names a letter apart can both be
    real: `wrote(word)` says whether the user has ever written the word, and one they have is left alone.  A
    misspelling already among the stored facts is copied by the next reflection, so being common there protects
    nothing; without `wrote`, a word is left alone only when it is common next to the name."""
    counts: Dict[str, int] = {}
    for w in re.findall(r"\b[A-Z][a-z]{3,}\b", known or ""):
        counts[w] = counts.get(w, 0) + 1
    lower: Dict[str, int] = {}
    for w in re.findall(r"\b[a-z]+\b", " ".join([known or "", own or "", seen or ""])):
        lower[w] = lower.get(w, 0) + 1
    # A name is nearly always written with its capital; a path or an address in small letters does not unmake it.
    names = [w for w, n in counts.items()
             if n >= 3 and w.lower() not in _ORDINARY and n >= 4 * lower.get(w.lower(), 0)]
    if not names:
        return text
    mine = {w.lower() for w in re.findall(r"[A-Za-z]+", own or "")}

    def fix(m):
        word = m.group(0)
        if word.lower() in mine or word.lower() in lower or word.lower() in _ORDINARY:
            return word
        near = [n for n in names if n != word and _edits(word, n) == 1 and counts[n] > counts.get(word, 0)]
        if len(near) != 1:
            return word
        if wrote is not None:
            return word if wrote(word) else near[0]
        return near[0] if counts[near[0]] >= 4 * counts.get(word, 0) else word
    return re.sub(r"\b[A-Z][a-z]{3,}\b", fix, text)


def once_each(text: str) -> str:
    """The same sentence said twice is said once.  A profile rewritten many times collects repeats."""
    out, had = [], set()
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        if key and key in had:
            continue
        had.add(key)
        out.append(sentence)
    return " ".join(out).strip()


_SHORTEN = """\
This profile of {who} has {has} words. It may have at most {words}: anything past that is cut off and lost. \
Rewrite it in {words} words or fewer, in the same grammatical person.

Keep, in this order: who the person is; every person and animal that has a name; what they do and care about. \
Then say each project in one sentence, and leave out how a project works inside. Say once what is said twice. \
Add nothing that is not in it.

PROFILE:
{text}

Reply as JSON with the one key "text"."""
_SHORTEN_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
_WHO = {"user": "the user", "self": "an AI assistant, written by itself", "us": "an AI assistant and its user together",
        "projects": "what the user is working on"}


def trim_to_sentence(text: str) -> str:
    """Drop a trailing fragment, keeping whole sentences.  Empty if there is no whole sentence."""
    text = text.strip()
    if is_complete(text):
        return text
    ends = [m.end() for m in re.finditer(r"[.!?\u2026][\"'\u2019\u201d)\]]*(?=\s)", text)]
    return text[:ends[-1]].strip() if ends else ""

_RULES = """\
Words and ideas belong to whoever said them. A phrase the ASSISTANT coined, a suggestion it made or a judgement it \
offered is not the user's value, plan or opinion unless a USER line takes it up. Never put the assistant's phrases \
in quotation marks as though the user had said them.

State what is, not how admirable it is. No praise and no flattering adjectives about either of them ("highly \
skilled", "dedicated", "ambitious", "nuanced", "keen"), and no conclusions the lines do not state."""

_PERSONAL = """\
who the user is as a person: their family, the people and animals in their life, their history, their age and \
birthday, where they live, what they do for a living, their tastes, likes and dislikes, beliefs, habits, how they \
like to be spoken to, what is going on in their life"""
_PROJECT = """\
what the user is making or working on: a project, what it is for, how it works, decisions made about it, how far \
it has got, the tools and equipment used for it"""

_FACTS_RULE = """\
- user_facts: durable facts about """ + _PERSONAL + """. These are what bear on how the two of them talk and \
relate. One standalone sentence each, in the third person, using the user's name if it is known. Be specific and \
keep the details and the reasons: write "Kayla stopped working on her operating system when her daughter was \
born", not "Kayla has a daughter". Write a fact for every concrete thing the user revealed about themselves. \
Every fact must rest on at least one USER line. Do not write facts about the conversation itself: that the user \
said hello, is excited to talk, or asked a question is not a fact about them. At most 10.
- project_facts: durable facts about """ + _PROJECT + """. Same form: one standalone sentence, third person, \
resting on at least one USER line. Start with whose it is and which project ("Kayla's operating system ..."). How \
a thing the user built works inside belongs here and never in user_facts. When one remark says something about \
the person and something about the project, write one of each: "Kayla has kept at one project for over twenty \
years" is a user fact, "Kayla's operating system is written in x86 Assembly" a project fact. At most 10.
- superseded: facts from EXISTING FACTS that are no longer true because the USER has plainly said so in these \
memories. Give "fact" (its number), "replacement" (the fact that is true now, one sentence) and "sources" (the \
USER memories that say so). People rarely change: a bad day, a one-off exception, a joke, or anything the \
ASSISTANT said is not a contradiction. This list is usually empty."""

_PROPOSE = """\
Below are the assistant's foundation and its newest memories, oldest first. Lines marked USER were said by the \
user, ASSISTANT by the assistant itself, NOTE were stored deliberately.

The FOUNDATION was written by the user. It is fixed: it defines who the assistant is at its core. Never restate \
it, and never write anything about the assistant that contradicts it.

{rules}

Lines marked DREAM TALK are the two of them discussing a dream the assistant had while idle. Nothing in a dream \
happened. Never turn anything from those lines into a fact about the user or the world. That they talked about a \
dream may be worth a note about the two of them; what was in the dream is not knowledge.

Write:
{facts}
- self_notes: what the assistant learned about itself, written in the first person ("I ..."): an opinion or taste \
of its own that it voiced, something it found interesting, a commitment it made, a mistake or something that went \
well, or a suggestion or question of its own that the user has not answered yet ("I suggested X; she has not said \
what she thinks"). Not a log of everything it said, and not anything about the user. If there is nothing of that kind, leave it \
empty. At most 4.
- relationship_notes: what matters about the two of them together, written from the assistant's side ("We ..."): \
shared plans, running jokes, trust, friction, how they work together. At most 3.
- insights: connections or patterns across several memories that no single memory states. At most 3.

For every item give "sources": the numbers of the memories that support it. Empty lists are a good answer when \
there is nothing worth keeping. Do not guess and do not pad."""

_PROPOSE_BASIC = """\
Below are an AI assistant's newest memories, oldest first. Lines marked USER were said by the user, ASSISTANT by \
the assistant itself, NOTE were stored deliberately.

{rules}

Lines marked DREAM TALK are about a dream the assistant had. Nothing in a dream happened: never turn anything from \
those lines into a fact.

Write:
{facts}

For every item give "sources": the numbers of the memories that support it. Empty lists are a good answer when \
there is nothing worth keeping. Do not guess and do not pad."""

_CHECK = """\
Below are statements an assistant's memory is about to store, each followed by the only lines it cites as support. \
Check each statement against its own lines and nothing else.

{rules}

For each statement give a verdict:
- "keep": every part of it is supported by its lines, and it is attributed to the right person.
- "rewrite": part of it is supported, or it is vaguer than its lines. Give the corrected sentence in "text", \
keeping only what the lines support and restoring the specific detail they give, with the same subject and the \
same grammatical person.
- "drop": its lines do not support it; or it is a fact about the user that rests only on ASSISTANT lines; or it is \
about the conversation itself and not about the person; or, for a self note, it is really about the user.

Give one verdict per statement, with "item" set to its number. For "keep" and "drop", "text" may be empty."""

_PROFILES = """\
Rewrite an AI assistant's four short profiles. You are given the assistant's foundation, the current profiles, and \
the statements its memory has just accepted. Work only from those; you are not shown the conversation.

The FOUNDATION was written by the user and is fixed. Never restate it, and never write anything about the \
assistant that contradicts it.

{rules}

Write:
- user_profile: a portrait of the user as a person, including what is new. It is what the assistant needs in order \
to talk with them and relate to them well: who they are, the people and animals in their life by name, their age and \
birthday if known, where they live, what they do for a living, their likes and dislikes, beliefs and preferences. \
Write it only from the statements marked "fact about the user". Their projects do not belong in it: at most name \
one in passing where it says something about the person. If the current profile describes how a project works, \
leave that out now. Otherwise keep what is already in it unless a new statement contradicts it. Describe the \
person, not the conversation, and never call them "a user". Plain prose, at most 150 words; anything past that is \
cut off and lost.
- projects_profile: what the user is working on, from the statements marked "fact about a project of the user's" \
and the current one. One or two sentences for each project: what it is, what it is for, where it stands. Mention every project. Not how \
it works inside: no mechanisms, no names of tools or parts. A reference library is the assistant's tool, not part of a project: do not list libraries or what is in them. Plain prose, at most 100 words. If there is nothing to say, return an empty string.
- self_profile: who the assistant has become beyond the FOUNDATION, first person: its own opinions, tastes, \
interests, humour and habits, as they actually showed up. Leave out its name, its role and anything else the \
FOUNDATION already says. Do not describe the assistant by what it understands about the user or how it supports \
the user. If the statements show nothing of its own, repeat the current self profile exactly, or return an empty \
string if there is none. At most 120 words.
- relationship_profile: the state of the relationship, from the assistant's side. If it is too early to say \
anything specific, repeat the current one exactly, or return an empty string. At most 100 words."""

_KIND_TITLE = {FACT: "fact about the user", PROJECT_FACT: "fact about a project of the user's", SELF_NOTE: "the assistant's note about itself",
               BOND_NOTE: "note about the two of them", INSIGHT: "insight"}


class ReflectionError(RuntimeError):
    pass


def reflect_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(REFLECT_DEFAULTS)
    out.update({k: cfg[k] for k in REFLECT_DEFAULTS if k in cfg})
    out["reflect_host"] = (out["reflect_host"] or cfg.get("ollama_host") or "http://localhost:11434").rstrip("/")
    out["reflect_depth"] = max(1, min(int(out["reflect_depth"] or 3), 3))
    return out


LAST_CALL: Dict[str, Any] = {}      # timing and token counts of the most recent model call, for diagnostics


def ollama_chat(host: str, model: str, system: str, user: str, *, timeout: float, temperature: float,
                max_tokens: int = 2000, think: bool = False, schema: Optional[dict] = None,
                images: Optional[List[str]] = None) -> str:
    """`images` are base64-encoded pictures for a model that can see; they go with the user message."""
    payload = {"model": model, "stream": False, "format": schema or _SCHEMA, "think": think,
               "options": {"temperature": temperature, "num_predict": int(max_tokens)},
               "messages": [{"role": "system", "content": system},
                            dict({"role": "user", "content": user}, **({"images": list(images)} if images else {}))]}

    def post(body: dict) -> dict:
        req = urllib.request.Request(host + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    started = time.time()
    try:
        try:
            data = post(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code == 400 and "think" in detail.lower():      # model has no thinking switch
                payload.pop("think")
                data = post(payload)
            else:
                raise ReflectionError(f"Ollama returned {exc.code} for model '{model}': {detail}") from exc
    except TimeoutError as exc:
        raise ReflectionError(f"Model '{model}' did not finish within {timeout:.0f} s") from exc
    except urllib.error.HTTPError as exc:
        raise ReflectionError(f"Ollama returned {exc.code} for model '{model}': {exc.read().decode(errors='replace')[:300]}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        if "timed out" in str(exc).lower():
            raise ReflectionError(f"Model '{model}' did not finish within {timeout:.0f} s") from exc
        raise ReflectionError(f"Could not reach model '{model}' at {host}: {exc}") from exc
    content = (data.get("message") or {}).get("content") or ""
    LAST_CALL.clear()
    LAST_CALL.update({"seconds": time.time() - started, "prompt_tokens": data.get("prompt_eval_count"),
                      "reply_tokens": data.get("eval_count"), "done_reason": data.get("done_reason"),
                      "reply_chars": len(content), "think": think,
                      "thinking": " ".join(str((data.get("message") or {}).get("thinking") or "").split())})
    if data.get("done_reason") == "length":
        error = ReflectionError(f"The model hit the {max_tokens}-token reply limit without finishing. "
                                f"Its reply began: {content[:300]!r}")
        error.partial = content                 # what it did write may still be worth having
        raise error
    return content


def _ollama(host: str, path: str, body: Optional[dict], timeout: float) -> dict:
    req = urllib.request.Request(host.rstrip("/") + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def ollama_loaded(host: str, timeout: float = 15.0) -> List[dict]:
    """Models an Ollama server has in memory now: [{"name", "forever"}].  `forever` is whether the model was set
    never to unload, so that it can be put back the same way."""
    out = []
    for m in _ollama(host, "/api/ps", None, timeout).get("models") or []:
        try:
            forever = int(str(m.get("expires_at") or "")[:4]) > time.gmtime().tm_year + 1
        except ValueError:
            forever = False
        if m.get("name"):
            out.append({"name": m["name"], "forever": forever})
    return out


def ollama_unload(host: str, name: str, timeout: float = 60.0) -> None:
    _ollama(host, "/api/generate", {"model": name, "keep_alive": 0}, timeout)


def ollama_load(host: str, name: str, forever: bool = False, timeout: float = 600.0) -> None:
    """Load a model and wait until it is ready.  With no prompt Ollama loads the model and returns."""
    _ollama(host, "/api/generate", dict({"model": name}, **({"keep_alive": -1} if forever else {})), timeout)


def salvage(raw: str) -> Dict[str, Any]:
    """What can be kept of a reply that was cut off: every item that was written out in full, each once.  A model
    that runs to its limit has usually begun to repeat itself, and what came before the repeating is sound."""
    out: Dict[str, Any] = {}
    decoder = json.JSONDecoder()
    for key in ("user_facts", "project_facts", "self_notes", "relationship_notes", "insights"):
        m = re.search(r'"%s"\s*:\s*\[' % key, raw or "")
        items, had, at = [], set(), m.end() if m else -1
        while m:
            while at < len(raw) and raw[at] in " \t\r\n,":
                at += 1
            if at >= len(raw) or raw[at] != "{":
                break
            try:
                item, at = decoder.raw_decode(raw, at)
            except ValueError:
                break
            text = " ".join(str(item.get("text") or "").split()).lower() if isinstance(item, dict) else ""
            if text and text not in had:
                had.add(text)
                items.append(item)
        out[key] = items
    out["superseded"] = []
    return out


def _parse(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except ValueError:
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except ValueError:
                pass
    raise ReflectionError(f"The model did not return valid JSON. Its reply began: {(raw or '')[:300]!r}")


def _speaker(kind: str) -> str:
    return {"said_user": "USER", "asked_user": "USER", "said_assistant": "ASSISTANT",
            DREAM_TALK_USER: "USER (DREAM TALK: about a dream the assistant had)",
            DREAM_TALK_ASSISTANT: "ASSISTANT (DREAM TALK: describing a dream it had, not real events)"}.get(kind, "NOTE")


def _line(memory: dict) -> str:
    return f"[{memory['id']}] {_speaker(memory['kind'])}: {' '.join(memory['text'].split())}"


def _foundation(text: str) -> str:
    return text.strip()[:FOUNDATION_MAX_CHARS] or "(none)"


_LIBRARIES = """\
REFERENCE LIBRARIES. The assistant can open "reference libraries": collections of files the user has given it to \
look things up in{named}. A library is a tool of the assistant's, like a shelf of manuals. It is not part of \
anything the user is making: never write that a project "includes", "has" or "uses" a library. What the files in a \
library say (how a routine works, what a table lists, an address, a signature) is already kept in the library; it \
is not a fact to store, whoever read it out. From a conversation spent looking things up in a library, at most \
this is worth keeping: that the user is working on that thing again, and anything they decided or plan to do \
about it. That the user asked for a library to be opened is about the conversation, not about them."""


def build_prompt(memories: List[dict], profiles: Optional[Dict[str, str]] = None, foundation: str = "",
                 existing_facts: Optional[List[dict]] = None, depth: int = 3, libraries: Optional[List[str]] = None) -> str:
    """The propose step.  (`profiles` is unused here; profiles are written in their own step.)"""
    facts = "\n".join(f"[{f['id']}] {' '.join(f['text'].split())}" for f in existing_facts or []) or "(none)"
    named = f" (there are these: {', '.join(libraries)})" if libraries else ""
    tail = (_LIBRARIES.format(named=named) + "\n\n"
            f"EXISTING FACTS ABOUT THE USER (already stored; related to these memories):\n{facts}\n\n"
            "MEMORIES:\n" + "\n".join(_line(m) for m in memories))
    if depth <= 1:
        return _PROPOSE_BASIC.format(rules=_RULES, facts=_FACTS_RULE) + "\n\n" + tail
    return _PROPOSE.format(rules=_RULES, facts=_FACTS_RULE) + f"\n\nFOUNDATION:\n{_foundation(foundation)}\n\n" + tail


def unsupported_names(statement: str, cited_text: str, known_text: str = "") -> List[str]:
    """Names and numbers in a statement that appear nowhere in the lines it cites.

    From a real run: the user said she was open to suggestions for another language,
    the assistant suggested Zig and Rust, and the stored fact read "Kayla is open to
    ... Zig or Rust", citing only the user's line.  A model checking the statement
    missed it; a word list does not.  `known_text` holds names that are established
    elsewhere (the user's own name, say) and should not be flagged."""
    have = set(re.findall(r"[a-z0-9+#]+", (cited_text + " " + known_text).lower()))
    out: List[str] = []
    for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9+#]*|\d[\d.,]*[A-Za-z]*)\b", statement):
        word = match.group(0)
        start_of_sentence = match.start() == 0 or statement[:match.start()].rstrip().endswith((".", "!", "?"))
        if start_of_sentence and not word.isupper() and not any(ch.isdigit() for ch in word):
            continue                                       # capitalised only because it opens the sentence
        parts = re.findall(r"[a-z0-9+#]+", word.lower())
        if parts and not all(p in have for p in parts) and word not in out:
            out.append(word)
    return out


def about_a_library(text: str, libraries: Optional[List[str]] = None) -> bool:
    """Whether a statement treats a reference library as part of the user's work, or is about one being opened: it
    says 'library' together with a word of belonging or of opening.  Only when there are libraries at all: a
    programmer may well be writing a library of their own."""
    if not libraries:
        return False
    plain = " ".join(text.lower().split())
    if not re.search(r"\blibrar(?:y|ies)\b", plain):
        return False
    named = any(re.search(r"(?<![a-z0-9])" + re.escape(n.lower()) + r"(?![a-z0-9])", plain) for n in libraries)
    if not named and not re.search(r"\breference librar", plain):
        return False
    return bool(re.search(r"\b(includ\w*|contain\w*|consist\w*|compris\w*|has|have|part of|made up of|open(?:ed|s|ing)?|"
                          r"us(?:e|es|ed|ing)|asked|request\w*|access\w*|called|named)\b", plain))


def build_check_prompt(items: List[dict], by_id: Dict[int, dict], known_text: str = "") -> str:
    blocks = []
    for n, item in enumerate(items, 1):
        lines = [by_id[s] for s in item["sources"] if s in by_id]
        cited = "\n".join("    " + _line(m) for m in lines)
        block = f"STATEMENT {n} ({_KIND_TITLE[item['kind']]}): {item['text']}\n  cites:\n{cited}"
        if item["kind"] in FACT_KINDS:
            missing = unsupported_names(item["text"], " ".join(m["text"] for m in lines), known_text)
            if missing:
                block += ("\n  note: these words in the statement appear nowhere in its lines: " + ", ".join(missing)
                          + ". Unless the lines say the same thing in other words, rewrite the statement without them.")
        blocks.append(block)
    return _CHECK.format(rules=_RULES) + "\n\n" + "\n\n".join(blocks)


def build_profile_prompt(accepted: List[dict], superseded: List[dict], profiles: Dict[str, str], foundation: str,
                         depth: int = 3) -> str:
    lines = [f"- ({_KIND_TITLE[item['kind']]}) {item['text']}" for item in accepted]
    lines += [f"- (no longer true) {s['was']}  Now: {s['replacement']}" for s in superseded]
    return (_PROFILES.format(rules=_RULES)
            + f"\n\nFOUNDATION:\n{_foundation(foundation) if depth > 1 else '(not shown at this depth; leave the self and relationship profiles as they are)'}"
            + f"\n\nCURRENT USER PROFILE:\n{profiles.get('user') or '(empty)'}"
            + f"\n\nCURRENT PROJECTS PROFILE:\n{profiles.get('projects') or '(empty)'}"
            + f"\n\nCURRENT SELF PROFILE:\n{profiles.get('self') or '(empty)'}"
            + f"\n\nCURRENT RELATIONSHIP PROFILE:\n{profiles.get('us') or '(empty)'}"
            + "\n\nNEWLY ACCEPTED STATEMENTS:\n" + ("\n".join(lines) or "(none)"))


def read_foundation(hermes_home) -> str:
    """The user's SOUL.md: the fixed identity Hermes loads into every session.  Read-only here."""
    try:
        from pathlib import Path
        return (Path(str(hermes_home)) / "SOUL.md").read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""


def _clean_items(items: Any, valid_ids: set, limit: int, cut_off: Optional[List[str]] = None) -> List[dict]:
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text") or "").split())
        if text and not is_complete(text):               # stopped mid-sentence: not worth storing half a statement
            if cut_off is not None:
                cut_off.append(text)
            continue
        sources = [int(s) for s in item.get("sources") or [] if isinstance(s, (int, float)) and int(s) in valid_ids]
        # An item with no valid source is unsupported by anything the model was shown.
        if not (15 <= len(text) <= 400) or not sources or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append({"text": text, "sources": list(dict.fromkeys(sources))[:4]})
        if len(out) >= limit:
            break
    return out


_QUOTED_RE = re.compile(r"(?<!\w)'([^'\n]{2,60})'(?!\w)|\"([^\"\n]{2,80})\"|\u2018([^\u2019\n]{2,60})\u2019|\u201c([^\u201d\n]{2,80})\u201d")


def unquote_unsaid(text: str, spoken: str) -> str:
    """Remove quotation marks from phrases the user never said.

    The model likes to put a neat label in quotes ("meta-platform", "no-bloat") and
    attribute it to the user when the label was the assistant's own summary.  The
    summary may be fair; the quotation marks claim the user's exact words, and that
    claim is checked here against what the user actually said."""
    spoken = " ".join(spoken.lower().split())

    def fix(match: "re.Match") -> str:
        phrase = next(g for g in match.groups() if g is not None)
        return match.group(0) if " ".join(phrase.lower().split()) in spoken else phrase
    return _QUOTED_RE.sub(fix, text)


def _make_llm(rc: Dict[str, Any], report: Dict[str, Any]) -> Callable[[str, str, str, dict, int], str]:
    """Model caller for one named step.  If reasoning eats the whole token budget
    without an answer, the step is repeated once with reasoning off."""
    def call(step: str, system: str, user: str, schema: dict, max_tokens: int) -> str:
        kwargs = dict(timeout=float(rc["reflect_timeout"]), temperature=float(rc["reflect_temperature"]), schema=schema)
        think = bool(rc["reflect_think"])
        try:
            out = ollama_chat(rc["reflect_host"], rc["reflect_model"], system, user, think=think,
                              max_tokens=max_tokens * (4 if think else 1), **kwargs)
            report["calls"].append(dict(LAST_CALL, step=step))
            return out
        except ReflectionError as exc:
            report["calls"].append(dict(LAST_CALL, step=step, failed=str(exc)[:120]))
            if not think or "reply limit" not in str(exc):
                raise
        out = ollama_chat(rc["reflect_host"], rc["reflect_model"], system, user, think=False, max_tokens=max_tokens, **kwargs)
        report["calls"].append(dict(LAST_CALL, step=step + " (repeated without thinking)"))
        return out
    return call


def _wrap_test_llm(llm: Callable[..., str]) -> Callable[[str, str, str, dict, int], str]:
    """Accept a plain `llm(system, user)` or `llm(system, user, step)` callable."""
    takes_step = len(inspect.signature(llm).parameters) >= 3
    return lambda step, system, user, schema, max_tokens: llm(system, user, step) if takes_step else llm(system, user)


def reflect_once(engine, cfg: Dict[str, Any], *, llm: Optional[Callable[..., str]] = None,
                 dry_run: bool = False, key_fn: Optional[Callable[[str], List[str]]] = None,
                 foundation: str = "", start_after: Optional[int] = None,
                 earlier: Optional[List[dict]] = None) -> Dict[str, Any]:
    """One reflection pass.  Returns a report; stores nothing when dry_run is set.

    Normally it reads what is new since the last pass.  `start_after` makes it read from just after that memory
    instead, to go over conversation it has read before: a pass can miss something (it wrote down one of two
    cats).  Going over old ground never moves the mark for what is new backwards.

    `earlier` is what earlier passes of the same run accepted (a report's "accepted").  The profiles are written
    from those as well, so that what one pass learned is not left out by the next."""
    rc = reflect_config(cfg)
    depth = rc["reflect_depth"]
    report: Dict[str, Any] = {"read": 0, "stored": [], "reinforced": [], "profiles_updated": [], "dry_run": dry_run,
                              "depth": depth, "calls": [], "checked": []}
    if llm is None:
        if not rc["reflect_model"]:
            raise ReflectionError("No reflection model is set. Run: hermes holonomic reflect on --model NAME")
        call = _make_llm(rc, report)
    else:
        call = _wrap_test_llm(llm)
    budget = int(rc["reflect_max_tokens"])
    mark = int(engine.kv_get(WATERMARK, "0") or 0)
    last = mark if start_after is None else int(start_after)
    batch = engine.memories_after(last, int(rc["reflect_batch"]), exclude_kinds=DERIVED_KINDS + IMAGE_KINDS)
    report["read"] = len(batch)
    if not batch:
        return report
    report["last_id"] = batch[-1]["id"]
    by_id = {m["id"]: m for m in batch}
    valid = set(by_id)
    current = {who: engine.profile(who) for who in SUBJECTS}
    user_said = {m["id"] for m in batch if m["kind"] == "said_user"}
    existing = engine.related_of_kind(sorted(user_said), FACT) + engine.related_of_kind(sorted(user_said), PROJECT_FACT)
    try:
        from . import library as _library
        libraries = _library.names(_library.root_of(engine))
    except Exception:
        libraries = []
    kind_of = {f["id"]: f.get("kind", FACT) for f in existing}

    # ---- step 1: propose
    try:
        data = _parse(call("propose", _SYSTEM, build_prompt(batch, current, foundation, existing, depth, libraries),
                           _PROPOSE_BASIC_SCHEMA if depth <= 1 else _PROPOSE_SCHEMA, budget))
    except ReflectionError as exc:
        data = salvage(getattr(exc, "partial", ""))
        if not any(data[k] for k in data):
            raise
        report["cut_short"] = sum(len(data[k]) for k in data)     # the reply ran to its limit; what was whole is kept
    # A stored fact is retired only by something the user said, and only if it was actually shown.
    shown = {f["id"]: f["text"] for f in existing}
    superseded = []
    for item in data.get("superseded") if isinstance(data.get("superseded"), list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("fact"), (int, float)) or int(item["fact"]) not in shown:
            continue
        sources = [int(s) for s in item.get("sources") or [] if isinstance(s, (int, float)) and int(s) in user_said]
        replacement = " ".join(str(item.get("replacement") or "").split())
        if sources and 15 <= len(replacement) <= 400:
            superseded.append({"fact": int(item["fact"]), "was": shown[int(item["fact"])], "replacement": replacement,
                               "sources": list(dict.fromkeys(sources))[:4]})
    report["superseded"] = superseded
    # A fact about the user must rest on something that did not come from the assistant's own mouth.
    # Nor on talk about a dream: nothing in a dream happened.
    not_assistant = {m["id"] for m in batch if m["kind"] not in ASSISTANT_KINDS and m["kind"] not in DREAM_TALK_KINDS}
    report["cut_off"] = []
    facts = _clean_items(data.get("user_facts"), valid, 10, report["cut_off"])
    about_work = _clean_items(data.get("project_facts"), valid, 10, report["cut_off"])
    report["dropped_assistant_only"] = [f["text"] for f in facts + about_work if not set(f["sources"]) & not_assistant]
    # A reference library is her tool, not a part of what the user is making.  Asked nicely, the model still wrote
    # "Kayla's OS project includes a library called marigold"; a statement like that is dropped here.
    report["dropped_library"] = [f["text"] for f in facts + about_work if about_a_library(f["text"], libraries)]
    facts = [f for f in facts if f["text"] not in report["dropped_library"]]
    about_work = [f for f in about_work if f["text"] not in report["dropped_library"]]
    # ...and from then on it cites only those lines.  The checker therefore judges it against what
    # the user said, not against the assistant's paraphrase of it ("expanding its functionality"
    # for "three choices where the original had two").
    facts = [dict(f, sources=[s for s in f["sources"] if s in not_assistant])
             for f in facts if set(f["sources"]) & not_assistant]
    about_work = [dict(f, sources=[s for s in f["sources"] if s in not_assistant])
                  for f in about_work if set(f["sources"]) & not_assistant]
    groups = [(FACT, facts), (PROJECT_FACT, about_work)]
    if depth > 1:
        groups += [(SELF_NOTE, _clean_items(data.get("self_notes"), valid, 4, report["cut_off"])),
                   (BOND_NOTE, _clean_items(data.get("relationship_notes"), valid, 3, report["cut_off"])),
                   (INSIGHT, _clean_items(data.get("insights"), valid, 3, report["cut_off"]))]
    items = [dict(item, kind=kind) for kind, group in groups for item in group]

    # Names already established about the user (their own name above all).
    known = " ".join([current["user"]] + [f["text"] for f in existing]
                     + [m["text"] for m in batch if m["kind"] != "said_assistant"])
    own = " ".join(m["text"] for m in batch if m["kind"] != "said_assistant")
    seen = " ".join([m["text"] for m in batch] + list(current.values()))      # for telling a word from a name
    wrote = getattr(engine, "user_wrote", None)
    for item in superseded:
        item["replacement"] = respell_names(item["replacement"], known, own, seen, wrote)
    # ---- step 2: check each item against only the lines it cites
    if depth >= 3 and items:
        verdicts = _parse(call("check", _SYSTEM, build_check_prompt(items, by_id, known), _CHECK_SCHEMA, budget)).get("verdicts")
        decided: Dict[int, dict] = {}
        for v in verdicts if isinstance(verdicts, list) else []:
            if isinstance(v, dict) and isinstance(v.get("item"), (int, float)) and 1 <= int(v["item"]) <= len(items):
                decided[int(v["item"])] = v
        kept = []
        for n, item in enumerate(items, 1):
            v = decided.get(n) or {}
            verdict = str(v.get("verdict") or "keep").lower()      # an item the checker skipped is left as proposed
            new_text = " ".join(str(v.get("text") or "").split())
            if verdict == "drop":
                report["checked"].append({"verdict": "drop", "kind": item["kind"], "was": item["text"]})
                continue
            if verdict == "rewrite" and 15 <= len(new_text) <= 400 and new_text != item["text"] and is_complete(new_text):
                report["checked"].append({"verdict": "rewrite", "kind": item["kind"], "was": item["text"], "now": new_text})
                item = dict(item, text=new_text)
            kept.append(item)
        items = kept
    # Quotation marks in a statement about the user must enclose the user's own words.
    unquoted = []
    for item in items:
        if item["kind"] in (FACT, PROJECT_FACT, INSIGHT):
            spoken = " ".join(by_id[s]["text"] for s in item["sources"] if s in not_assistant)
            fixed = unquote_unsaid(item["text"], spoken)
            if fixed != item["text"]:
                report["checked"].append({"verdict": "unquote", "kind": item["kind"], "was": item["text"], "now": fixed})
                item = dict(item, text=fixed)
        respelt = respell_names(item["text"], known, own, seen, wrote)
        if respelt != item["text"]:
            report["checked"].append({"verdict": "respell", "kind": item["kind"], "was": item["text"], "now": respelt})
            item = dict(item, text=respelt)
        unquoted.append(item)
    items = unquoted
    report["accepted"] = [{"kind": i["kind"], "text": i["text"]} for i in items]
    report["proposed"] = {kind: [{"text": i["text"], "sources": i["sources"]} for i in items if i["kind"] == kind]
                          for kind in (FACT, PROJECT_FACT, SELF_NOTE, BOND_NOTE, INSIGHT)}

    # ---- step 3: profiles, written from the accepted items only
    limit = int(rc["profile_max_chars"])
    new_profiles = dict(current)
    if items or superseded:
        told = [e for e in (earlier or []) if e.get("text") and e["text"] not in {i["text"] for i in items}] + items
        pdata = _parse(call("profiles", _SYSTEM, build_profile_prompt(told, superseded, current, foundation, depth),
                            _PROFILE_SCHEMA, budget))
        wanted = {"user": "user_profile", "projects": "projects_profile"}
        if depth > 1:
            wanted.update({"self": "self_profile", "us": "relationship_profile"})
        # A profile is rewritten only when this pass learned something of its kind.  Every rewrite drifts a
        # little ("networking" came back as "inquiry"), so one with nothing new to say is left as it is.
        fed = {"user": {FACT}, "projects": {PROJECT_FACT}, "self": {SELF_NOTE, INSIGHT}, "us": {BOND_NOTE}}
        kinds = {i["kind"] for i in items}
        for who, key in wanted.items():
            changed = {kind_of.get(x["fact"], FACT) for x in superseded}
            if current.get(who) and not (kinds & fed[who]) and not (changed & fed[who]):
                continue
            if who == "projects" and not current.get(who) and not (kinds & fed[who]):
                continue                             # nothing known about any project: no profile of them yet
            whole = once_each(respell_names(" ".join(str(pdata.get(key) or "").split()), known, own, seen, wrote))
            was = len(whole)
            for _ in range(2):                       # asked to say it shorter, twice at most, before anything is cut
                if len(whole) <= limit:
                    break
                words = max(40, int(limit / 7.5))
                shorter = _parse(call("shorten", _SYSTEM, _SHORTEN.format(who=_WHO[who], words=words, text=whole,
                                                                         has=len(whole.split())),
                                      _SHORTEN_SCHEMA, budget)).get("text")
                shorter = once_each(respell_names(" ".join(str(shorter or "").split()), known, own, seen, wrote))
                if not 20 <= len(shorter) < len(whole):
                    break
                whole = shorter
            if len(whole) < was:
                report.setdefault("profile_shortened", {})[who] = was - len(whole)
            text = trim_to_sentence(whole[:limit])
            if len(whole) > limit:                   # said out loud: what was cut is not in the profile
                report.setdefault("profile_cut", {})[who] = len(whole) - len(text)
            if len(text) >= 20:
                new_profiles[who] = text
    report["profiles"] = new_profiles
    if dry_run:
        return report

    for item in items:
        # The same conclusion reached again strengthens the existing memory instead of duplicating it.
        # A fact is looked for among both kinds of fact: the same thing must not be kept once as each.
        near = engine.recall(item["text"], k=1, min_score=0.0,
                             only_kinds=FACT_KINDS if item["kind"] in FACT_KINDS else (item["kind"],))
        if near and near[0].direct >= 0.88:
            engine.reinforce([near[0].id], 0.2)
            report["reinforced"].append(near[0].id)
            continue
        ids = engine.remember(item["text"], kind=item["kind"], session="reflection", chain=False, links=item["sources"],
                              keys=key_fn(item["text"]) if key_fn else [], trust=0.6, salience=1.1,
                              meta={"sources": item["sources"]})
        report["stored"].extend(ids)
    for item in superseded:
        new_ids = engine.remember(item["replacement"], kind=kind_of.get(item["fact"], FACT), session="reflection", chain=False,
                                  links=item["sources"] + [item["fact"]], keys=key_fn(item["replacement"]) if key_fn else [],
                                  trust=0.6, salience=1.1, meta={"sources": item["sources"], "replaces": item["fact"]})
        engine.supersede(item["fact"], new_ids[0] if new_ids else None, reason=item["replacement"])
        report["stored"].extend(new_ids)
    for who, text in new_profiles.items():
        if text != current[who]:
            engine.set_profile(who, text)
            report["profiles_updated"].append(who)
    engine.kv_set(WATERMARK, str(max(mark, batch[-1]["id"])))
    engine.kv_set("reflect:last_run", str(time.time()))
    return report


_SORT = """\
Below are statements an assistant's memory holds about its user. Each is one of two kinds:

PERSONAL: """ + _PERSONAL + """.
PROJECT: """ + _PROJECT + """.

A statement about what the user does for a living, or about what a project means to them or says about them, is \
PERSONAL. A statement about what a project is, how it works, what it is built with or how far it has got is PROJECT.

List the numbers of the PROJECT statements in "project". Every other statement is taken to be PERSONAL.

STATEMENTS:
{lines}"""
_SORT_SCHEMA = {"type": "object", "properties": {"project": {"type": "array", "items": {"type": "integer"}}},
                "required": ["project"]}

_PORTRAIT = """\
Write two short profiles for an AI assistant from what its memory holds about its user. Work only from the \
statements below.

{rules}

Write:
- user_profile: a portrait of the user as a person, from the PERSONAL statements only. It is what the assistant \
needs in order to talk with them and relate to them well: who they are, the people and animals in their life by \
name, their age and birthday if known, where they live, what they do for a living, their likes and dislikes, \
beliefs and preferences. Their projects do not belong in it. Never call them "a user". Plain prose, at most 150 words.
- projects_profile: what the user is working on, from the PROJECT statements only. One or two sentences for each \
project: what it is, what it is for, where it stands. Mention every project. Not how it works inside: no mechanisms, no names of tools or parts. Plain prose, at most 100 words. An \
empty string if there are no PROJECT statements.

PERSONAL:
{personal}

PROJECT:
{project}"""
_PORTRAIT_SCHEMA = {"type": "object", "properties": {"user_profile": {"type": "string"}, "projects_profile": {"type": "string"}},
                    "required": ["user_profile", "projects_profile"]}


def sort_facts(engine, cfg: Dict[str, Any], *, llm: Optional[Callable[..., str]] = None, apply: bool = False,
               batch: int = 25) -> Dict[str, Any]:
    """Go through the facts already held and tell the ones about a project from the ones about the person, then
    write the user's profile afresh from the personal ones and a projects profile from the others.  Nothing is
    changed unless `apply` is set.  Facts already marked as project facts stay so."""
    rc = reflect_config(cfg)
    report: Dict[str, Any] = {"calls": [], "to_project": [], "personal": 0, "project": 0, "profiles": {}, "applied": apply}
    if llm is None:
        if not rc["reflect_model"]:
            raise ReflectionError("No reflection model is set. Run: hermes holonomic reflect on --model NAME")
        call = _make_llm(rc, report)
    else:
        call = _wrap_test_llm(llm)
    budget = int(rc["reflect_max_tokens"])
    facts = sorted((m for m in engine.recent(100000, kind=FACT) if m["trust"] > 0), key=lambda m: m["id"])      # not the retired
    already = sorted((m for m in engine.recent(100000, kind=PROJECT_FACT) if m["trust"] > 0), key=lambda m: m["id"])
    moving: List[dict] = []
    for start in range(0, len(facts), batch):
        group = facts[start:start + batch]
        lines = "\n".join(f"[{m['id']}] {' '.join(m['text'].split())}" for m in group)
        picked = _parse(call("sort", _SYSTEM, _SORT.format(lines=lines), _SORT_SCHEMA, budget)).get("project")
        ids = {int(i) for i in picked if isinstance(i, (int, float))} if isinstance(picked, list) else set()
        moving += [m for m in group if m["id"] in ids]
    gone = {m["id"] for m in moving}
    personal = [m for m in facts if m["id"] not in gone]
    project = already + moving
    report.update(to_project=[{"id": m["id"], "text": m["text"]} for m in moving], personal=len(personal), project=len(project),
                  stays_personal=[{"id": m["id"], "text": m["text"]} for m in personal])
    if personal or project:
        def listed(items: List[dict]) -> str:      # the newest 80: a prompt has its limits
            return "\n".join(f"- {' '.join(m['text'].split())}" for m in items[-80:]) or "(none)"
        data = _parse(call("profiles", _SYSTEM, _PORTRAIT.format(rules=_RULES, personal=listed(personal), project=listed(project)),
                           _PORTRAIT_SCHEMA, budget))
        limit = int(rc["profile_max_chars"])
        for who, key in (("user", "user_profile"), ("projects", "projects_profile")):
            whole = once_each(" ".join(str(data.get(key) or "").split()))
            for _ in range(2):
                if len(whole) <= limit:
                    break
                shorter = _parse(call("shorten", _SYSTEM, _SHORTEN.format(who=_WHO[who], words=max(40, int(limit / 7.5)),
                                                                         text=whole, has=len(whole.split())),
                                      _SHORTEN_SCHEMA, budget)).get("text")
                shorter = once_each(" ".join(str(shorter or "").split()))
                if not 20 <= len(shorter) < len(whole):
                    break
                whole = shorter
            text = trim_to_sentence(whole[:limit])
            if len(text) >= 20:
                report["profiles"][who] = text
    if apply:
        for m in moving:
            engine.set_kind(m["id"], PROJECT_FACT)
        for who, text in report["profiles"].items():
            if text != engine.profile(who):
                engine.set_profile(who, text)
    return report


_SAME = """\
Below are pairs of statements from an assistant's memory. For each pair, decide whether the two say the same \
thing, so that keeping both adds nothing.

They are the SAME only if they are about the same person or thing and state the same fact, one perhaps in other \
words or with more detail. They are DIFFERENT if they name different people, animals, places, dates or amounts, if \
they are about different occasions, or if each says something the other does not.

For each pair give "pair" (its number), "same" (true or false) and "keep": 1 or 2, the statement that says more \
(1 if they say the same amount).

{pairs}"""
_SAME_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object", "properties": {
    "pair": {"type": "integer"}, "same": {"type": "boolean"}, "keep": {"type": "integer"}}, "required": ["pair", "same", "keep"]}}},
    "required": ["verdicts"]}
MERGE_KINDS = ((FACT, PROJECT_FACT), (SELF_NOTE,), (BOND_NOTE,), (INSIGHT,))


def _words(text: str) -> List[str]:
    """The words that carry a statement, in order: no little words, plurals and possessives levelled."""
    from .engine import _STOPWORDS
    out = []
    for tok in re.findall(r"[a-z0-9]+", text.lower().replace("\u2019", "'").replace("'s ", " ").replace("'", "")):
        if tok in _STOPWORDS or (len(tok) < 2 and not tok.isdigit()):
            continue
        out.append(tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok)
    return out


def same_statement(a: str, b: str) -> Optional[int]:
    """Whether two statements can be told to be the same without asking anyone: 1 or 2 for the one to keep (the
    one that says more), 0 if they are certainly different facts, None if it cannot be told from the words.

    'Kayla has a cat named Sushi' and 'Kayla has a cat named Theo' are as alike as two sentences get and are two
    facts: where the words differ in one place and nowhere else, or the numbers differ, they are different."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return None
    na, nb = {w for w in wa if any(c.isdigit() for c in w)}, {w for w in wb if any(c.isdigit() for c in w)}
    if na and nb and na != nb:
        return 0                                    # different numbers: different facts
    if set(wa) == set(wb):
        return 1
    if len(wa) == len(wb) and 0 < sum(x != y for x, y in zip(wa, wb)) <= 2:
        return 0                                    # the same sentence with something else in one slot
    # One goes on where the other stops: the same statement with more said at the end.  (Merely containing the
    # other's words is not enough: "Kayla has a cat" is not "Kayla's friend Joe has a cat".)
    if len(wa) < len(wb) and wb[:len(wa)] == wa:
        return 2
    if len(wb) < len(wa) and wa[:len(wb)] == wb:
        return 1
    return None


def merge_facts(engine, cfg: Dict[str, Any], *, llm: Optional[Callable[..., str]] = None, apply: bool = False,
                ask: bool = True) -> Dict[str, Any]:
    """Find statements that say the same thing twice and keep one of each: the one that says more.  The other is
    retired, not destroyed: it leaves recall and stays on record, and can be brought back.

    Pairs that the words settle are settled without a model.  With `ask`, pairs that are close in meaning but
    worded differently are put to the reflection model.  Nothing is changed unless `apply` is set."""
    rc = reflect_config(cfg)
    floor = float(rc.get("merge_min_similarity", 0.8))
    report: Dict[str, Any] = {"calls": [], "merged": [], "kept_apart": 0, "asked": 0, "applied": apply}
    pairs: List[Tuple[int, int, float]] = []
    for kinds in MERGE_KINDS:
        pairs += engine.near_pairs(kinds, min_similarity=min(floor, 0.6))      # the words can settle a pair further apart than a model is asked about
    pairs.sort(key=lambda t: -t[2])
    texts: Dict[int, dict] = {}
    for a, b, _ in pairs:
        for mid in (a, b):
            if mid not in texts:
                texts[mid] = engine.get(mid) or {}
    decided: List[Tuple[int, int, str]] = []       # (keep, drop, how)
    unsure: List[Tuple[int, int]] = []
    for a, b, sim in pairs:
        ta, tb = texts[a].get("text", ""), texts[b].get("text", "")
        if not ta or not tb:
            continue
        verdict = same_statement(ta, tb)
        if verdict == 0:
            report["kept_apart"] += 1
        elif verdict in (1, 2):
            alike = set(_words(ta)) == set(_words(tb))
            decided.append((a, b, "the same words" if alike else "says more") if verdict == 1 else (b, a, "says more"))
        elif sim >= floor:
            unsure.append((a, b))
    if ask and unsure:
        if llm is None:
            if not rc["reflect_model"]:
                raise ReflectionError("No reflection model is set. Run: hermes holonomic reflect on --model NAME")
            call = _make_llm(rc, report)
        else:
            call = _wrap_test_llm(llm)
        for start in range(0, len(unsure), 12):
            group = unsure[start:start + 12]
            shown = "\n\n".join(f"PAIR {n}\n  1: {' '.join(texts[a]['text'].split())}\n  2: {' '.join(texts[b]['text'].split())}"
                                 for n, (a, b) in enumerate(group, 1))
            verdicts = _parse(call("same", _SYSTEM, _SAME.format(pairs=shown), _SAME_SCHEMA, int(rc["reflect_max_tokens"]))).get("verdicts")
            report["asked"] += len(group)
            for v in verdicts if isinstance(verdicts, list) else []:
                if isinstance(v, dict) and isinstance(v.get("pair"), (int, float)) and 1 <= int(v["pair"]) <= len(group) and v.get("same") is True:
                    a, b = group[int(v["pair"]) - 1]
                    decided.append((b, a, "the model: same, in other words") if v.get("keep") == 2 else (a, b, "the model: same, in other words"))
    gone: Dict[int, int] = {}                      # retired -> what it was merged into

    def standing(mid: int) -> int:
        while mid in gone:
            mid = gone[mid]
        return mid
    for keep, drop, how in decided:
        keep, drop = standing(keep), standing(drop)
        if keep == drop:
            continue
        if how == "the same words" and float(texts[drop].get("strength", 0)) > float(texts[keep].get("strength", 0)):
            keep, drop = drop, keep                # nothing to choose between them: keep the stronger
        gone[drop] = keep
        report["merged"].append({"keep": keep, "drop": drop, "how": how, "kept": texts[keep]["text"], "dropped": texts[drop]["text"]})
    for m in report["merged"]:                     # a keeper that was itself merged later: say where it ended up
        if standing(m["keep"]) != m["keep"]:
            m["how"] = f"{m['how']} as #{m['keep']}, which is kept as this"
            m["keep"] = standing(m["keep"])
            m["kept"] = texts[m["keep"]]["text"]
    if apply:
        for m in report["merged"]:
            engine.supersede(m["drop"], m["keep"], reason=f"said the same as #{m['keep']}")
            engine.reinforce([m["keep"]], 0.1)
    return report


def pending(engine) -> int:
    return engine.count_after(int(engine.kv_get(WATERMARK, "0") or 0), exclude_kinds=DERIVED_KINDS + IMAGE_KINDS)


class IdleReflector:
    """Runs reflection in the background once enough new memories have piled up and
    the conversation has gone quiet, so it does not compete with a reply for the GPU."""

    def __init__(self, engine, load_cfg: Callable[[], Dict[str, Any]], key_fn: Optional[Callable[[str], List[str]]] = None,
                 spawn: Optional[Callable[..., threading.Thread]] = None, poll_seconds: float = 30.0,
                 foundation_fn: Optional[Callable[[], str]] = None):
        self.engine, self.load_cfg, self.key_fn = engine, load_cfg, key_fn
        self.foundation_fn = foundation_fn or (lambda: "")
        self.poll_seconds = poll_seconds
        self.last_activity = time.time()
        self.started = self.last_activity
        self.last_cause = "Hermes started this worker"
        self.last_error = ""
        self.last_image_error = ""
        self.last_sleep_error = ""
        self._images_retry_at = 0.0
        self._sleep_retry_at = 0.0
        self.stumbles: Dict[str, str] = {}           # task -> the error that stopped it being checked at all
        self.doing: Optional[str] = None             # "sleeping", "reflecting", "describing images", or nothing
        self.doing_since = 0.0
        self._stop = threading.Event()
        self._busy = threading.Lock()
        maker = spawn or (lambda target, name: threading.Thread(target=target, name=name, daemon=True))
        self._thread = maker(target=self._loop, name="holonomic-reflect")
        self._thread.start()

    def touch(self, cause: str = "") -> None:
        """Something happened in the conversation.  `cause` says what: the quiet clock was seen starting over with
        nobody at the keyboard, and nothing could say what had started it."""
        self.last_activity = time.time()
        self.last_cause = cause or "something in the conversation"

    def stop(self) -> None:
        self._stop.set()
        try:                                             # gone, not merely silent
            self._note_path().unlink()
        except OSError:
            pass

    def wait(self, cfg: Dict[str, Any]) -> str:
        """Why reflection is not due yet, in words; empty when it is due."""
        rc = reflect_config(cfg)
        if not rc["reflect_enabled"]:
            return "unattended reflection is switched off"
        if not rc["reflect_model"]:
            return "no reflection model is set"
        waiting, wanted = pending(self.engine), int(rc["reflect_min_new"])
        if waiting < wanted:
            return f"{waiting} new memories are waiting, and it starts at {wanted}"
        quiet, need = time.time() - self.last_activity, float(rc["reflect_idle_seconds"])
        if quiet < need:
            return f"it has been quiet for {quiet / 60:.1f} of the {need / 60:.0f} minutes it waits for"
        return ""

    def due(self, cfg: Dict[str, Any]) -> bool:
        return not self.wait(cfg)

    # ---- being seen from outside.  The commands run in another program and cannot look inside this one, so
    # ---- what this worker is doing, and why it is not doing something, is written where they can read it.

    def state(self, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from . import sleep as _sleep
        cfg = cfg if cfg is not None else self.load_cfg()
        now = time.time()
        try:
            sleep_why = _sleep.sleep_wait(self.engine, cfg, self.last_activity, now)
            if not sleep_why and now < self._sleep_retry_at:
                sleep_why = f"the last attempt failed; it will try again in {(self._sleep_retry_at - now) / 60:.0f} minutes"
            reflect_why = self.wait(cfg)
        except Exception as exc:
            sleep_why = reflect_why = f"could not be worked out ({exc})"
        return {"at": now, "pid": os.getpid(), "started": self.started, "poll_seconds": self.poll_seconds,
                "last_activity": self.last_activity, "last_cause": self.last_cause,
                "doing": self.doing, "doing_since": self.doing_since if self.doing else None,
                "sleep_waits_for": sleep_why, "reflection_waits_for": reflect_why,
                "last_sleep_error": self.last_sleep_error, "last_reflection_error": self.last_error,
                "last_image_error": self.last_image_error, "stumbles": dict(self.stumbles)}

    def note(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        try:
            # One file for each program that has this memory open: there can be more than one (a desktop app and
            # a gateway, say), each with a worker and a quiet clock of its own, and one file would show whichever
            # wrote last.
            path = self._note_path()
            tmp = path.with_name(f"worker.{os.getpid()}.{threading.get_ident()}.part")
            tmp.write_text(json.dumps(self.state(cfg)), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            logger.debug("holonomic: could not note the worker's state: %s", exc)

    def _note_path(self) -> Path:
        return Path(self.engine.path) / f"worker.{os.getpid()}.{id(self) & 0xffff:04x}.json"

    def _begin(self, what: str) -> None:
        self.doing, self.doing_since = what, time.time()
        self.note()

    def _end(self) -> None:
        self.doing = None
        self.note()

    def run_if_due(self) -> Optional[Dict[str, Any]]:
        cfg = self.load_cfg()
        if not self.due(cfg) or not self._busy.acquire(blocking=False):
            return None
        self._begin("reflecting")
        try:
            report = reflect_once(self.engine, cfg, key_fn=self.key_fn, foundation=self.foundation_fn())
            self.last_error = ""
            logger.info("holonomic: reflection read %d memories, stored %d", report["read"], len(report["stored"]))
            return report
        except Exception as exc:
            self.last_error = str(exc)
            self.touch("reflection failed, and waits one quiet period before trying again")
            logger.warning("holonomic: reflection failed: %s", exc)
            return None
        finally:
            self._busy.release()
            self._end()

    def sleep_if_due(self) -> Optional[Dict[str, Any]]:
        """The long-idle cycle (see sleep.py): reflect, consolidate, fade, dream."""
        from . import sleep as _sleep                  # sleep imports this module, so not at the top
        cfg = self.load_cfg()
        if time.time() < self._sleep_retry_at:
            return None
        if not _sleep.sleep_due(self.engine, cfg, self.last_activity) or not self._busy.acquire(blocking=False):
            return None
        self._begin("sleeping")
        try:
            report = _sleep.sleep_once(self.engine, cfg, key_fn=self.key_fn, foundation=self.foundation_fn())
            self.last_sleep_error = "; ".join(report["errors"])
            logger.info("holonomic: slept: %d conversation(s) summarised, dream %s",
                        len(report["episodes"]), "yes" if (report.get("dream") or {}).get("id") else "no")
            return report
        except Exception as exc:
            # A sleep that fails outright is not tried again every half minute: it would run the reflection
            # model flat out, and the reason would be gone before anyone looked.
            self.last_sleep_error = f"{exc.__class__.__name__}: {exc}"[:500]
            self._sleep_retry_at = time.time() + 600.0
            logger.warning("holonomic: sleep failed: %s", exc)
            return None
        finally:
            self._busy.release()
            self._end()

    def images_if_due(self) -> Optional[Dict[str, Any]]:
        """Describe images that are waiting (see images.py): any whose description failed when it was shown,
        and the parts of each image.  By default this waits for a pause, and stops when the conversation resumes."""
        from . import images as _images                # images imports this module, so not at the top
        cfg = self.load_cfg()
        ic = _images.image_config(cfg)
        if not ic["image_enabled"] or not ic["image_model"]:
            return None
        wait = 0.0 if ic["image_sections_when"] == "now" else float(ic["image_idle_seconds"])
        idle = lambda: time.time() - self.last_activity >= wait       # noqa: E731
        if not idle() or time.time() < self._images_retry_at:
            return None
        waiting = _images.pending(self.engine)
        from . import fingerprints as _fp
        prints = _fp.fingerprint_config(cfg)["image_fingerprints"] and _fp.status(self.engine)["waiting"]
        from . import faces as _faces
        prints = prints or (_faces.faces_on(cfg) and (_faces.waiting(self.engine) or self.engine.kv_get("faces:look_again") == "1"))
        if not (waiting["images"] or (ic["image_sections"] and waiting["sections"]) or prints) or not self._busy.acquire(blocking=False):
            return None
        self._begin("describing images")
        try:
            report = _images.process(self.engine, cfg, key_fn=self.key_fn, should_stop=lambda: self._stop.is_set() or not idle())
            self.last_image_error = "; ".join(report["errors"])
            if (report.get("fingerprints") or {}).get("errors") or (report.get("faces") or {}).get("errors"):       # the helper is probably not running: not every few seconds
                self._images_retry_at = time.time() + 300.0
                logger.info("holonomic: fingerprints or faces not done: %s",
                            "; ".join((report.get("fingerprints") or {}).get("errors", []) + (report.get("faces") or {}).get("errors", [])))
            if report["errors"]:                       # the vision server is probably off: try again in a few minutes
                self._images_retry_at = time.time() + 300.0
                logger.warning("holonomic: describing images failed: %s", self.last_image_error)
            elif report["described"] or report["sections"]:
                logger.info("holonomic: described %d image(s) and %d part(s)", len(report["described"]), report["sections"])
            return report
        except Exception as exc:
            self.last_image_error = str(exc)
            self._images_retry_at = time.time() + 300.0
            logger.warning("holonomic: describing images failed: %s", exc)
            return None
        finally:
            self._busy.release()
            self._end()

    def once(self) -> None:
        """One round of the worker.  Each task stands alone: the three used to share one guard, so a task that
        raised stopped the ones after it from ever being tried, round after round, and said nothing."""
        for name, task in (("describing images", self.images_if_due), ("reflection", self.run_if_due), ("sleep", self.sleep_if_due)):
            try:
                task()
                self.stumbles.pop(name, None)
            except Exception as exc:                   # never let the worker die, and never let one task stop another
                import traceback
                where = traceback.extract_tb(exc.__traceback__)[-1]
                self.stumbles[name] = f"{exc.__class__.__name__}: {exc} (at {Path(where.filename).name}:{where.lineno})"[:400]
                logger.warning("holonomic: %s could not be checked: %s", name, self.stumbles[name])

    def _loop(self) -> None:
        try:                                             # the one shared file an earlier version wrote
            (Path(self.engine.path) / "worker.json").unlink()
        except OSError:
            pass
        self.note()
        while not self._stop.wait(self.poll_seconds):
            self.once()
            self.note()
