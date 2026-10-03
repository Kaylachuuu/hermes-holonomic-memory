# Holonomic memory for Hermes Agent

A local, associative long-term memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com),
inspired by the holonomic brain theory. Links between memories are not stored in a table: they are
superposed as interference patterns on fixed-size complex vectors ("plates") and recovered by
resonance with a cue.

Status: storage, recall and Hermes integration work. Reflection and dreaming are planned, not built.

## How it works

- Every piece of text is embedded by an Ollama model (default `nomic-embed-text`) and lifted into a
  4,096-dimension vector of unit complex numbers (a phasor).
- An association is written as `plate += cue * target`. Many associations share one plate. A memory is
  associated with the message before it in the session, with the user's previous message, with any
  memory it is explicitly linked to, and with the names and topics it mentions.
- Recall has two steps. First, memories similar to the query are found by vector similarity plus a
  bonus for shared content words (a question and the statement that answers it are far apart as vectors).
  Then the best matches are used as cues on the plates, which returns memories that are *linked* to
  them even when they share no words or meaning with the query.
- A plate returns a noisy blend. A cleanup step resolves the blend against the memories known to be
  on that plate, and rejects anything not clearly above the plate's measured noise.
- Plates fill up. When one reaches its energy capacity it is sealed and a new one is started, so
  total capacity is bounded only by disk.
- Memories carry a realm (`waking`, later `dream`). Recall only probes the realms it is asked for.

Everything is local: one SQLite file and one projection matrix under `$HERMES_HOME/holonomic/`.

## Install

1. Copy this folder to `$HERMES_HOME/plugins/holonomic/` so that `__init__.py` sits directly inside it.
   On Windows that is `%LOCALAPPDATA%\hermes\plugins\holonomic\`; on Linux and macOS `~/.hermes/plugins/holonomic/`.
2. Run `hermes memory setup`, choose `holonomic`, and enter the Ollama address and embedding model.
3. Start Hermes. `hermes memory status` should show the provider as active.

Requirements: Python 3.10+ and an Ollama server
with an embedding model pulled. `numpy` and `threadpoolctl` are declared in `plugin.yaml`; the setup wizard
installs them after you choose the provider, and Hermes needs a restart afterwards.

## Configuration

`$HERMES_HOME/holonomic.json`. Only the first two are asked for during setup.

| Key | Default | Meaning |
|---|---|---|
| `ollama_host` | `http://localhost:11434` | Ollama server address |
| `embed_model` | `nomic-embed-text` | Embedding model. Changing it later invalidates the store |
| `embed_timeout` | `10` | Seconds before an embedding call is abandoned |
| `recall_k` | `6` | Memories injected per turn |
| `min_score` | `0.2` | Floor: a memory scoring below this is never injected |
| `score_band` | `0.25` | A memory must also score within this much of the best match |
| `lexical_weight` | `0.2` | How much sharing content words with the message adds to similarity |
| `assistant_weight` | `0.75` | Ranking weight of the agent's own past statements |
| `max_context_chars` | `2400` | Size budget of the injected block |
| `max_item_chars` | `420` | Each recalled memory is trimmed to this |
| `dual_query` | `false` | Also match each message as a statement (a second embedding call) |
| `store_assistant` | `true` | Also remember the agent's own replies |
| `max_turn_chars` | `4000` | Longer messages are cut before storing |
| `dim` | `4096` | Plate dimension. Fixed once the store exists |
| `plate_capacity` | `128` | Plate energy at which a plate is sealed. Fixed once the store exists |

## What the agent sees

Each turn, relevant memories are injected like this:

```
## Holonomic Memory (recalled; may be incomplete or outdated)
- [#12] (2026-10-03, user said) The snapshot was taking forty minutes because the disk is nearly full.
- [#13] (2026-10-03, user said, linked) Unrelated, but the bakery on Elm Street started selling cardamom buns.
```

It also gets one tool, `holonomic_memory`, with the actions `recall`, `remember`, `related`,
`feedback`, `forget` and `stats`.

User messages are stored one sentence per memory, so each fact is separately findable. Questions
are stored to keep the conversation chain intact but are never recalled. Sentences in which the model
says it has no memory, or does not know something, are not stored; nor are short replies to bare questions.

Memories from the current session are not injected while they are still in the context window; they
become eligible again after Hermes compresses the context. Writes to Hermes' built-in memory are
mirrored. Subagents and scheduled runs can read memory but do not write to it.

## Measured

Associative recall on a synthetic benchmark (six-turn conversations with one off-topic aside each):

| Memories per plate | Off-topic neighbour recovered | False associations per query |
|---|---|---|
| 46 | 100% | 0.03 |
| 60 (default) | 100% | 0.03 |
| 90 | 98.5% | 0.01 |
| 180 | 64% | 0.05 |

With real `nomic-embed-text` vectors on a small scripted test (`selftest.py`): 8 of 8 top hits correct,
8 of 8 off-topic asides recovered, no false associations; about 20 ms per recall, of which about
15 ms is the embedding call.

Recall time without embedding, two CPU cores: 5 ms at 2,000 memories, 10 ms at 20,000, 32 ms at
100,000. Disk at 100,000 memories: about 210 MB of plates and 155 MB of cleanup vectors.

## Inspecting the store

```
hermes holonomic stats
hermes holonomic list -n 20
hermes holonomic recall "what is my name"
```

`recall` prints every candidate with its scores and whether it would have been injected.

## Testing

```
python scripts/run_tests.py          # engine tests; provider tests too if a Hermes checkout is found
python scripts/selftest.py --host http://YOUR-OLLAMA:11434
python scripts/bench.py
```

Provider tests need the Hermes source: set `HERMES_SRC` or keep a `hermes-agent` checkout next to
this folder.

## Limits

- The plates hold associations only. Text and one embedding per memory are stored conventionally,
  and finding memories similar to a query is ordinary vector search.
- Associations are one step deep per recall.
- Forgetting removes a memory's text and vector. Its traces stay in the plates as faint noise.
- Changing the embedding model, plate dimension or capacity requires a new store.
- Tested against Hermes Agent v0.21.5 (commit bd0affe5).

## Roadmap

- Reflection: during idle time a language model reads recent memories and writes conclusions about
  the user and about itself, stored as linked memories.
- Dreaming: loose, blurred cues pull distant memories together into a narrative, kept in the `dream`
  realm and never mixed into factual recall.
- Separate, configurable model and endpoint for each of those, independent of the conversation model.
- Consolidation of old plates and gradual decay.

## Licence

MIT
