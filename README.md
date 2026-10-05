# Holonomic memory for Hermes Agent

A local, associative long-term memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com),
inspired by the holonomic brain theory. Links between memories are not stored in a table: they are
superposed as interference patterns on fixed-size complex vectors ("plates") and recovered by
resonance with a cue.

Status: storage, recall, Hermes integration and reflection work on a real install. Sleep (consolidation,
fading and dreaming) is built and off by default; it has only been run against a stand-in model so far.

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

## Reflection

Off by default. When on, a language model periodically reads the memories stored since its last pass and writes:

- facts about the user, notes about itself, notes about the relationship, and insights connecting
  several memories. Each is stored as a memory linked through the plates to the memories it came from,
  so recalling either surfaces the other.
- three short profiles: the user, itself, and the two together. These go into the system prompt, so
  who the user is never depends on a search matching.

A fact about the user must cite at least one thing that was not said by the agent itself; facts resting
only on the agent's own words are dropped. Quotation marks in a fact about the user are removed unless
the quoted words appear in what the user actually said.

Facts about the user change by contradiction, not by time. When the user plainly says something that
makes a stored fact untrue, reflection stores the new fact and marks the old one superseded: it leaves
recall but stays on record, linked to what replaced it. Only the user's own words can do this. Nothing
fades yet; when fading is added, facts about the user are exempt from it.

Hermes' `SOUL.md` is the fixed foundation. Reflection is shown it on every pass and told to grow within
it, never to restate or contradict it, and this plugin never writes to it. The self profile is only what
the agent has become beyond that foundation. You can write any profile yourself as a starting point,
and the agent can search one subject at a time through its memory tool (`subject`: `user`, `self`, `us`).

A pass is up to three small model calls, each with one job: propose items from the new memories; check
each item against only the lines it cites, keeping, rewriting or dropping it; then rewrite the profiles
from the items that survived, never from the raw conversation. `reflect_depth` sets how much of that runs:

| Depth | What a pass does |
|---|---|
| 1 | Facts about the user and the user profile only |
| 2 | Everything, proposed and stored unchecked |
| 3 (default) | Everything, with each item checked before it is stored |

If a reasoning model uses up its token budget thinking without answering, that step is repeated once
without thinking.

It runs in the background once enough new memories exist and the conversation has been quiet for a
while, so it does not compete with a reply for the GPU. Items that cite no real memory are discarded,
a conclusion reached twice strengthens the existing memory instead of duplicating it, and every
version of a profile is kept.

```
hermes holonomic reflect on --model NAME      # NAME as shown by `ollama list`; also --host URL, --depth 1|2|3
hermes holonomic reflect off
hermes holonomic reflect                      # status: on or off, memories waiting, last run
hermes holonomic reflect now --dry-run        # run once and show the result without storing it
hermes holonomic reflect now                  # run once now, whether or not it is switched on
hermes holonomic profile --history
hermes holonomic profile --set user "Kayla is ..."     # write a starting profile: user, self or us
```

| Key | Default | Meaning |
|---|---|---|
| `reflect_enabled` | `false` | The toggle |
| `reflect_model` | (none) | Ollama model that does the reflecting |
| `reflect_host` | same as `ollama_host` | Ollama server for that model |
| `reflect_depth` | `3` | How much each pass does (see above) |
| `reflect_min_new` | `12` | New memories needed before a pass |
| `reflect_idle_seconds` | `300` | Quiet time needed before a pass |
| `reflect_batch` | `60` | Memories read per pass |
| `reflect_temperature` | `0.3` | Sampling temperature |
| `reflect_max_tokens` | `2000` | Hard cap on the model's reply |
| `reflect_think` | `false` | Let a reasoning model think first. Slow, and small models can think until the token budget is gone |
| `reflect_timeout` | `600` | Seconds before a pass is abandoned |
| `profile_max_chars` | `1200` | Size limit of each profile |

Background reflection only runs while a Hermes process is alive (the desktop app or gateway). A
terminal session that exits straight away never goes quiet for long enough; use `reflect now` there.

## Short-term and long-term memory, and sleep

Conversation as it happened is short-term memory. Facts, notes, profiles and accounts of past
conversations are long-term. A sleep cycle moves experience from one to the other. It runs when the
agent has been idle for a long stretch, or by hand, and has four steps, each of which can be switched off:

1. **Reflect**: as above.
2. **Consolidate**: for each conversation that has gone quiet, the agent writes a short first-person
   account of it. The account is stored long-term, dated as the conversation, and linked to it.
3. **Fade**: conversation that has been summarised halves in strength every `fade_half_life_days`.
   Below `fade_threshold` a memory is left out of everyday recall. Nothing is deleted. What the agent
   knows about the user, its notes and its accounts do not fade, and whatever is recalled is strengthened.
4. **Dream**: a few recent memories, and older memories each one echoes (faded ones included), are woven
   into a dream. The agent then rereads it awake, notes what it makes of it, and records a connection
   only if a recent and an older memory really bear on each other.

A sleep has one dream, plus one more for every forty memories since the last sleep, up to three. Each
dream draws its starting memories across all recent conversations, preferring ones not yet drawn from,
so different days run together; a later dream starts from memories the earlier ones did not use.

Dreams and what was made of them live in their own realm. Factual recall never reads it, and by default
a dream strengthens nothing, so it does not bring faded memories back to the surface. The agent can still
recall its dreams and talk about them: the latest one is in its system prompt for a few days, dreams are
offered when the conversation turns to dreaming, and its memory tool has a `dreams` action. They are
always labelled as dreams.

**Talking about dreams.** A conversation about a dream really happened, so it is stored as ordinary
conversation, but what was said in it describes a dream. Such pieces are labelled as dream talk: either
they say so ("in my dream", "did you dream") or they closely resemble a stored dream. Dream talk is
labelled wherever it is recalled, no fact about the user may rest on it, the account of that conversation
is told to report it as talk about a dream, and dreams are never seeded from it.
A sentence about how dreaming works ("your dreams are kept in a separate realm") is not dream talk
unless it also resembles a dream, and only a question about what was dreamed makes the whole reply
dream talk. `hermes holonomic dreamtalk` finds and labels such conversation stored by an earlier
version, and `hermes holonomic relabel ID --as said|dream` corrects a label by hand.

**Deep recall.** Everyday recall skips faded memories. The agent's memory tool takes `deep: true`, which
searches them too and follows links two steps out. A faded memory recovered that way is strengthened
back into everyday reach.

```
hermes holonomic sleep                        # status
hermes holonomic sleep now --dry-run          # run the cycle and show the result without storing it
hermes holonomic sleep now --only consolidate,dream
hermes holonomic sleep on                     # run unattended when idle (needs a reflection model)
hermes holonomic dreams
hermes holonomic recall "..." --deep
```

| Key | Default | Meaning |
|---|---|---|
| `sleep_enabled` | `false` | Run the cycle unattended |
| `sleep_idle_seconds` | `3600` | Quiet time needed before sleeping |
| `sleep_min_hours` | `12` | Minimum time between sleeps |
| `consolidate_enabled` | `true` | Write accounts of finished conversations |
| `consolidate_quiet_minutes` | `30` | A conversation counts as finished after this |
| `fade_enabled` | `true` | Let summarised conversation fade |
| `fade_half_life_days` | `5` | How fast it fades |
| `fade_threshold` | `0.35` | Strength below which a memory leaves everyday recall |
| `dream_enabled` | `true` | Dream during sleep |
| `dream_model`, `dream_host` | the reflection model and server | Model that dreams |
| `dream_temperature` | `1.0` | Sampling temperature for the dream itself |
| `dream_max_per_sleep` | `3` | Most dreams in one sleep |
| `dream_memories_per_extra` | `40` | One more dream for every this many memories since the last sleep |
| `dream_min_words`, `dream_max_words` | `100`, `180` | Length of each dream |
| `dream_reinforce` | `false` | Let a dream strengthen the old memories it touches |

## Images

Off by default. Turn it on with a model that can see:

```
hermes holonomic images on --model gemma4 [--host http://OTHER-OLLAMA:11434] [--sections idle|now|off]
```

When the user attaches an image to a message, three things are kept:

- **The file.** The original bytes, unchanged, in `holonomic/images/`, plus a copy that fits inside
  2048x1536 (1536x2048 for a portrait image), keeping its shape, for looking at and showing. Showing the same file again
  does not store it twice; it is counted as seen again, and she is told she has seen it before.
- **A description of the whole image**, written by the vision model and stored as an ordinary memory
  (kind `image`), linked to what was said when it was shown. It is recalled, fades and can be
  reached by deep recall like any other memory.
- **A description of each part.** The image is cut into a 3x3 grid of parts, each half the width and
  height and overlapping its neighbours by half, so something cut in two by one part lies near the
  middle of the next. Each part is shown to the model on its own, with the description of the whole
  for context, and stored (kind `image_part`) unless the model says it shows nothing notable. Parts
  are cut from the original when it is larger than the copy. By default this waits until the
  conversation has been quiet for two minutes and stops when it resumes.

Everything the model names in an image or a part is also filed as a label, so "every image with a
cat in it" is an exact lookup: `hermes holonomic images find cat`, or the tool's `images` action
with `label`. It lists every image in which the thing was noticed, and only those.

A recalled image appears once, with its file and the parts that matched:

```
- [#77] (2026-10-05, image #3 you were shown; file: C:\...\holonomic\images\ab12.view.jpg) A photo of a black cat asleep on a grey sofa...
    in the top left [#79]: A window with a blue curtain and a spider plant on the sill.
```

The agent shows an image by writing `MEDIA:` and the file path in its reply (Hermes' own
convention; the desktop app and messaging platforms display it inline, the terminal does not).
The tool's `look` action shows a stored image to the vision model again with a question.

```
hermes holonomic images                       # status, and what is waiting to be described
hermes holonomic images add photo.jpg --say "This is my cat"     # keep and describe an image from the terminal
hermes holonomic images process               # describe everything that is waiting
hermes holonomic images list | show 3 | find cat | labels
hermes holonomic images look 3 "what colour is the car?" [--section 4]
hermes holonomic images redo 3 --fix "That is a couch, not someone's lap"      # describe again, with a correction taken as true
hermes holonomic images forget 3 --yes [--keep-files]
```

| Key | Default | Meaning |
|---|---|---|
| `image_enabled` | `false` | Keep and describe images |
| `image_model`, `image_host` | reflection's | Vision model and its Ollama server |
| `image_view_width`, `image_view_height` | `2048`, `1536` | The copy fits inside this; existing copies are not remade |
| `image_sections` | `true` | Also describe the image part by part |
| `image_grid` | `3` | Parts per side (3 = nine parts) |
| `image_sections_when` | `idle` | `idle`: when the conversation is quiet; `now`: straight away |
| `image_idle_seconds` | `120` | How quiet |
| `image_from_original` | `true` | Cut parts from the original, not the copy |
| `image_section_min_side` | `640` | Smaller images are not cut into parts |

If the vision model cannot be reached the image is still kept, and described when the model is
back. Reflection does not read image descriptions: what a picture shows is not something the user
said. Each image and part has an empty place for a vector, reserved for fingerprints from a vision
embedding model (recognising similar pictures and things within them without words).

### Images in dreams

`dream_images` sets what images do when she dreams. Each level includes the ones before it.

| Mode | What happens | Needs |
|---|---|---|
| `off` | Images play no part in dreams | |
| `words` (default) | What she saw in recent images, and in older images they resemble, joins the fragments a dream is made from | Image memory on |
| `pictures` | Scenes from each dream are drawn by an image generator, from her own description of them | An image generator |
| `from_images` | A scene that resembles images she has seen is drawn starting from those images, laid over each other | An image generator that can rework a picture |

```
hermes holonomic dreams images                 # what is set now
hermes holonomic dreams images from_images --api comfyui --host http://10.0.0.21:8188 --size 768x512 --count 3
hermes holonomic dreams images --people yes    # allow images with real people in them to be drawn from
```

The plugin does not draw. It asks a server you run: ComfyUI (`comfyui`), the Stable Diffusion WebUI
API (`a1111`: AUTOMATIC1111, Forge, SD.Next) or the OpenAI images API (`openai`), which several
local servers speak. `hermes holonomic dreams images --test "a cat asleep in a greenhouse"` draws
one picture to a file, storing nothing, to check the connection.

With ComfyUI, a single-file model (Stable Diffusion 1.5, SDXL) in `models/checkpoints` is used as it
is; with none named, the first one the server lists. Z-Image-Turbo is chosen by naming it:
`--model z_image_turbo_bf16.safetensors` (it also needs `qwen_3_4b.safetensors` in
`models/text_encoders` and `ae.safetensors` in `models/vae`). Pictures of a dream are kept in the dream realm: they are not listed among images she was
shown, are not described as one, and reach her only with the dream they belong to, labelled as
pictures of a dream. She can show them the same way as any image.

Each picture is drawn several times and the vision model, shown all the attempts with the scene they
are meant to show, chooses which one is kept; the others are discarded. With one graphics card
shared between the language model and the image generator, `--swap on` unloads the language models
(not the embedding model) while pictures are drawn, asks the image server to free the card when it
is done, and loads them again exactly as they were, so the agent is ready to talk when the dream
step ends. A message sent while pictures are being drawn makes Ollama load the model again at once;
the drawing may then fail for lack of memory, and the dream is kept without those pictures.

In `from_images` mode, images in which the vision model saw real people are not drawn from unless
`dream_image_use_people` is set; a scene with a person in it is then drawn from words alone.
`hermes holonomic images dream ID yes|no|default` decides it for one image, whatever the general rule:
`no` keeps a particular photo out of dream pictures, `yes` allows one when people are otherwise excluded. Stored
images are only read. If the image generator cannot be reached the dream is kept without pictures.

| Key | Default | Meaning |
|---|---|---|
| `dream_images` | `words` | `off`, `words`, `pictures` or `from_images` |
| `dream_image_seeds` | `2` | Recent images a dream draws on |
| `dream_image_api`, `dream_image_host`, `dream_image_model` | none | The image generator |
| `dream_image_count` | `3` | Pictures per dream |
| `dream_image_candidates` | `3` | Each picture is drawn this many times; she looks at them and keeps one |
| `dream_image_choose_think` | `false` | Let her reason before choosing; if the reasoning runs away she is asked again without it |
| `dream_image_swap` | `false` | One graphics card: unload the language models while pictures are drawn, reload them after |
| `dream_image_width`, `dream_image_height` | `768`, `512` | Size of a dream picture |
| `dream_image_strength` | `0.75` | `from_images`: how far a picture may move from the images it starts from |
| `dream_image_style` | `dreamlike, soft light, slightly out of focus` | Added to every scene |
| `dream_image_use_people` | `false` | `from_images`: may images with real people be drawn from |
| `dream_image_describe_source` | `true` | `from_images`: add her description of the source image to the scene, so its details are asked for |

## Inspecting the store

```
hermes holonomic stats
hermes holonomic list -n 20
hermes holonomic recall "what is my name"
hermes holonomic show 37 40
hermes holonomic forget 52 --yes
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
- Everyday recall follows associations one step; deep recall follows two.
- Forgetting removes a memory's text and vector. Its traces stay in the plates as faint noise.
- Changing the embedding model, plate dimension or capacity requires a new store.
- One user. Everyone who talks to the agent is treated as the same person.
- An image is found through what the vision model wrote about it. A thing it did not mention is not
  found until she looks at the image again. Two different files of the same picture are two images.
- Only images the user attaches are kept, not images that tools return.
- Tested against Hermes Agent v0.21.5 (commit bd0affe5).

## Roadmap

- Rebuild long-term plates from what survives, so old plates can be retired.
- A deeper reflection level for stronger models: older linked memories as context, and periodic
  re-derivation of the profiles from all stored facts.
- Fingerprints for images and their parts from a vision embedding model, with names bound to them,
  so people, places and things are recognised across images without words.
- Video dreams; unloading the chat model while pictures are drawn, for a single GPU.
- More than one user.

## Licence

MIT
