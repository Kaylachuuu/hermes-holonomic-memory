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
hermes holonomic profile --set user "Kayla is ..."     # write a starting profile: user, projects, self or us
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

A pass reads what is new since the last one, and can miss something (it once wrote down one of two
cats). `hermes holonomic reflect now --again 3` goes over the last three days of conversation again.
Facts she already has are reinforced, not stored twice, and the mark for what is new does not move
back. `--dry-run` shows what it would add.

**Who the user is, and what the user is working on.** Reflection keeps two kinds of fact about the user. A
personal fact is about them as a person: family, the people and animals in their life, age and birthday, what
they do for a living, likes and dislikes, beliefs, how they like to be spoken to. A project fact is about
something they are making: what it is, how it works, decisions made, how far it has got. The user's profile is
written from personal facts only, because it is what she needs in order to talk with them and relate to them.
A separate, shorter profile says what they are working on, a sentence or two for each project. Project facts
are recalled like any other memory when the talk turns to the project; they no longer shape the portrait of
the person.

    hermes holonomic reflect sort              sort the facts she already has, and show the two profiles it would write
    hermes holonomic reflect sort --apply      do it
    hermes holonomic relabel ID --as project   put one fact right (or --as personal)

**The same thing said twice.** Going over old ground, or simply being told something again, can leave a fact
in memory two or three times. `reflect merge` finds statements that say the same thing and keeps one of each,
the one that says more. Where the words settle it (the same words, or one statement going on where the other
stops) no model is asked; statements close in meaning but worded differently are put to the reflection model.
"Kayla has a cat named Sushi" and "Kayla has a cat named Theo" are as alike as two sentences get and are two
facts: a pair that differs in one place, or in its numbers, is never merged. The statement not kept is
retired, not destroyed: it leaves recall, stays on record, and can be brought back. Unattended sleep does the
part that needs no model on its own (`merge_in_sleep`).

    hermes holonomic reflect merge             show what would be retired
    hermes holonomic reflect merge --apply     do it
    hermes holonomic reflect merge --quick     only what the words settle; the model is not asked
    hermes holonomic reflect merge --restore ID    bring a retired statement back

Guards that run on every reflection. A name one letter off from one that is well established (it wrote
"Kaylar" for "Kayla" through a whole run once) is put right before the fact is stored, and the report says so;
an ordinary word is never touched, and neither is any word you have written yourself, so two real names a
letter apart both stand. The profiles are written from everything a run took in, not only its last pass, and a
profile is rewritten only when the pass learned something of its kind (a fact for yours, a note about herself
or an insight for hers, a note about the two of you for the shared one): every rewrite drifts a little. A
profile is told to begin with the person's life (people and animals by name); a sentence said twice is kept
once; and one that comes back longer than `profile_max_chars` (1200) is sent back, twice at most, to be said
shorter before anything is cut. If it is still too long, the report says how much was cut.


## Reference libraries

A library is material to work from (manuals, notes, file format descriptions, source code), built from a
folder of files. It has its own plates, beside the memory store and never mixed into it: what is in a
library does not fade, is not reflected on, is not dreamt about, and is nobody's memory.

Nothing from a library reaches a conversation unless it is open in that conversation, and a new conversation
starts with none open. Working on a project in the evening does not put its details into the next morning's
talk. She still remembers, as ordinary memory, that the two of you worked on it.

In conversation:

- "Make a library called x86 from C:\Users\you\Documents\x86-reference." She makes it and reads the folder
  in the background; ask her how it is getting on.
- "Use the x86 and fat-filesystems libraries for this." They are open until the conversation ends. From the
  next message on she is given what they have on each message, with the file each piece came from, and she
  can look things up in them herself.
- Opening is yours to decide, and that is enforced, not merely asked of her: the tool opens a library only
  when your own message asks for it (it names the library and speaks of a library, data store or reference
  material), or when you say yes after she has asked whether to open it. Told only in her instructions not to
  open one unasked, she reasoned that it was "logical" to and did.
- A question a library would answer does not need it open. She may look the answer up once, says which
  library it came from, and nothing more from that library reaches the conversation.
- A new conversation is told which libraries the last one used, so she can ask whether to pick the work up
  again.

From the command line:

    hermes holonomic library                               what there is
    hermes holonomic library create NAME "FOLDER" [--about "what it is for"]
    hermes holonomic library update NAME                   read the folder again: new and changed files in, missing ones out
    hermes holonomic library rebuild NAME                  read every file again (after an update changes how files are cut)
    hermes holonomic library show NAME                     its files, and how many pieces each became
    hermes holonomic library search NAME "words"           what she would be given
    hermes holonomic library delete NAME --yes             remove the library (the folder of material is not touched)

What is read: text of any kind (Markdown, plain text, source code, HTML with its markup removed), `.docx`,
and `.pdf` if `pypdf` is installed in Hermes' Python (a scanned PDF is pictures and has no text to read).
Writing is cut into pieces at its headings and paragraphs; source code at its routines (assembly labels,
Pascal and BASIC procedures, C and Python functions), with the comment above a routine kept with it, so a
piece is "BOOT.ASM > ReadSectors" and not merely somewhere in BOOT.ASM. An old copy of a source file under
another name (BOOT.BAK, DIR.OLD) is recognised by what is in it. The pieces of one section are bound together,
so finding one brings its neighbour. A passage that is in several files (a backup, an earlier version of the
project) is given once, with a note of where else it is. A file that is a program and not text is left out,
judged by its contents. Exact words count for more here than in memory (`library_lexical_weight`),
because a question about `INT 13h` wants the passage that says `INT 13h`. A build can be stopped and run
again; only what is new or changed is read. `hermes holonomic context` shows what the libraries gave her.

Settings: `library_k` (4 pieces per message), `library_min_score` (0.3), `library_context_chars` (3600),
`library_piece_chars` (1100), `library_max_file_mb` (40), `library_max_files` (5000).

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

Writing in an image is kept only when two separate looks agree on it, or the user stated it. A
vision model asked what a small sign says will often supply something plausible instead of saying it
cannot read it, and such inventions come from a single look, while a legible sign is read the same
way by overlapping parts. Writing reported once is replaced by `[writing I could not read for
certain]` and labels made from it are dropped. Parts are not told what writing the overall
description quoted, so each reads for itself; they are told what the user said about the image.

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
hermes holonomic images forget 3                  # set aside: she no longer recalls it; nothing is deleted
hermes holonomic images removed                   # what is set aside
hermes holonomic images restore 3                 # put it back exactly as it was
hermes holonomic images delete 3 --yes            # remove for good; only works on an image already set aside
```

Forgetting an image, from the command line or by the agent in conversation, only sets it aside: her
description, the parts, what was said when it was shown and every correction are kept with it, and
the files stay on disk, so `restore` brings it back exactly. The agent can undo its own mistake the
same way. Removing an image and its files for good is a second step, `delete`, which refuses any
image that has not been set aside first and is available only from the command line.


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

Each picture is drawn several times. The vision model looks at each attempt on its own and notes what
it shows, what looks wrong and a score; then it reads its notes side by side and chooses which one is
kept. The others are discarded. (Shown all the attempts in one message, the model this was tuned with
reported seeing one picture, or two that were the same, and kept the first. One picture per look
works with any vision model.) `sleep now` prints her notes on each attempt and how long the drawing took.

Before she looks, she lists what a picture of the moment has to show (the main things, their
distinguishing details, what the dream did to them). Each attempt is then questioned about those
things one at a time, about any writing in it, and about the bodies of people and animals, and its
score is worked out from her answers. Asked only what a picture showed and whether anything was
wrong, she scored nearly everything 8 to 10 and missed details that were plainly there.

`dream_image_enlarge` makes the picture she keeps larger with an upscaling model on the image server,
after she has chosen it: `hermes holonomic dreams images --enlarge 2`. For ComfyUI, put an upscaling
model in `models/upscale_models` (default name `RealESRGAN_x2plus.pth`). This is how to get large
pictures without drawing them large; see the note on size below.

A picture drawn from an image she has seen stays close to that image. If her best attempt at such a
picture scores below `dream_image_redraw_below`, it is drawn again holding less tightly to the image
(`dream_image_redraw_strength`). She then compares the new attempts with the one she had chosen, from
her notes on all of them, and keeps whichever she prefers.

**Signatures and watermarks.** A signature on a photo is not carried into pictures drawn from it. The
first time a dream draws from an image, the vision model is asked which corner, if any, is signed or
watermarked; that corner is smoothed over in the copy given to the image generator (the stored image
is never changed), and sentences about the signature are left out of what the painter is told.
`hermes holonomic images signature ID none|bottom-right|...` corrects the answer for one image, and
`hermes holonomic dreams images --signature keep` turns this off.

**Picture size and the graphics card.** Past a certain size a picture no longer fits on the card
beside the drawing model, and drawing becomes several times slower with no error and nothing in the
image server's log but the step time. With Z-Image-Turbo on a 16 GB card, 1024x768 drew at about 2
seconds a step and 1280x960 at about 9. If drawing is slow, try the next size down before anything
else: `hermes holonomic dreams images --size 1024x768 --test "a cat asleep on a couch"`. With one graphics card
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
| `dream_image_enlarge` | `0` | Make the kept picture this many times larger with an upscaling model (comfyui, a1111); `0` = off |
| `dream_image_enlarge_model` | | Upscaling model: a file in ComfyUI's `models/upscale_models`, or an A1111 upscaler name |
| `dream_image_keep_signature` | `false` | Carry a signature or watermark on an image into pictures drawn from it |
| `dream_image_redraw_below` | `9` | A picture drawn from an image is drawn again when her best attempt scores below this out of 10; `0` = never |
| `dream_image_redraw_strength` | `0.85` | How freely the second drawing departs from the image |
| `dream_image_choose_think` | `false` | Let her reason before comparing her notes on the attempts; if the reasoning runs away she is asked again without it |
| `dream_image_swap` | `false` | One graphics card: unload the language models while pictures are drawn, reload them after |
| `dream_image_width`, `dream_image_height` | `768`, `512` | Size of a dream picture |
| `dream_image_strength` | `0.75` | `from_images`: how far a picture may move from the images it starts from |
| `dream_image_style` | `dreamlike, soft light, slightly out of focus` | Added to every scene |
| `dream_image_use_people` | `false` | `from_images`: may images with real people be drawn from |
| `dream_image_describe_source` | `true` | `from_images`: add her description of the source image to the scene, so its details are asked for |

### Picture fingerprints

A description records what the vision model thought to mention. A fingerprint is a list of numbers
worked out from the picture itself by a model made for that, so two pictures of the same thing come
out close together whether or not anyone would describe them alike. Each kept image gets one for the
whole picture and one for each of its parts, so a thing in the corner of one image can be matched
with the same thing filling another.

Ollama cannot make these, so a small helper server does. It is not loaded by Hermes; run it with any
Python that has `torch` and `transformers` (ComfyUI's will do), and it downloads its model, DINOv2
base, about 350 MB, on first start:

```
C:\Users\you\ComfyUI\.venv\Scripts\python.exe "%LOCALAPPDATA%\hermes\plugins\holonomic\tools\fingerprint_server.py"
hermes holonomic images fingerprints on           # [--host URL] if the helper is on another machine
hermes holonomic images similar 12                # images that look like image #12
hermes holonomic images similar 12 --all          # every image with its figures, to judge the threshold
```

The helper runs on the processor by default (`--device cpu`), which is quick enough and leaves the
graphics card to the models that need it. Fingerprints are made when images are described and when
the conversation is quiet. The agent finds look-alikes with the `images` action, `image_id` and
`similar`. Fingerprints from different models cannot be compared: if the helper's model changes, they
are made again.

`image_fingerprint_min` (default `0.5`) is how alike two pictures must be to be called alike, on a
scale where 1 is the same picture. It was chosen before being tried on real photographs; run
`similar --all` on your own images and set it where related and unrelated pictures separate. A part
of an image the vision model judged to show nothing worth recording is not matched on, since two
plain walls are alike and it says nothing about the images.

**Named things.** When the user names a particular animal, object or place in an image ("This is
Sushi, my cat"), the name is kept with that image as an example of what the thing looks like: the
parts of the image whose descriptions use the name, or the whole image if none does. When a later
image has a part that looks like it (`image_name_min`, default `0.6`), the vision model is told
before it describes the image that it may be looking at Sushi, and to use the name only if it can
see such a thing. The name is recorded for the image only when she then uses it, so two things have
to agree: the fingerprint and her own look.

```
hermes holonomic images name 12 Sushi --what "a long-haired black and white cat"
hermes holonomic images name 31 Sushi --not      # that image does not show Sushi
hermes holonomic images names                    # what she knows by name, and where she has seen it
hermes holonomic images names forget Sushi
```

An image that has only just arrived with a message is checked the same way before she answers, and
she is told what it may show. Otherwise she has only the conversation to go by: shown Theo and then a
second cat, she called the second one Theo. What she recognises is then stated to her outright (that she knows it, by what name, and not to ask
whose it is) and its name is added to what her memory is searched for, so recognising the cat brings
back what she knows about the cat even when the message says only "look at this". A name mentioned
in passing in the image's description was not enough: she admired "a little tuxedo cat" and asked
whether it was the user's. A person in the picture must not decide which cat it is: the same woman in two photographs made
their parts as alike (0.61 to 0.67) as the same cat made them (0.61). Where faces have been looked
for, a part of an image with most of a face in it is left out of the comparison on both sides, as is
the whole of any image with a face in it; where they have not, what is written about a part is gone
by. Within an image the user named something in, the parts that count as showing it are those whose
descriptions speak of it by name or by the word for what it is ("cat"). An example with a person in
it is used only if the thing has none without.

The agent does the same through the `images` action (`image_id`, `name`, `what`; `wrong` to take a
name back). An image she recognised a thing in by herself is never used as an example of it, so one
mistake cannot grow into many; only images the user named are. People are not named this way: a
fingerprint says that two things look alike, not who someone is, and a face model for that is the
next stage. `image_name_min` was chosen from a dozen photographs and should be checked against your
own with `images similar ID --all`.

| Key | Default | Meaning |
|---|---|---|
| `image_fingerprints` | `false` | Make and use picture fingerprints |
| `image_fingerprint_host` | `http://127.0.0.1:8189` | Where `tools/fingerprint_server.py` listens |
| `image_fingerprint_min` | `0.5` | How alike (0 to 1) two pictures must be to be called alike |
| `image_names` | `true` | Recognise things the user has named (needs fingerprints) |
| `image_name_min` | `0.6` | How alike a part must be to a named thing to be offered to the vision model as it |

### Faces

A picture fingerprint says two things look alike; it cannot tell one person from another who looks
similar. A face model can. With one, the agent can know a particular person in an image. This is the
most personal thing the plugin does, so all of it is off until the user turns it on, and each part is
turned on separately.

| Key | Default | Meaning |
|---|---|---|
| `face_learn` | `none` | Whose faces may be learned: `none` (no face is looked for), `me` (only the user's; no other face is kept), `named` (people the user has named; no other face is kept), `often` (every face is kept, so that someone who keeps appearing can be noticed) |
| `face_ask_names` | `false` | With `often`: she may ask, once, who a person is who has appeared in `face_often_images` images |
| `face_name_unasked` | `false` | Whether she is told who is in an image when the user has not asked. Off: only when the message asks who someone is |
| `dream_image_people` | unset | Whose images a dream picture may be drawn from: `none`, `me` (every face in the image is the user's), `named` (every face is someone the user named), `anyone`. Unset follows the older `dream_image_use_people`, which is `false` |
| `dream_name_people` | `none` | Whether a dream is told who is in the images it draws on, so they can be in it by name: `none`, `me`, `named` |
| `face_min` | `0.40` | How alike two faces must be to be the same person |

```
hermes holonomic faces                              # what is allowed, whether the helper can do faces, who she knows
hermes holonomic faces learn named                  # none | me | named | often
hermes holonomic faces unasked on                   # may she say who is in a picture without being asked
hermes holonomic faces ask on                       # with 'often': may she ask who someone is
hermes holonomic faces show 37                      # the faces in image #37, numbered from the left
hermes holonomic faces name 37 Kayla --me           # that face is you
hermes holonomic faces name 66 Emma --face 2        # with several faces, say which
hermes holonomic faces not 66 Emma                  # that face is not Emma
hermes holonomic faces dream Emma no                # never in an image a dream is drawn from, nor named in one
hermes holonomic faces image 42 off                 # look for no faces in this image (a street full of strangers)
hermes holonomic faces people | often | scan | forget NAME | forget --all --yes
hermes holonomic dreams images --who named --name-people me
```

Saying who someone is when showing a picture ("this is me", "me and my daughter Emma") teaches her
too, within what `face_learn` allows; with several faces she needs to be told which is which, or it
is left for `faces name`. What a person looks like is taken only from faces the user spoke for: a
face she recognised herself is never an example. In `me` and `named` no fingerprint is kept of
anyone else's face. Tightening `face_learn` drops what the new rule does not allow the next time
images are gone through. `faces image ID off` sets one image apart: every face kept from it is dropped and
none is looked for in it again, whatever the general rule.

The helper server does the work and needs OpenCV for it, in the same Python:

```
<ComfyUI>\.venv\Scripts\python.exe -m pip install opencv-python-headless
```

On its next start it downloads two small models from the OpenCV model zoo (YuNet to find faces,
SFace to fingerprint them, 39 MB together). Without OpenCV everything else works and
`hermes holonomic faces` says faces are not available.

## Inspecting the store

```
hermes holonomic stats
hermes holonomic list -n 20
hermes holonomic recall "what is my name"
hermes holonomic show 37 40
hermes holonomic forget 52 --yes
```

`recall` prints every candidate with its scores and whether it would have been injected.

`hermes holonomic context` prints exactly what memory gave the agent for the most recent message:
recalled memories, what was seen in an attached image, what was recognised. When she does not act on
something, look there first to see whether she was told it.

## Backing up and restoring

    hermes holonomic backup                        write the whole store to one zip file
    hermes holonomic backup --label before-update  with a word added to its name
    hermes holonomic backup --list                 the backups there are
    hermes holonomic restore                       show what the newest backup would put back
    hermes holonomic restore --yes                 do it (with Hermes closed)
    hermes holonomic restore NAME.zip --yes        a particular one

A backup holds everything: memories, images, dream pictures, reference libraries and settings. It can be taken
while Hermes is running: each database is copied through SQLite's own backup, which gives a consistent copy of
a store that is being written to, and the copy is checked before the backup is kept. Backups go to
`holonomic-backups` in your Documents (`--to FOLDER`, or the `backup_dir` setting); the ten newest are kept
(`--keep N`, or `backup_keep`; 0 keeps them all).

A restore needs Hermes closed. It does not delete what is there: the present store is set aside beside it as
`holonomic.before-restore_<date>`, so a restore can itself be undone. Delete that folder yourself once you are
sure. A file that is not one of these backups, or whose store is damaged, is refused and nothing is changed.

## Testing

```
python scripts/run_tests.py          # engine tests; provider tests too if a Hermes checkout is found
python scripts/selftest.py --host http://YOUR-OLLAMA:11434
python scripts/bench.py
```

Provider tests need the Hermes source: set `HERMES_SRC` or keep a `hermes-agent` checkout next to
this folder.

## Using a different model

This plugin was developed and tuned against one setup. On anything very different, expect to
re-tune before trusting what it stores.

| Role | Model it was tuned with |
|---|---|
| Chat, reflection, dreaming, describing images | Gemma 4 through Ollama, 64k context, on a 16 GB Tesla V100 |
| Embeddings | `nomic-embed-text` through Ollama |
| Dream pictures | Z-Image-Turbo (and SDXL, Stable Diffusion 1.5) through ComfyUI |

**What carries over to any model.** The safeguards are rules in code, applied after the model
answers: facts about the user must cite something the user said; writing in an image is kept only
when two looks agree or the user stated it; a reply cut off mid-sentence is retried or trimmed; a
failed step falls back to something safe. A stronger model trips these less often; a weaker one
more.

**What was tuned to this model and may need changing.**

| Setting or behaviour | Why it is the way it is | With another model |
|---|---|---|
| `reflect_think`, `dream_image_choose_think` off | Reasoning ran past its allowance without answering | A stronger model may reason well; test with `--think` |
| `reflect_max_tokens`, `image_max_tokens` | Sized to this model's typical replies | May be too tight or too generous |
| `reflect_depth` 3 | A second pass that checks each item was needed | Depth 2 may be enough; a weaker model may need 1 |
| "Never use the double quotation mark" in prompts | This model ended its JSON early at a quote | Harmless elsewhere |
| Dream openings chosen in code | Every dream began in a corridor | May be unnecessary |
| `dream_temperature` 1.0, word limits | What produced dreams and not summaries here | Adjust to taste |
| The two-look rule for writing in images | This model invents sign text it cannot read | Discards some true text; a better vision model may not need it |
| `dream_image_strength`, steps, size | Right for Z-Image-Turbo; SDXL wanted a different strength | Re-tune for each image model |
| `dream_talk_similarity`, `min_score`, `score_band` | Measured with `nomic-embed-text` | Re-measure with a different embedding model |

**The embedding model cannot be swapped freely.** Stored memories are indexed by it, so changing it
means a new store. Chat, vision and image models can be changed at any time.

**A model that cannot see** cannot describe images or choose between dream pictures: set
`image_model` to one that can, or leave image memory off.

**How to check a new model before trusting it.** Back up the data folder, then compare against
what you know to be true:

```
hermes holonomic reflect now --dry-run       # are the facts things you actually said?
hermes holonomic sleep now --dry-run         # is the account of each conversation accurate? does the dream read as a dream?
hermes holonomic images redo ID              # on a photo you know well: is anything invented?
hermes holonomic dreams images --test "..."  # does the image generator draw what was asked?
```

The prompts themselves are in `reflect.py`, `sleep.py` and `images.py`. They were written by
watching this model fail and correcting for it, so wording that steers Gemma 4 may not steer
another family as well.

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
