# Chatterbox Turbo YouTube Narration Suite Plan

## Goal

Build a Gradio-based narration production suite for generating long-form YouTube narration using a cloned voice with Chatterbox Turbo.

The target workflow is:

1. Paste a long narration script.
2. Split it into stable TTS-sized chunks.
3. Generate chunks using the cloned voice.
4. Validate each chunk using Whisper/text checks and voice/audio quality checks.
5. Edit or regenerate only bad chunks.
6. Merge approved chunks into a final narration file with a user-specified filename.
7. Persist the full session so Kaggle disconnects or runtime limits do not lose progress.

No implementation has been done yet. This document captures the current context and recommended build plan.

## Current target app

Primary file:

- `gradio_tts_turbo_app.py`

Current behavior:

- Loads one `ChatterboxTurboTTS` model into Gradio state.
- Uses only one device:

  ```python
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  ```

- On Kaggle T4 x2, this means the app will likely use only `cuda:0`.
- Generates one audio result from one text box.
- Uses `default_concurrency_limit=1`.
- UI label says `max chars 300`, but there is no long-text chunking pipeline.
- There is no session persistence, chunk manifest, Whisper validation, per-chunk editing, result filename field, or merge/finalization flow.

Relevant model file:

- `src/chatterbox/tts_turbo.py`

Important source behavior:

- `ChatterboxTurboTTS.generate()` calls the tokenizer with `truncation=True`, so long text can be silently truncated if passed directly.
- The generated waveform is returned as a CPU tensor after watermarking.
- `audio_prompt_path` triggers `prepare_conditionals()` each time it is passed.
- Reference audio must be longer than 5 seconds according to the source assertion.
- Turbo warns that `cfg_weight`, `exaggeration`, and `min_p` are not supported and will be ignored.

## Chatterbox Turbo context

Official/current references checked:

- Hugging Face model card: https://huggingface.co/ResembleAI/chatterbox-turbo
- Resemble AI Chatterbox Turbo page: https://www.resemble.ai/learn/models/chatterbox-turbo
- General Chatterbox page: https://www.resemble.ai/learn/models/chatterbox

Relevant facts:

- Chatterbox Turbo is English-focused.
- It is a 350M parameter model optimized for low-latency generation.
- It supports zero-shot voice cloning from roughly 5 seconds of reference audio.
- It is suitable for narration and creative workflows, not only real-time agents.
- It supports paralinguistic tags such as:
  - `[laugh]`
  - `[chuckle]`
  - `[sigh]`
  - `[cough]`
  - `[gasp]`
  - `[shush]`
  - `[groan]`
  - `[sniff]`
  - `[clear throat]`
- It applies PerTh watermarking to generated audio.

Turbo-specific UI implication:

- Keep these generation controls:
  - temperature
  - top_p
  - top_k
  - repetition_penalty
  - reference audio
  - loudness normalization
- Remove, hide, or clearly de-emphasize these for Turbo:
  - exaggeration
  - CFG
  - min_p

## Model selection requirement

The Gradio app should support selecting the generation backend per narration session.

Recommended UI options:

- `Turbo`
- `Nano`
- `Original`

Backend mapping:

```text
Turbo    -> ChatterboxTurboTTS.from_pretrained(device, nano=False)
Nano     -> ChatterboxTurboTTS.from_pretrained(device, nano=True)
Original -> ChatterboxTTS.from_pretrained(device)
```

Recommended behavior:

- Store the selected model in `session.json`.
- Treat model selection as session-level state.
- Avoid mixing models inside one final narration unless the user explicitly accepts the inconsistency.
- If a user changes the model after chunks have already been generated, mark the session as mixed-model or require affected chunks to be regenerated.
- Use a small adapter layer so chunk generation calls one internal interface regardless of selected backend.

Model-specific settings:

- Turbo/Nano:
  - `temperature`
  - `top_p`
  - `top_k`
  - `repetition_penalty`
  - `norm_loudness`
  - reference audio
- Original:
  - `temperature`
  - `exaggeration`
  - `cfg_weight`
  - `min_p`
  - `top_p`
  - `repetition_penalty`
  - reference audio

For the first implementation pass, the app can expose a common settings panel and only pass supported arguments to the selected backend.

## Kaggle T4 x2 context

References checked:

- Kaggle GPU usage docs: https://www.kaggle.com/docs/efficient-gpu-usage
- Kaggle CLI accelerator docs: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
- NVIDIA T4 specs: https://www.nvidia.cn/data-center/tesla-t4/
- Google Cloud T4 summary: https://cloud.google.com/blog/products/ai-machine-learning/nvidia-tesla-t4-gpus-now-available-in-beta

Relevant assumptions:

- Kaggle offers a T4 x2 accelerator option.
- Typical notebook constraints are around:
  - 4 CPU cores
  - about 29-30 GB system RAM
  - 12 hour runtime sessions
  - persistent `/kaggle/working` storage
- Each NVIDIA T4 has 16 GB VRAM, though `nvidia-smi` may show closer to about 15 GB usable.

Implementation implication:

- Do not rely on one model object with `cuda`.
- Load one model instance per GPU:

  ```text
  cuda:0 -> ChatterboxTurboTTS worker 0
  cuda:1 -> ChatterboxTurboTTS worker 1
  ```

- Dispatch chunks across workers.
- Persist every generated chunk immediately to disk.

## Recommended suite design

### 1. Project/session workspace

Each narration project should have a session directory.

On Kaggle:

```text
/kaggle/working/chatterbox_sessions/{session_id}/
```

Local fallback:

```text
./outputs/chatterbox_sessions/{session_id}/
```

Recommended layout:

```text
{session_dir}/
  session.json
  reference/
    reference.wav
  chunks/
    0001.txt
    0001.wav
    0002.txt
    0002.wav
  validation/
    0001.json
    0002.json
  final/
    result_name.wav
```

`session.json` should store:

- session ID
- project name
- result filename
- original full script
- chunk list
- edited chunk text
- generation params
- reference audio path
- per-chunk output path
- per-chunk status
- per-chunk validation summary
- final merged output path

Suggested chunk statuses:

- `pending`
- `generating`
- `generated`
- `validating`
- `needs_review`
- `approved`
- `failed`
- `excluded`

### 2. Script/chunking system

Chunking must happen before calling `ChatterboxTurboTTS.generate()`.

Recommended behavior:

- Split by paragraphs and sentence boundaries.
- Target about 220-350 characters per chunk.
- Avoid splitting inside brackets.
- Preserve paralinguistic tags.
- Keep punctuation.
- Avoid overlap by default because overlap would duplicate spoken words.
- Allow manual editing after auto-splitting.

Important:

- Whisper validation must handle tags specially because Whisper may ignore non-speech events or transcribe them inconsistently.
- Text comparison should remove or normalize bracketed tags before computing strict similarity, while still preserving the tags for TTS generation.

### 3. Voice/reference setup

UI should support:

- Upload reference audio.
- Record reference audio.
- Save/copy reference audio into the session directory.

Reference validation should check:

- duration is greater than 5 seconds
- clipping
- silence ratio
- loudness
- file readability

For YouTube narration, stable delivery is more important than exaggerated emotion. Default settings should prioritize consistency.

### 4. Generation system

Required actions:

- Generate all chunks.
- Generate selected chunk.
- Regenerate failed chunks.
- Regenerate selected chunk.

Parallel T4 x2 design:

- Detect GPU count with `torch.cuda.device_count()`.
- If two GPUs exist, initialize one TTS worker per GPU.
- Each worker owns:
  - a device string, e.g. `cuda:0`
  - a `ChatterboxTurboTTS` instance
  - its prepared conditionals or reference setup
  - a lock/serial queue
- Dispatch chunk jobs round-robin or through a shared work queue.
- Save each chunk immediately after it is generated.
- Update the session manifest after each chunk.

Avoid:

- Sharing one model object across GPUs.
- Keeping all generated chunks only in memory.
- Passing the full script to `generate()`.

### 5. Whisper text validation

Goal:

```text
intended chunk text -> generated wav -> Whisper transcript -> similarity check
```

Validation should flag:

- missing phrases
- hallucinated extra words
- repeated words
- empty transcript
- very low similarity
- obvious truncation

Candidate ASR options:

- `faster-whisper`
  - likely better performance
  - needs Kaggle compatibility testing
- `openai-whisper`
  - simpler conceptually
  - may be slower/heavier
- `transformers` Whisper pipeline
  - repo already depends on `transformers`
  - still requires model download and runtime validation

Recommended initial default:

- Start with a small or medium Whisper model.
- Make validation optional.
- Store transcript and score in `validation/{chunk_id}.json`.

Important:

- Whisper should not run at the same time as both TTS workers if VRAM becomes tight.
- If VRAM pressure is high, use one GPU for validation after generation batches, or use a smaller ASR model.

### 6. Audio quality validation

Speaker-identity validation is intentionally out of scope for this suite. Chatterbox already uses its own voice conditioning during generation, and an additional speaker-similarity score would add complexity without being necessary for the current YouTube narration workflow.

The audio-quality validator should check:

- silence at start/end
- clipping
- duration sanity
- loudness
- abnormal near-zero waveform

### 7. Chunk review UI

The user must be able to fix one bad sentence without regenerating the entire narration.

Recommended UI elements:

- Chunk table/list:
  - index
  - text preview
  - status
  - validation score
  - transcript preview
  - audio path/status
- Selected chunk editor:
  - full editable text
  - generated audio player
  - Whisper transcript
  - validation details
- Buttons:
  - Generate all
  - Generate selected
  - Regenerate selected
  - Regenerate failed
  - Approve selected
  - Exclude selected
  - Save session

Use a selected-chunk editor rather than relying on a large fully-editable Gradio table for v1. It will be more robust.

### 8. Result filename and final merge

Add a result filename input.

Filename validation:

- Strip path separators.
- Allow only safe filename characters.
- Force `.wav` unless later adding explicit format selection.
- Prevent writing outside the session final directory.

Merge/finalize button should:

- Check that all non-excluded chunks have generated audio.
- Warn or block if chunks are still `failed` or `needs_review`.
- Concatenate chunks in order.
- Insert configurable silence gaps, e.g. 150-350 ms.
- Optionally normalize final loudness.
- Save final `.wav`.
- Return final audio and downloadable file.

Recommended YouTube defaults:

- Final format: `.wav`
- Gap between chunks: 200 ms
- Loudness normalization: enabled
- Manual approval required only for failed/flagged chunks

## Recommended implementation phases

### Phase 1: persistence and chunk pipeline

Implement:

- session directory creation
- `session.json`
- safe result filename input
- long-text chunker
- chunk table/list
- selected chunk editor
- sequential chunk generation
- immediate chunk saving
- merge/finalize button

This phase proves the core narration workflow.

### Phase 2: dual-GPU generation

Implement:

- GPU detection
- one TTS worker per GPU
- chunk dispatch queue
- progress updates
- per-worker error handling

This phase uses Kaggle T4 x2 effectively.

Initial implementation status:

- Added GPU detection through `torch.cuda.device_count()`.
- Added a process-global model adapter cache keyed by model and device.
- Added optional multi-GPU batch generation for `Generate All`.
- Added UI controls for enabling/disabling parallel generation and setting max GPU workers.
- Added a chunk table `Device` column so Kaggle runs can confirm use of `cuda:0` and `cuda:1`.
- Kept `Generate Selected` on the primary device for predictable single-chunk edits.

### Phase 3: Whisper validation

Implement:

- ASR model loading
- chunk transcription
- text normalization
- similarity scoring
- validation JSON saving
- needs-review flags
- regenerate failed chunks action

Initial implementation status:

- Added optional `faster-whisper` loading with CPU, `cuda:0`, and `cuda:1` device choices.
- Added Whisper transcription for selected chunks and all generated chunks.
- Added tag-aware text normalization so Chatterbox event tags do not count as missing speech.
- Added SequenceMatcher scoring with missing-word and extra-word diagnostics.
- Added atomic validation reports under `validation/{chunk_id}.json`.
- Added per-chunk validation state, score, transcript, and review details to the chunk table.
- Added `Regenerate Chunks Needing Review`, followed by automatic revalidation.

Kaggle setup for this phase:

```bash
pip install faster-whisper
```

Use `small.en` on a T4 when quality is more important, or `base.en`/`tiny.en` when VRAM or runtime is constrained. Keep Whisper on one GPU and run validation after TTS generation if loading both models causes CUDA out-of-memory errors.

### Phase 4: audio quality validation

Implement:

- silence/clipping/loudness/duration checks
- review UI details

Initial implementation status:

- Added optional selected-chunk and all-chunk audio checks using `torchaudio` and NumPy.
- Added leading/trailing silence, clipping, duration, peak, RMS, and near-silent detection.
- Added configurable thresholds for each check.
- Added per-chunk audio quality status and combined validation reports under `validation/{chunk_id}.json`.
- Added audio-quality review details to the chunk table and selected-chunk panel.

### Phase 5: polish

Implement:

- resume existing session
- project picker
- export validation report
- optional MP3 export
- better progress display
- settings presets for narration styles

Initial implementation status:

- Added saved-session discovery and a refreshable session picker.
- Added one-click loading for previously saved sessions.
- Added channel-specific presets based on the mechanism-first mental-models editorial identity.
- Added downloadable JSON validation reports with per-chunk summaries.
- Added optional MP3 export through ffmpeg during finalization.
- Added live progress updates for batch generation and validation operations.

Remaining polish candidates:

- project metadata and settings presets stored directly in the session manifest

## Engineering risks and notes

- Long text must never be passed directly to Turbo generation.
- Turbo ignores some controls from the original Chatterbox model; the UI should reflect this.
- Whisper validation must account for paralinguistic tags.
- Loading TTS and Whisper on both T4 GPUs may exceed practical VRAM, so validation should be configurable.
- Kaggle runtime limits make persistence mandatory.
- Gradio concurrency must be managed carefully so multiple user clicks do not corrupt the same session.
- File writes should be atomic where possible: write temp file first, then rename.

## Minimal v1 acceptance criteria

The first useful version should support:

- paste long script
- auto-split into chunks
- edit selected chunk
- generate chunks sequentially
- save chunks to session directory
- regenerate selected chunk
- enter result filename
- merge generated chunks into one `.wav`
- resume from `session.json`

After that, add:

- T4 x2 parallel generation
- Whisper validation
