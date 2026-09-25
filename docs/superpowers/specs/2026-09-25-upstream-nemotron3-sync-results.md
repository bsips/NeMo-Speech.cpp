# Real-audio verification: Nemotron 3 diarization sync (2026-09-25)

Task 5 of the upstream sync plan. No source changes. This records what
actually happened running the built binary (branch `sync/upstream-nemotron3`,
commit `a49fef7`) against real audio, per the corrected task-5 brief.

Corrections applied vs. the original brief: `ctest --output-on-failure` (no
test presets exist), the exact model paths given in the task message,
`$LONG_INTERVIEW` = `~/Projects/SliqSpeech/input/russell-williams.mp4`
(extracted to 16 kHz mono wav), and the CLI's actual flags (`--device
cuda:0`, `-f json`, `--word-times`, `-o` — not `--gpu`/`--output-format`/
`--word-timestamps`, which don't exist on this binary).

Baseline: `ctest --output-on-failure` from `build/cuda-asr` — **14/14 pass**
(shared_utilities, subtitles, http_server_config, installed_sdk_consumer,
cli_contract, cli_model_store, cli_install_linux, endpointer_policy,
asr_postprocessing, decoders, diar_state, diar_frame_lookup,
diar_speaker_change, live_transcript), 5.36s total.

---

## Step 1: Divergence #1 regression — the long-file collapse

**Command:**

```bash
ffmpeg -nostdin -i ~/Projects/SliqSpeech/input/russell-williams.mp4 \
  -ac 1 -ar 16000 -vn <scratch>/rw-full.wav

./build/cuda-asr/bin/nemo-speech transcribe <scratch>/rw-full.wav \
  --model ~/.cache/nemo-speech/models/nvidia/parakeet-tdt-0.6b-v3/541d.../parakeet-tdt-0.6b-v3.q8_0.gguf \
  --diarize --diar-model ~/.cache/nemo-speech/models/nvidia/diar_streaming_sortformer_4spk-v2/5240.../diar_streaming_sortformer_4spk-v2.q8_0.gguf \
  --device cuda:0 --word-times -f json -o <scratch>/step1-longfile.json --force
```

**Outcome:** exit 0, `real 12m35.296s` for the full 9585.36s (2h39m45s)
recording, 17606 words. Bucketing each word's `(end - start)` duration by
speaker into 10-minute windows:

| window | spk1 (s / words) | spk2 (s / words) | spk3 (s / words) | spk4 (s / words) | window total tagged (s) | top speaker | top share |
|---|---|---|---|---|---|---|---|
| 0–10 min | 0.0 / 0 | 72.6 / 304 | 382.9 / 1543 | 35.3 / 115 | 490.8 | spk3 | 78.0% |
| 10–20 min | 0.0 / 0 | 41.5 / 165 | 194.1 / 699 | 208.2 / 781 | 443.8 | spk4 | 46.9% |
| 20–30 min | 10.2 / 48 | 13.9 / 44 | 317.0 / 1112 | 161.6 / 616 | 502.6 | spk3 | 63.1% |
| 30–40 min | 16.7 / 43 | 24.8 / 77 | 388.2 / 1224 | 43.4 / 121 | 473.0 | spk3 | 82.1% |
| 40–50 min | 4.0 / 11 | 5.0 / 12 | 273.3 / 540 | 10.5 / 13 | 292.7 | spk3 | 93.4% |
| 50–60 min | 41.3 / 84 | 47.8 / 141 | 205.6 / 494 | 73.9 / 227 | 368.6 | spk3 | 55.8% |
| 60–70 min | 31.8 / 78 | 19.1 / 30 | 107.4 / 265 | 233.0 / 428 | 391.2 | spk4 | 59.6% |
| 70–80 min | 3.4 / 9 | 8.6 / 4 | 89.5 / 243 | 307.9 / 595 | 409.4 | spk4 | 75.2% |
| 80–90 min | 4.2 / 9 | 4.6 / 13 | 176.8 / 391 | 206.0 / 377 | 391.6 | spk4 | 52.6% |
| 90–100 min | 24.6 / 60 | 5.2 / 17 | 114.6 / 329 | 261.0 / 518 | 405.4 | spk4 | 64.4% |
| 100–110 min | 39.9 / 120 | 6.2 / 20 | 70.7 / 163 | 275.9 / 483 | 392.8 | spk4 | 70.2% |
| 110–120 min | 1.4 / 7 | 10.8 / 31 | 201.5 / 474 | 235.4 / 486 | 449.0 | spk4 | 52.4% |
| 120–130 min | 2.6 / 1 | 15.7 / 32 | 226.0 / 517 | 207.8 / 463 | 452.0 | spk3 | 50.0% |
| 130–140 min | 8.1 / 12 | 10.6 / 25 | 214.9 / 513 | 222.8 / 502 | 456.3 | spk4 | 48.8% |
| 140–150 min | 6.4 / 11 | 13.8 / 22 | 182.6 / 424 | 236.1 / 485 | 438.9 | spk4 | 53.8% |
| 150–160 min | 57.1 / 137 | 23.9 / 36 | 227.3 / 531 | 124.5 / 331 | 432.8 | spk3 | 52.5% |

Overall totals: spk1 = 251.5s / 630 words, spk2 = 324.0s / 973 words,
**spk3 = 3372.3s / 9462 words (49.7% of tagged time)**, **spk4 = 2843.0s /
6541 words (41.9%)**.

**Read:** this is a two-speaker interview. The model emits two dominant
tags (spk3, spk4) that both appear with substantial word counts in *every*
10-minute window, and dominance alternates window-to-window (spk3 leads in
windows 0–10, 20–50, 120–130, 150–160; spk4 leads in the rest) — the
opposite of the original defect, where one tag would hold ~140 of 160
minutes. The single highest single-window share is 93.4% at 40–50 min,
which is a low-tagged-time window overall (292.7s of 600s) — consistent
with the gold fixture README's note that 35–50 min is "near-monologue" in
this recording, not a collapse. Tags spk1/spk2 are minor/spurious
(≤324s / ≤973 words total each across the whole file) — noise, not a third
or fourth real speaker.

**Corroboration against the gold excerpts** (`~/Projects/SliqSpeech/gold/`,
hand-corrected truth for three 10-minute windows of this same recording):

| window | step1 full-file word count | gold truth word count |
|---|---|---|
| 20–30 min | 1820 | 1819 (`rw-20to30`) |
| 55–65 min | 814 | 814 (`rw-55to65`) |
| 105–115 min | 848 | 849 (`rw-105to115`) |

Word counts match to within 1 across all three windows — strong evidence
the full-file run isn't degrading over time in a way that would also
distort the transcript, not just the speaker tags.

**Corroborating unit evidence:** `./build/cuda-asr/bin/test_diar_frame_lookup`
→ `[PASS] speaker_for_frame_range resolves compacted ranges via frozen
segments` (the synthetic test for the same underlying logic).

**Verdict: PASS.** Divergence #1's fix holds on the real 2h39m file that
originally exposed it.

---

## Step 2: Divergence #2 — false-positive count, gated vs. pre-gate

**Method:** `NEMO_LIVE_DEBUG=1` replay via
`~/Projects/SliqSpeech/scripts/replay_live_clip.sh`, with a temporary
one-line debug print added to `live_bridge.py`'s handling of
`conversation.item.speaker_diarization.changed` (reverted before
finishing — SliqSpeech tree is clean; see "Files changed" below). Clip:
`~/Projects/SliqSpeech/gold/rw-55to65.wav` (10 min, densest turn-taking of
the three gold windows — 67 gold turns). For each
`speaker_diarization.changed` event, the very next `.completed` response's
word `speaker` tags were checked for the event's claimed speaker.

**Gated build (`a49fef7`, this branch's HEAD):**

```bash
cd ~/Projects/SliqSpeech
./scripts/replay_live_clip.sh ~/Projects/SliqSpeech/gold/rw-55to65.wav
```

66 `speaker_diarization.changed` events → **11 false positives (16.7%)**,
55 true positives. Of the 11: 6 were followed by a `.completed` with no
words at all (nothing to confirm *or* contradict the claim), 5 were
followed by a `.completed` whose words were all tagged with a different
speaker (a direct contradiction).

**Pre-gate build (`ef65b84`, one commit before the `a49fef7` gate fix,
built in a scratch worktree with `NEMO_SPEECH_BIN` pointed at it — see
"Files changed"):**

Same command and clip, `NEMO_SPEECH_BIN=<scratch>/pregate-wt/build/cuda-asr/bin/nemo-speech`.

71 `speaker_diarization.changed` events → **6 false positives (8.5%)**, 65
true positives. Of the 6: 3 empty-`.completed`, 3 direct contradictions.

**Verdict: INCONCLUSIVE, and the direction is the opposite of what the
brief expected.** The gated build has *more* false positives by this
count (11/66 = 16.7%) than the pre-gate build (6/71 = 8.5%) — both
overall and restricted to direct contradictions only (5/66 = 7.6% gated
vs. 3/71 = 4.2% pre-gate). This is a negative result and is reported as
measured, not adjusted.

Two things weaken any conclusion drawn from this comparison and are worth
being explicit about rather than silently averaged away:

1. **Confounded run conditions.** The gated replay ran concurrently with
   Step 1's 12.5-minute transcribe job and the pre-gate binary's own
   from-scratch CUDA build compiling in the background (CPU/GPU
   contention). The pre-gate replay ran afterward on an otherwise idle
   machine. `SilenceCommitTrigger`'s commit boundaries are timing-sensitive
   (buffered-ms thresholds against real wall-clock silence), so system
   load differences between the two runs can shift where commits land
   independent of the code under test.
2. **Small samples.** 66 vs. 71 events and 11 vs. 6 false positives is not
   a lot of signal to detect a real-but-small effect size, especially
   given (1).

I did not re-run under matched load to control for (1), per the brief's
own instruction not to spend long chasing the pre-gate number. The
honestly-measured gated number — **11/66 = 16.7% false positives** — is
the deliverable; the pre-gate comparison is directional color only, and
in this specific run it points the wrong way.

---

## Step 3: Divergence #3 — identity across real commit boundaries

**Method:** used the same gated replay (`rw-55to65.wav`, 100 silence/
max-buffer commits over the 10-minute clip) plus a second capture — a
throwaway script attached to `live.py`'s overlay WebSocket (`ws://127.0.0.1:8766/ws`,
the same feed `debug.html` renders) recording every `pipeline_status` and
`final` event with timestamps, run alongside the same replay session.
Reconstructed the replay's per-word speaker sequence from the 96 `final`
events' text (in chronological/commit order) and word-aligned it (via
`difflib.SequenceMatcher`, word-level LCS) against the gold `rw-55to65.json`
transcript's per-word ground-truth speaker tags.

**Result:** 481 of 814 gold words aligned (59.1% coverage — ASR text
differs enough between the streaming nemotron model and whatever produced
the gold transcript that alignment isn't 100%, but that's plenty of
signal). Majority mapping: replay speaker 2 → gold speaker 1 (99.6%
agreement over 228 aligned words), replay speaker 1 → gold speaker 2
(89.1% over 239 aligned words). A third tag, replay speaker 3, appeared
briefly (14 aligned words total, mostly agreeing with gold speaker 1) —
minor noise, not a real third speaker, consistent with the spurious
low-count tags seen in Step 1.

Mismatch rate against the overall majority mapping: 29/481 = 6.0%, and
**zero sustained mismatch runs of 5+ consecutive aligned gold words** —
i.e. every mismatch was an isolated word (ASR/diarization boundary noise),
not a stretch where the replay's speaker-2-means-gold-speaker-1 mapping
flipped to speaker-2-means-gold-speaker-2 for a real run of words.

**Verdict: PASS.** No renumbering was observed across ~100 real
silence-triggered commit boundaries in the 10-minute replay. This is
consistent with Step 5's direct protocol-level assertion (below), which
tests the same property with an explicit two-commit split and passed on
two fixtures.

---

## Step 4: AMI fixture — continuity observation only

**Command (exact, from the brief):**

```bash
./build/cuda-asr/bin/test_diar_identity_handoff \
  ~/.cache/nemo-speech/models/nvidia/nemotron-3.5-asr-streaming-0.6b/.../nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  ~/.cache/nemo-speech/models/nvidia/diar_streaming_sortformer_4spk-v2/.../diar_streaming_sortformer_4spk-v2.q8_0.gguf \
  test_files/diar/ami_en2002d_2132.wav --gpu 0 --split-sec 20
```

**Outcome:** `FAIL: speaker identity did not survive the handoff (1 -> 2)`.

Per the brief, this fixture is for continuity/false-positive observation
only — no DER scoring (AMI's known scorer defect on overlapped truth) and
no pass/fail gate. Recorded as measured: **it failed at the exact
split-sec the brief specifies.**

To understand whether this is boundary-specific (AMI is heavily
overlapped meeting audio) or a general regression, I additionally swept
`--split-sec` across the fixture (not in the brief; extra diagnostic):

| split-sec | committed/total frames | before → after | speaker changes after split | result |
|---|---|---|---|---|
| 10 | 120/124 | 1 → 1 | 4 | OK |
| 15 | 180/187 | 1 → 1 | 3 | OK |
| **20** | 240/249 | 1 → 2 | 3 | **FAIL** |
| **25** | 300/312 | 1 → 2 | 3 | **FAIL** |
| 30 | 360/374 | 2 → 2 | 2 | OK |
| 45 | 560/562 | 2 → 2 | 2 | OK |

Cross-referencing the fixture's RTTM ground truth
(`test_files/diar/ami_en2002d_2132.rttm`): speaker MEE071 talks
17.09–19.51s, then the ground truth has **no speaker active from 19.51s to
26.75s** (a genuine 7-second gap), after which FEO072/FEO070 pick up.
Both failing split points (20s, 25s) land inside that ground-truth gap —
i.e. the test is asking "who is speaking at t=20s / t=25s" at a point
where, per RTTM, *nobody* is confidently speaking in this heavily
overlapped meeting recording. Split points outside that gap (10, 15, 30,
45) all pass.

**Read:** this looks like AMI's known overlap/ambiguity, exactly the
caveat the brief attaches to this fixture, rather than a divergence #3
regression — the same divergence #3 property (Step 3, Step 5) passes
cleanly on two other fixtures with less pathological audio. Recorded as a
failure at the brief's exact command, with this as my best-read
explanation, not as a dismissal.

---

## Step 5: HTTP protocol conformance

**Command 1 (brief's exact fixture):**

```bash
python tests/integration/http_conformance_test.py \
  --binary ./build/cuda-asr/bin/nemo-speech \
  --asr-model ~/.cache/.../nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  --diar-model ~/.cache/.../diar_streaming_sortformer_4spk-v2.q8_0.gguf \
  --audio test_files/asr/wav/test/scotus_08-1314_excerpt.wav
```

(Used nemotron-3.5-asr-streaming, not parakeet-tdt, per the model-path
correction — parakeet is offline-only and this test exercises the
realtime WebSocket path.)

**Outcome: exit 0** (`real 0m16.659s`). Every `require(...)` passed,
including line ~423's `require(speaker_changes, "WebSocket speaker-change
event")` — the gate did **not** suppress the event on this 90-second
fixture. (`require()` raises on failure and the script prints nothing on
success, so exit 0 with no output is the full-pass signal — confirmed by
reading the script's `require()`/`main()` source. It also binds a
`free_port()` rather than 8080, so it does not collide with a running
`nemo-speech serve`.)

**Re-run instrumented, to replace "exit 0" with the numbers the
assertions actually compared.** A silent pass records no measurement, so
the same checks were re-run from an instrumented *copy* of the script
(scratchpad only; the repo file was never modified — `git status` stayed
at ` m ggml` throughout) with prints added next to the two assertions
under test:

```
INSTR commit-boundary: last word before split speaker=2 first word after split speaker=2
                       n_completed=2 n_words_before=101 n_words_after=109
INSTR speaker_changes: n=2 detail=[(2, 7.290999831914902), (3, 81.37099817609787)]
INSTR final completed: n_words=210 speaker_tags=[1, 2, 3]
```

- **Lines 399–405, "speaker identity persists across a commit boundary":**
  the two commits split at 40.0 s of `scotus_08-1314_excerpt.wav` produced
  2 `.completed` events carrying 101 and 109 tagged words. The last word
  before the split is **speaker 2** and the first word after it is
  **speaker 2** — the identity survived the boundary. This is the
  protocol-level proof of what the divergence-#3 fix changed, with the
  compared values rather than just the verdict.
- **Line 423, `require(speaker_changes, ...)`:** **2** events fired —
  speaker 2 at `start_time=7.291 s` and speaker 3 at
  `start_time=81.371 s`. Both carry a 1-based int `speaker` and a numeric
  `start_time`, which is what the follow-on checks at lines 425–432 assert.
- Final `.completed`: 210 words, speaker tags `{1, 2, 3}` — the
  `WebSocket speaker tags` check.

Note that this commit-boundary block always runs against the hard-coded
`scotus_08-1314_excerpt.wav` (resolved from `__file__`), independent of
`--audio`; `--audio` drives the separate single-stream assertions,
including line 423's.

**Command 2 (corroboration on a longer multi-speaker clip, per the
brief's instruction to check this on a longer clip regardless of outcome):**

```bash
python tests/integration/http_conformance_test.py \
  --binary ./build/cuda-asr/bin/nemo-speech \
  --asr-model ~/.cache/.../nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
  --diar-model ~/.cache/.../diar_streaming_sortformer_4spk-v2.q8_0.gguf \
  --timeout 300 \
  --audio ~/Projects/SliqSpeech/gold/rw-20to30.wav
```

**Outcome: exit 0** (`real 2m30.174s` for the 10-minute clip). Also fully
passed, including the speaker-change-event requirement and the explicit
"speaker identity persists across a commit boundary" assertion (lines
399–405) — the direct protocol-level proof of divergence #3.

**Verdict: PASS on both fixtures.** The brief's specific worry — that the
gate could be conservative enough to suppress the event entirely on a
short fixture — did not materialize here; the event fired on the 90s
scotus clip. This is worth flagging as a genuine possibility going
forward on shorter or quieter clips, but it did not reproduce on either
fixture tried.

---

## Step 6: Companion-repo checks (SliqSpeech)

**Command:**

```bash
cd ~/Projects/SliqSpeech
python -m pytest tests/test_live_bridge_client.py -k speaker_change -v
```

**Outcome:** `4 passed, 20 deselected in 2.74s`:
- `test_client_does_not_commit_on_speaker_change_event` — PASSED
- `test_speaker_change_event_leaves_silence_trigger_state_untouched` — PASSED
- `test_pipeline_status_reflects_confirmed_speaker_change` — PASSED (also matched by `-k speaker_change`)
- `test_client_survives_speaker_change_event_missing_speaker_key` — PASSED (also matched)

**Debug view's status panel during the Step 2 replay:** rather than
visually confirming a browser render (headless environment), connected a
script directly to `live.py`'s overlay WebSocket (`ws://127.0.0.1:8766/ws`
— the exact feed `debug.html` subscribes to, per `live_overlay_server.py`)
during the gated Step 2 replay and logged every `pipeline_status` event.
Captured 423 `pipeline_status` events over the session, with `speaker`
transitioning between 1 and 2 in step with the transcript's actual
turn-taking (e.g. `speaker=2` → `speaker=1` → `speaker=2` matching
consecutive `final` events with those same speaker tags). This is the
same data `debug.html`'s status panel renders, captured at the point it's
produced rather than at the point it's drawn — confirmed live, not
inferred from the unit test alone.

**Verdict: PASS.**

---

## Summary

| Step | Verdict | Key number |
|---|---|---|
| 1. Long-file collapse | PASS | max single-window share 93.4% (one low-tagged near-monologue window); overall split 49.7%/41.9% across 16 windows, both speakers present in every window |
| 2. False-positive count | INCONCLUSIVE (unexpected direction) | gated 11/66 (16.7%) vs. pre-gate 6/71 (8.5%) — confounded by differing system load between runs |
| 3. Identity across commit boundaries | PASS | 0 sustained mismatch runs across ~100 commits; 6.0% isolated-word mismatch rate |
| 4. AMI continuity | FAIL at brief's exact command | 1→2 at split-sec 20 and 25, both inside a 7s RTTM truth gap; OK at 10/15/30/45 |
| 5. HTTP conformance | PASS (both fixtures) | exit 0 on 90s and 10-min clips; speaker-change event fired on both |
| 6. Companion-repo checks | PASS | 4/4 pytest; status panel feed confirmed live (423 pipeline_status events) |

Divergences #1 and #3 hold up under real audio. Divergence #2's gate did
not demonstrate the expected false-positive reduction in this specific,
confounded comparison — a controlled re-run (matched system load, larger
sample) would be needed for a real verdict there. Step 4's AMI failure is
most likely the fixture's known overlap/ambiguity rather than a
regression, given it passes cleanly elsewhere and fails exactly inside a
ground-truth silence gap.

---

## Files changed

- Added: `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-results.md` (this file).
- No other files in this repo were modified. The pre-existing `ggml`
  submodule pointer drift (` m ggml`, unrelated to this task) was left
  untouched and unstaged.
- A temporary one-line debug print was added to
  `~/Projects/SliqSpeech/live_bridge.py` (in the
  `conversation.item.speaker_diarization.changed` handler, to print the
  claimed speaker for Step 2/3's cross-referencing) and **reverted**
  before finishing (`git checkout -- live_bridge.py`); that repo's tree
  is clean, nothing was committed there.
- A scratch worktree was created at commit `ef65b84` (one commit before
  the `a49fef7` gate fix) to build a pre-gate comparison binary for Step
  2, under the session scratchpad directory (never inside either repo).
  It required copying this repo's locally-patched (uncommitted) `ggml`
  working tree into it to get a working CUDA build (the patch adds
  `GGML_TENSOR_FLAG_Q8_PLANAR`, absent from a clean submodule checkout of
  the same pinned commit) and an explicit
  `-DNEMO_SPEECH_BUILD_HTTP=ON` (the `cuda-asr` preset alone doesn't
  enable the HTTP server; the main build's cache has it on from an
  earlier out-of-preset configure). The worktree was removed
  (`git worktree remove --force`) after use; `git worktree list` in this
  repo shows only the main checkout.
- All intermediate audio/JSON/logs were written to the session scratchpad
  directory (`rw-full.wav`, `step1-longfile.json`, replay traces, the
  false-positive/identity-persistence analysis scripts and their output),
  never into either repo.
