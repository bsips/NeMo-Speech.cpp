# Fork changes

This fork (`bsips/NeMo-Speech.cpp`) tracks `NVIDIA/NeMo-Speech.cpp`. This file
logs every place our `main` has diverged from upstream, in order, so a future
rebase/merge from upstream knows what to reconcile.

Companion project: [`NeMoSpeech`](https://github.com/bsips/NeMoSpeech) drives
this project's `nemo-speech` CLI and is where this fork's fixes were
discovered from real-world use. See its `docs/superpowers/` plans/specs and
the linked Obsidian vault notes for the user-facing side of each story.

## Divergences

### 2026-09-10 — Combined-mode diarization collapse on long audio

- **Upstream base:** `a5b6953` (`main`, matches the `v0.1.0` release tag's
  code for the affected files — no drift between them at time of fix).
- **PR:** [bsips/NeMo-Speech.cpp#1](https://github.com/bsips/NeMo-Speech.cpp/pull/1),
  merged as `202c8e6`.
- **Files:** `src/asr/diar/diar_pipeline.{h,cpp}`,
  `tests/cpp/asr/test_diar_frame_lookup.cpp` (new),
  `tests/cpp/asr/CMakeLists.txt`.
- **What:** `DiarStream::speaker_for_frames()` clamped any word-timestamp
  query predating the retained live probability window (`probs_base_`) to
  the window's first frame — correct only when every query lands "near the
  frontier" (soon after the audio is processed), as in streaming/interactive
  use. `Recognizer::recognize()` (the whole-file path behind `transcribe
  --diarize`) tags every word in a single pass at the very end instead,
  silently violating that assumption for any file long enough to trigger
  timeline compaction (~20 min default). Result: combined-mode diarization
  collapsed onto one dominant speaker for most of a long recording, while
  the standalone `diarize` command (whose `segments()` already folds
  `frozen_segs_` back in) was unaffected.
- **Fix:** extracted the frame-range resolution into
  `speaker_for_frame_range()`; a query starting before `probs_base_` now
  resolves against `frozen_segs_` (nearest segment by time if it falls in a
  gap) instead of clamping into the live window.
- **Verified against:** a real 2h39m two-speaker interview that previously
  collapsed onto one speaker for ~140 of 160 minutes; after the fix, both
  speakers alternate proportionally across every 10-minute window of the
  full file. Full detail and the original bug report live in
  `NeMoSpeech`'s project notes (see companion link above).
- **Upstreaming status:** not yet proposed to `NVIDIA/NeMo-Speech.cpp`. This
  fork stays ahead of upstream on this fix until/unless that happens.
- **Rebased onto:** `97a15af` on 2026-09-25 (see the 2026-09-25 sync entry).

### 2026-09-13 — Live speaker-change event on the realtime WebSocket

- **Upstream base:** `23a06766d248c4d208f17b4ca3bf50ce2d01b131` (`main` at
  the commit this branch forked from).
- **PR:** [bsips/NeMo-Speech.cpp#2](https://github.com/bsips/NeMo-Speech.cpp/pull/2),
  merged as `b376bf0`.
- **Files:** `src/asr/diar/diar_pipeline.{h,cpp}`, `src/asr/recognizer.{h,cpp}`,
  `server/http/http_server.cpp`, `tests/cpp/asr/test_diar_speaker_change.cpp` (new),
  `tests/cpp/asr/test_diar_recognizer.cpp`, `tests/cpp/asr/CMakeLists.txt`,
  `tests/integration/http_conformance_test.py`,
  `docs/superpowers/specs/2026-09-13-live-speaker-change-event-design.md` (new),
  `docs/superpowers/plans/2026-09-13-live-speaker-change-event.md` (new).
- **What:** `/v1/audio/transcriptions/realtime` only reported diarization
  results as word-level speaker tags on `.completed` events -- i.e. only once
  a client sent `input_audio_buffer.commit`. `NemoSpeech`'s client-side
  silence-based segmentation could therefore go up to 15s without
  finalizing during continuous speech, since there was no way to know a
  speaker change had happened sooner. Added a new, additive
  `conversation.item.speaker_diarization.changed` event, built on
  `DiarStream::segments()`'s existing incremental, hysteresis-cleaned
  segment tracking, so a client can commit immediately on a real speaker
  change instead.
- **Fix:** see `docs/superpowers/specs/2026-09-13-live-speaker-change-event-design.md`
  for the full design.
- **Verified against:** the pure `detect_speaker_change`/`SpeakerChangeTracker`
  logic is unit-tested end to end in
  `tests/cpp/asr/test_diar_speaker_change.cpp` (synthetic segments, no model
  needed). The event's JSON schema, `--diar-model` gating, and the critical
  0-based-to-1-based `+1` conversion were verified end-to-end against a real
  running server (observed `"speaker": N` correctly on the wire). A genuine
  mid-stream speaker transition was **not** verified against real audio,
  because no multi-speaker fixture is available in either this repo or the
  companion `NeMoSpeech` project (confirmed via md5sum: all committed wavs
  are byte-identical duplicates).
- **Upstreaming status:** not yet proposed to `NVIDIA/NeMo-Speech.cpp`.
- **Rebased onto:** `97a15af` on 2026-09-25 (see the 2026-09-25 sync entry).

### 2026-09-13 — Persistent diarizer identity across realtime commits

- **Upstream base:** `86e555d728539fc48939fdc014abde21925e2260`
- **PR:** [bsips/NeMo-Speech.cpp#3](https://github.com/bsips/NeMo-Speech.cpp/pull/3)
- **Files:** `src/asr/diar/diar_pipeline.h`, `src/asr/recognizer.{h,cpp}`,
  `server/http/http_server.cpp`, `test_files/asr/wav/test/scotus_08-1314_excerpt.wav` (new),
  `tests/cpp/asr/test_diar_identity_handoff.cpp` (new), `tests/cpp/asr/CMakeLists.txt`,
  `tests/integration/http_conformance_test.py`,
  `docs/superpowers/specs/2026-09-13-persistent-diarizer-identity-design.md`,
  `docs/superpowers/plans/2026-09-13-persistent-diarizer-identity.md`.
- **What:** the realtime WS handler destroyed the whole `RecognitionStream`
  -- including the diarizer's speaker-embedding cache -- on every
  `input_audio_buffer.commit`, so a speaker number had no relationship
  across commits. Rare to notice when commits were infrequent
  (silence-based, ~every 15s); made constantly visible by the recent
  speaker-change-commit feature (`#2`), which commits on every genuine
  turn. Verified the underlying model was never the problem: one real
  90s multi-speaker recording, processed as a single continuous stream,
  diarized consistently under both `streaming` and `offline` presets.
- **Fix:** see `docs/superpowers/specs/2026-09-13-persistent-diarizer-identity-design.md`
  for the full design, including a second bug found during planning
  (word/diarizer clock desync across a handoff) and its fix.
- **Verified against:** a new end-to-end test
  (`test_diar_identity_handoff`) proving speaker identity survives a
  commit boundary, using a real public-domain multi-speaker fixture
  (U.S. Supreme Court oral argument audio, case 08-1314, sourced from
  archive.org's official `SCOTUSOralArugments` collection); extended
  `tests/integration/http_conformance_test.py` proving the same property
  at the WebSocket protocol level across two real commits.
- **Upstreaming status:** not yet proposed to `NVIDIA/NeMo-Speech.cpp`.
- **Rebased onto:** `97a15af` on 2026-09-25 (see the 2026-09-25 sync entry).

### 2026-09-25 — Upstream sync onto the Nemotron 3 diarization base

- **Upstream base:** `97a15af` (`main`, "make Nemotron 3 Diarization the
  default diarizer" #52).
- **PR:** not yet opened.
- **Files:** `src/asr/diar/diar_pipeline.{h,cpp}`, `src/asr/recognizer.cpp`,
  `tests/cpp/asr/CMakeLists.txt`,
  `tests/cpp/asr/test_diar_identity_handoff.cpp`,
  `tests/cpp/asr/test_diar_speaker_change.cpp`,
  `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-design.md` (new),
  `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-results.md` (new),
  `docs/superpowers/plans/2026-09-25-upstream-nemotron3-sync.md` (new).
- **What:** merged 4 upstream commits that reworked the whole diarization
  subsystem for Nemotron 3 (v3, 10 ms native cadence vs v2's 80 ms).
  Merged rather than rebased: as of `d327c61` (this fork's `main` right
  before this sync started), 4 commits upstream vs 22 ours since the merge
  base -- all three divergences living in the subsystem upstream reworked.
  (This count is a snapshot at the decision point, not the sync branch's
  own final commit count, which is naturally higher once its own work is
  included.)
- **Semantic break fixed:** upstream's `flush_available()` changed from a
  drain loop to a single *provisional* chunk (replaced on replay, tracked
  by the new `provisional_frames_`), so divergence #3's commit-boundary
  force-drain stopped draining. The first fix attempt used
  `committed_frames()` for the handoff offset and was **wrong** -- it read
  ~1.6s too small, because frame indices are absolute over *fed* audio,
  not persistently-consumed audio; review caught it. The landed fix adds
  `DiarStream::fed_audio_sec()` instead and drops the force-drain
  entirely: `feed_audio()` already persists every whole chunk, and forcing
  the sub-chunk tail into persistent state would bake truncated right
  context into the AOSC state the next stream adopts.
- **Divergence #2 gated:** the speaker-change event is now gated on
  upstream's new `stable_frames()` frontier via a new `segments_before()`
  filter. Real-audio measurement (see below) found *more* false positives
  gated than pre-gate (11/66 = 16.7% vs. 6/71 = 8.5%), the opposite of the
  expected direction -- but both runs were confounded by differing system
  load and the sample is small, so this is inconclusive, not evidence the
  gate is harmful. A controlled re-run under quiet load is still needed
  before claiming any false-positive improvement.
- **Superseded by upstream:** our hardcoded 2-frame word anchor in
  `speaker_for_word_time()` is replaced by upstream's model-derived
  `word_anchor_frames_`. Ours was a latent bug -- 2 frames is 160 ms at
  v2's 80 ms cadence but 20 ms at v3's 10 ms cadence.
- **Still diverged:** divergence #1's `probs_base_` clamp fix remains
  absent upstream as of `97a15af`, comment claiming the clamp is "inert in
  practice" included. The 2h39m interview disproves it.
- **Verified against:** see
  `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-results.md`.
  Divergences #1 and #3 held up on real audio: the 2h39m interview shows
  all 16 ten-minute windows multi-speaker with no collapse, and speaker
  identity stayed continuous across ~100 real commit boundaries.
  Divergence #2's gate effect was inconclusive, as above. The AMI
  fixture's handoff test failed at `--split-sec` 20 and 25 (both landing
  inside a ~7s RTTM truth gap) while passing at 10/15/30/45.
- **Upstreaming status:** not proposed to `NVIDIA/NeMo-Speech.cpp`.
