# Upstream sync: Nemotron 3 diarization base — design

**Date:** 2026-09-25
**Branch:** `sync/upstream-nemotron3`
**Our base:** `d327c61` (`main`, after PR #3)
**Upstream target:** `97a15af` (`NVIDIA/NeMo-Speech.cpp@main`)
**Merge base:** `a5b6953`

## 1. Problem

`NVIDIA/NeMo-Speech.cpp@main` has moved 4 commits ahead of our merge base,
and those commits substantially rework the subsystem all three of our
divergences live in (`diar_pipeline`, `recognizer`, `sortformer_model`, plus a
new `rope_transformer`):

| Commit | Change |
| --- | --- |
| `97a15af` | make Nemotron 3 Diarization the **default** diarizer (#52) |
| `8c15060` | Nemotron 3 Diarization — v3 model, 10 ms native cadence (#50) |
| `302ebc9` | s2s docs: `max_streams` semantics, Docker UID mount fix (#39) |
| `07003da` | ggml: guard the `half_snake` fusion against allocator buffer reuse (#28) |

72 files, +4392/−464. Staying on the stale base costs us a real ggml
correctness fix, a latent-bug fix in our own code (see §4.2), and grows the
reconciliation debt every time upstream lands another diarization change.

## 2. Approach: merge, not rebase

Merge `upstream/main` into `sync/upstream-nemotron3`, then PR into
`bsips/main`.

The direction asymmetry decides the verb: **4 commits upstream, 22 ours.** A
rebase replays our 22 over a rewritten diar subsystem, and because all three
of our feature branches touched `diar_pipeline.cpp`, the same hunks would be
re-resolved once per branch. A merge resolves once. It also preserves the
published history of `bsips/main`, which already carries three merged PRs.

## 3. Conflict resolutions

A dry-run merge produces **4 conflicted files, ~40 lines total.** `recognizer.h`
and `docs/api.md` auto-merge cleanly.

### 3.1 `src/asr/diar/diar_pipeline.cpp` — 1 line

Include-block conflict only: ours `<iterator>`, theirs `<optional>`. Keep both.

### 3.2 `src/asr/diar/diar_pipeline.h` — 4/3 lines

Upstream re-initializes the compaction thresholds to `0` (resolved at runtime
from the model's native cadence, since v3 is 10 ms and v2 is 80 ms); our side
adds the `speaker_change_tracker_` member. Take **upstream's** threshold
fields and **keep** our tracker member. Ours must not win here — hardcoding
`15000`/`7500` would mean a ~2.5 min compaction window under v3 instead of
~20 min.

### 3.3 `tests/cpp/asr/CMakeLists.txt` — 1 hunk

Their `test_live_transcript` target vs our `test_diar_identity_handoff`. Keep
both.

### 3.4 `src/asr/recognizer.cpp` — 18/19 lines, the only substantive one

Our `finish(bool finish_diarizer)` body vs upstream's rewritten `finish()`
(which now threads `late_punctuation` through the result) plus their two new
methods `stable_speaker_time()` and `refresh_speaker_tags()`.

The two intents do not contradict. Resolution is to hand-combine: keep our
`finish_diarizer` branch (the commit-boundary path that must not close the
diarizer), adopt their `late_punctuation` plumbing into the returned result,
and keep both new methods. See §4 — the diarizer branch itself needs rework,
not just re-placement.

## 4. What auto-merges, and what that hides

### 4.1 Divergence #1 survives intact — and is still unfixed upstream

Our `speaker_for_frame_range()` / `speaker_from_frozen()` fix survives the
automerge. Upstream still carries the original defect verbatim in
`speaker_for_frames()`, including the comment asserting it is harmless:

> ranges before the retained window (only reachable when a caller tags words
> hours after they were spoken) clamp to its first frame. Word tagging happens
> near the frontier, so the clamp is inert in practice.

The 2h39m interview that motivated divergence #1 disproves "inert in
practice" — `Recognizer::recognize()` tags every word in one pass at the very
end, so every query on a long file lands far behind the frontier. Our fix must
be preserved through the merge, and this remains the fork's strongest
upstreaming candidate. **No upstream contribution is in scope here.**

### 4.2 Upstream's `word_anchor_frames_` correctly overwrites our `+ 2`

Our `speaker_for_word_time()` hardcoded a 2-frame anchor window. Upstream
replaces it with a model-derived `word_anchor_frames_`. Their version must
win: 2 frames is 160 ms at v2's 80 ms cadence but only 20 ms at v3's 10 ms
cadence. This is a latent bug in our code fixed for free by the merge — do not
reinstate the constant while resolving §3.

### 4.3 The semantic break: `flush_available()` changed meaning

This is the one failure the merge does **not** surface as a conflict marker,
because the call site's text never changes — only the callee moved.

Merge base (what divergence #3 was written against) drained:

```cpp
void DiarStream::flush_available(int64_t target_frame) {
    while (n_frames() < target_frame && run_one_chunk(/*force=*/true, /*final_flush=*/false)) {
    }
}
```

Upstream now runs at most one forced chunk, and per the new header contract
that chunk is a **preview**: "Forced non-final chunks are previews only; only
full chunks and the final flush advance persistent model/feature state." A new
`provisional_frames_` member tracks the tail that gets replaced on replay.

```cpp
void DiarStream::flush_available(int64_t target_frame) {
    if (!finished_ && n_frames() < target_frame)
        run_one_chunk(/*force=*/true, /*final_flush=*/false);
}
```

Our commit-boundary path in `RecognitionStream::finish()` calls
`flush_available(std::numeric_limits<int64_t>::max())` specifically to drain,
so that `n_frames()` yields a correct adoption offset for the next stream
(`http_server.cpp:1187-1199`: `commit` → `finish(finish_diarizer=false)` →
`extract_diar_stream()` → `persistent_diar` → next `streaming_recognize`).

Post-merge that call would advance at most one chunk, and whatever it advanced
would be provisional and subject to replay. The adoption offset would be read
off a frame count that is both incomplete and about to be rewound — producing
silently wrong speaker attribution across a commit boundary, which is the
exact bug class divergence #3 exists to prevent.

## 5. Design: fix the adoption offset (divergence #3)

Divergence #3 keeps its handoff architecture. Upstream's
`refresh_speaker_tags()`/`stable_speaker_time()` does **not** subsume it: that
mechanism retags words *within a living stream* as the diarizer revises its
timeline, whereas #3 carries diarizer state *across stream destruction*.
Nothing in `refresh_speaker_tags()` survives the `RecognitionStream` being
destroyed at a commit, and a fresh `DiarStream` renumbers speakers
arbitrarily. The two are complementary, not competing.

### 5.1 The force-drain was never the right mechanism

`DiarStream::feed_audio()` already calls `run_ready_chunks(/*end_of_stream=*/false)`,
which is `while (run_one_chunk(/*force=*/false, /*final_flush=*/false))` — so
**every whole chunk is already persisted as audio arrives.** At a commit
boundary the only thing our `flush_available(INT64_MAX)` force-drain added was
the sub-chunk *tail*.

We must not persist that tail. In upstream's code
`provisional = force && !final_flush`, so the only way to force the tail into
persistent state is `run_one_chunk(/*force=*/true, /*final_flush=*/true)` —
which bakes a truncated-right-context chunk permanently into the AOSC state
that the *next* stream then adopts. Upstream made forced chunks provisional
precisely to stop that from degrading subsequent predictions. Truncated right
context is acceptable at a true end of stream (nothing follows it); at a commit
boundary audio continues, so it is not.

And the tail is not lost by declining to force it: it stays in `mel_buf_` /
`audio_buf_`, and the adopted `DiarStream` processes it with full right context
as soon as the next stream feeds more audio.

### 5.2 The fix

**Stop draining at the commit boundary.** Derive the adoption offset from the
*committed* frame count instead of from `n_frames()`:

- Add a narrow accessor `int64_t DiarStream::committed_frames() const`
  returning `n_frames() - provisional_frames_` — the frontier of persistent
  state, excluding any preview tail.
- `RecognitionStream::finish(finish_diarizer=false)` drops the
  `flush_available(std::numeric_limits<int64_t>::max())` call entirely.
- The next stream's `diar_time_offset_sec_` derives from
  `committed_frames() * seconds_per_frame()`.

`stable_frames()` is deliberately *not* reused for this. For v2 it additionally
clamps to `birth_gate_.settled_frames()`, which is the right frontier for
deciding whether a speaker *label* is immutable (§6) but the wrong one for a
*clock* offset — it would understate elapsed audio and misalign the adopted
diarizer's timeline against the next stream's word times.

### 5.3 Word tagging at the boundary is unaffected

Tagging this final's tail words continues to use the provisional preview via
the existing `flush_diar_deficit_(u)` path. That is exactly what upstream built
the preview for, and it is discarded rather than adopted, so it cannot pollute
the handed-off state. The two concerns — *tag the words now* and *hand off a
clean clock* — end up cleanly separated, which is the part that makes this
reviewable.

## 6. Design: gate divergence #2 on `stable_frames()`

`SpeakerChangeTracker` currently reports off `segments().back()` with no
stability gating. That is the mechanism behind the documented false positives:
SliqSpeech's `live_bridge.py` cross-referenced `.changed` events against the
very next `.completed` and repeatedly found every word still tagged with the
*old* speaker. The event was consequently demoted to observability only —
`live_bridge.py:458` routes it to the debug status panel, and
`test_client_does_not_commit_on_speaker_change_event` now enforces that it
must not drive segmentation.

Upstream's new frontier supplies exactly the missing guarantee: "Speaker
labels strictly before this audio time are immutable."

**Change:** restrict the segment view the tracker sees to the `stable_frames()`
frontier, so a speaker change is reported only once it can no longer be
revised. For v2 `stable_frames()` clamps to `birth_gate_.settled_frames()`,
which is the conservative and correct choice.

**Wire effect:** the event fires later but truthfully. The added latency costs
nothing, because nothing downstream commits on it any more.

## 7. Verification

Build with the `cuda-asr` preset; full `ctest`.

### 7.1 Unit

- `test_diar_identity_handoff.cpp` — extend to assert the adoption offset
  derives from `committed_frames()` and **excludes a provisional preview tail**.
  The fixture must leave a sub-chunk tail pending at the boundary and take a
  preview over it, so an offset computed from `n_frames()` would overshoot;
  a fixture that ends exactly on a chunk boundary would pass either way.
  Also assert the adopted stream labels that tail with full right context
  (i.e. the tail was left in the mel buffer, not force-consumed).
- `test_diar_speaker_change.cpp` — extend with synthetic segments asserting
  nothing is reported before the stable frontier. No model needed.
- `test_diar_frame_lookup.cpp` — must still pass unmodified (divergence #1).

### 7.2 Real audio

- **Divergence #1 regression** (the reason this fork exists): re-run the 2h39m
  two-speaker interview and confirm both speakers still alternate
  proportionally across every 10-minute window, rather than collapsing onto
  one for ~140 of 160 minutes.
- **#3 and #2**: `scripts/replay_live_clip.sh` in the SliqSpeech repo — a
  virtual-sink clip replay, reproducible and requiring no live microphone.
  Confirm speaker identity holds across silence-triggered commits, and count
  `.changed` false positives before vs after the §6 gating by
  cross-referencing each event against the next `.completed`.
- **AMI fixture**: upstream's new `test_files/diar/ami_en2002d_2132.{wav,rttm,json}`
  is the first multi-speaker fixture with ground-truth RTTM the fork has had.
  Use it for identity-continuity and false-positive observation **only**.
  Do **not** score it with `tune_diarizer.compute_der`: AMI is meeting audio
  with heavy overlap, and `compute_der` has a known, unfixed order-dependence
  defect on overlapped truth (see SliqSpeech
  `docs/superpowers/specs/2026-09-21-der-scorer-overlap-order-dependence.md`).
  DER figures stay on the existing non-overlapped fixtures.

### 7.3 Companion repo

- SliqSpeech's debug status panel still receives speaker changes.
- `test_client_does_not_commit_on_speaker_change_event` still passes.

## 8. Deferred / explicitly out of scope

- **Adopting v3 / Nemotron 3 as SliqSpeech's diarizer.** SliqSpeech pins its
  model explicitly (`live.py` `--diar-model sortformer`;
  `run.py`/`tune_diarizer.py` `nvidia/diar_streaming_sortformer_4spk-v2`), so
  the new upstream default cannot silently flip it. The entire offline tuning
  corpus is v2-calibrated. Evaluating v3 needs its own sweep and its own spec.
- **Wiring `set_interim_words`/`refresh_speaker_tags` into SliqSpeech's live
  path.** Available after this merge, deliberately unused. Rendering captions
  from interim results instead of committing on silence would make #3's
  handoff unnecessary altogether, but that is a cross-repo re-architecture.
- **Renaming upstream's `NeMoSpeech::` CMake namespace or install paths.**
  `find_package(NeMoSpeech)`, `NeMoSpeech::ASR`,
  `~/Library/Caches/NeMoSpeech/models`, `%LOCALAPPDATA%\NeMoSpeech` are
  NVIDIA's own product identifiers, unrelated to this project's former name.
  Changing them would break `docs/sdk.md` and manufacture permanent merge
  conflicts. The separate stale-naming cleanup (workstream B) covers only our
  own dead cross-references, starting with `FORK_CHANGES.md:7`.
- **Fixing the `compute_der` overlap defect.** Stands on its own.
- **The dirty `ggml` submodule.** Local drift; upstream did not bump it.
  Cleaned separately.
- **Upstreaming divergence #1 to NVIDIA.** Noted in §4.1 as a candidate; not
  acted on.

## 9. `FORK_CHANGES.md` updates

- New divergence entry for this sync: the merge, the §5 adoption-offset fix, the §6
  gating.
- Amend the #1/#2/#3 entries with the new upstream base `97a15af`.
- Record that our hardcoded 2-frame word anchor was superseded by upstream's
  `word_anchor_frames_` (§4.2).
- Record that divergence #1 remains unfixed upstream as of `97a15af`.
