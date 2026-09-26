# Upstream Nemotron 3 Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `NVIDIA/NeMo-Speech.cpp@97a15af` into this fork, preserving all three divergences, fixing the adoption-offset break that upstream's new provisional-chunk semantics introduce, and gating the speaker-change event on upstream's new stability frontier.

**Architecture:** Merge (not rebase) on branch `sync/upstream-nemotron3`. Land the merge as its own reviewable commit, then fix semantics in separate commits. Divergence #3's force-drain is *removed* rather than rebuilt — `feed_audio()` already persists every whole chunk, so the adoption offset just needs to read a committed frame count instead of `n_frames()`. Divergence #2 gains a pure segment filter gated on `stable_frames()`.

**Tech Stack:** C++17, CMake presets (`cuda-asr`), CTest. Model-requiring tests are standalone executables with no `add_test()` (repo convention); pure-logic tests are registered with CTest.

**Spec:** `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-design.md`

## Global Constraints

- Our base: `d327c61`. Upstream target: `97a15af`. Merge base: `a5b6953`.
- **Never** reinstate the hardcoded 2-frame word anchor. Upstream's `word_anchor_frames_` must win (2 frames = 160 ms at v2's 80 ms cadence, 20 ms at v3's 10 ms).
- **Never** hardcode `compact_trigger_frames_ = 15000` / `compact_retain_frames_ = 7500`. Upstream's runtime-resolved `0` init must win.
- **Never** touch upstream's `NeMoSpeech::` CMake namespace or install paths (`find_package(NeMoSpeech)`, `NeMoSpeech::ASR`, `~/Library/Caches/NeMoSpeech/models`, `%LOCALAPPDATA%\NeMoSpeech`). Those are NVIDIA's product identifiers, not this project's former name.
- **Never** score `test_files/diar/ami_en2002d_2132.wav` with `tune_diarizer.compute_der` — overlapped truth triggers a known unfixed order-dependence defect.
- Do not commit the `ggml` submodule pointer. It is dirty from unrelated local drift; upstream did not bump it.
- `DiarSegment` fields are `t0`, `t1`, `speaker` (speaker is 0-based).
- No upstream contribution (no push/PR/comment to `NVIDIA/*`).

---

### Task 1: Land the merge with conflicts resolved

**Files:**
- Modify: `src/asr/diar/diar_pipeline.cpp` (include block)
- Modify: `src/asr/diar/diar_pipeline.h` (compaction fields + tracker member)
- Modify: `src/asr/recognizer.cpp` (`finish()`)
- Modify: `tests/cpp/asr/CMakeLists.txt` (both test targets)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a compiling merge commit on `sync/upstream-nemotron3`. Upstream's `DiarStream::stable_frames()`, `provisional_frames_`, `word_anchor_frames_`, and `RecognitionStream::stable_speaker_time()` / `refresh_speaker_tags()` / `set_interim_words()` become available to later tasks.

This task deliberately leaves the (now semantically broken) `flush_available(std::numeric_limits<int64_t>::max())` call in place. Task 3 removes it. Separating "merge" from "fix" is what keeps a 72-file merge reviewable.

- [ ] **Step 1: Start the merge and confirm the expected conflict set**

```bash
cd ~/Projects/NeMo-Speech.cpp
git checkout sync/upstream-nemotron3
git fetch upstream
git merge upstream/main
git diff --name-only --diff-filter=U
```

Expected: exit 1, and exactly these four paths:

```
src/asr/diar/diar_pipeline.cpp
src/asr/diar/diar_pipeline.h
src/asr/recognizer.cpp
tests/cpp/asr/CMakeLists.txt
```

If the set differs, upstream has moved since `97a15af` — stop and re-read the spec's §3 before continuing.

- [ ] **Step 2: Resolve `src/asr/diar/diar_pipeline.cpp` — keep both includes**

Replace the conflict block with both lines in alphabetical order:

```cpp
#include <iterator>
#include <optional>
```

- [ ] **Step 3: Resolve `src/asr/diar/diar_pipeline.h` — upstream's fields, our member**

Replace the conflict block with:

```cpp
    // Initialized to ~20 min trigger / ~10 min retained at native cadence.
    int64_t compact_trigger_frames_ = 0;
    int64_t compact_retain_frames_ = 0;
    SpeakerChangeTracker speaker_change_tracker_;
```

- [ ] **Step 4: Resolve `tests/cpp/asr/CMakeLists.txt` — keep both targets**

Replace the conflict block with:

```cmake
add_executable(test_diar_identity_handoff test_diar_identity_handoff.cpp)
target_link_libraries(test_diar_identity_handoff PRIVATE nemo_speech_asr)

# The presenter is CLI code; compile it in directly so tests need no CLI target.
add_executable(test_live_transcript test_live_transcript.cpp
    ${CMAKE_SOURCE_DIR}/app/live_transcript.cpp)
target_include_directories(test_live_transcript PRIVATE ${CMAKE_SOURCE_DIR}/app)
target_link_libraries(test_live_transcript PRIVATE nemo_speech_asr)
add_test(NAME live_transcript COMMAND test_live_transcript)
```

- [ ] **Step 5: Resolve `src/asr/recognizer.cpp` `finish()` — combine both intents**

Replace the conflict block with the following. Our `finish_diarizer` branch is kept verbatim; upstream's `late_punctuation` plumbing is adopted:

```cpp
    if (diar_) {
        if (finish_diarizer) {
            diar_->finish();  // true end of audio: flush the tail, close for good
        } else {
            // Commit boundary within a longer session: tag the tail via
            // the same on-demand mechanism next() already uses for
            // endpointed finals, without closing the diarizer.
            flush_diar_deficit_(u);
            // Force-drain any remaining audio the diarizer has been fed but not yet
            // chunked into a prediction, so n_frames() (read by the next stream's
            // adoption offset) reflects everything actually fed, not just what's
            // been recognized/flushed above. Truncated right-context on this final
            // partial chunk is already this codebase's accepted behavior on every
            // stream-ending flush.
            diar_->flush_available(std::numeric_limits<int64_t>::max());
        }
    }
    auto result = build_result_(u, /*is_final=*/true);
    result.late_punctuation = std::exchange(late_punctuation_, {}) + u.late_punctuation;
    return result;
}

double
RecognitionStream::stable_speaker_time() const {
    return diar_ ? diar_->stable_frames() * diar_->seconds_per_frame() : 0.0;
}

void
RecognitionStream::refresh_speaker_tags(Result& result) const {
    if (!diar_ || result.alternatives.empty())
        return;
    for (auto& word : result.alternatives.front().words) {
        const int speaker =
            diar_->speaker_for_word_time(word.start_time / 1000.0, word.end_time / 1000.0);
        word.speaker_tag = speaker >= 0 ? speaker + 1 : 0;
    }
```

Note the trailing `}` and the two new method bodies come from upstream's side — `finish()` closes, then the two new methods follow. Ensure `<utility>` is included for `std::exchange` (upstream's side already adds it; verify).

- [ ] **Step 6: Verify our divergence #1 fix and upstream's word anchor both survived**

```bash
grep -n 'speaker_for_frame_range\|speaker_from_frozen' src/asr/diar/diar_pipeline.cpp
grep -n 'word_anchor_frames_\|f0 + 2' src/asr/diar/diar_pipeline.cpp
```

Expected: `speaker_for_frame_range` and `speaker_from_frozen` both present (divergence #1 intact); `word_anchor_frames_` present and **no** `f0 + 2` match.

- [ ] **Step 7: Confirm no conflict markers remain, then build**

```bash
grep -rn '^<<<<<<<\|^>>>>>>>' src/ tests/ && echo "MARKERS REMAIN" || echo "clean"
cmake --preset cuda-asr
cmake --build --preset cuda-asr -j
```

Expected: `clean`, then a successful build.

- [ ] **Step 8: Run the registered test suite**

```bash
ctest --preset cuda-asr --output-on-failure
```

Expected: all tests pass, including `diar_frame_lookup`, `diar_speaker_change`, `diar_state`, and upstream's new `live_transcript`.

- [ ] **Step 9: Commit the merge**

```bash
git add src/asr/diar/diar_pipeline.cpp src/asr/diar/diar_pipeline.h \
        src/asr/recognizer.cpp tests/cpp/asr/CMakeLists.txt
git commit -m "$(cat <<'MSG'
merge: NVIDIA/NeMo-Speech.cpp@97a15af (Nemotron 3 diarization)

Resolves four conflicts: both includes in diar_pipeline.cpp; upstream's
runtime-resolved compaction thresholds plus our speaker_change_tracker_
member; both test targets in tests/cpp/asr/CMakeLists.txt; and finish()
combining our finish_diarizer branch with upstream's late_punctuation
plumbing and its two new speaker-tag methods.

Divergence #1's speaker_for_frame_range()/speaker_from_frozen() fix
survives intact -- upstream still carries the original probs_base_ clamp.
Upstream's word_anchor_frames_ deliberately supersedes our hardcoded
2-frame anchor, which was wrong at v3's 10 ms cadence.

The commit-boundary flush_available(INT64_MAX) call is knowingly left on
the old drain semantics here so this merge stays reviewable on its own;
the next commit replaces it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01V5yFwwzvUHBrTZsXs2v25Z
MSG
)"
git status -sb
```

Expected: the commit lands and `git status -sb` shows only ` m ggml` (do not stage it).

---

### Task 2: Failing test — the adoption offset must exclude the provisional tail

**Files:**
- Modify: `tests/cpp/asr/test_diar_identity_handoff.cpp`

**Interfaces:**
- Consumes: Task 1's merged tree; upstream's `DiarStream::n_frames()` and `provisional_frames_` behavior.
- Produces: an assertion that depends on `DiarStream::committed_frames()`, which Task 3 adds. This test will not compile until Task 3 — that is the intended failing state.

This test is a standalone executable requiring real model and audio arguments (repo convention — no `add_test()` entry).

- [ ] **Step 1: Add the committed-vs-total assertion after extraction**

In `tests/cpp/asr/test_diar_identity_handoff.cpp`, immediately after the existing null check on `extracted` and **before** `stream1.reset();`, insert:

```cpp
    // Upstream's forced non-final chunks are previews: they advance
    // n_frames() but are replaced on replay (provisional_frames_). Take
    // such a preview deliberately, then confirm committed_frames()
    // excludes it. The next stream's adoption offset must derive from the
    // committed frontier -- an offset read off n_frames() would overshoot
    // by the preview tail and desync the adopted diarizer's clock against
    // stream 2's word times.
    extracted->flush_available(std::numeric_limits<int64_t>::max());
    const int64_t total_frames = extracted->n_frames();
    const int64_t committed = extracted->committed_frames();
    if (committed > total_frames) {
        std::fprintf(
            stderr, "[identity-handoff] FAIL: committed_frames() %lld exceeds n_frames() %lld\n",
            static_cast<long long>(committed), static_cast<long long>(total_frames));
        return 1;
    }
    if (committed == total_frames) {
        std::fprintf(
            stderr,
            "[identity-handoff] FAIL: expected a provisional preview tail after "
            "flush_available(), but committed_frames() == n_frames() == %lld. The fixture "
            "must end mid-chunk at the split point, or this test cannot detect an offset "
            "computed from n_frames(). Try a different --split-sec.\n",
            static_cast<long long>(total_frames));
        return 1;
    }
    std::printf(
        "[identity-handoff] frames at split: committed=%lld total=%lld (provisional tail=%lld)\n",
        static_cast<long long>(committed), static_cast<long long>(total_frames),
        static_cast<long long>(total_frames - committed));
```

- [ ] **Step 2: Add the required include**

At the top of the same file, add to the include block (keeping alphabetical order within the group):

```cpp
#include <limits>
```

- [ ] **Step 3: Build and confirm it fails to compile**

```bash
cmake --build --preset cuda-asr --target test_diar_identity_handoff -j
```

Expected: FAIL — `error: 'class nemo_speech::asr::DiarStream' has no member named 'committed_frames'`. This is the correct failing state; Task 3 adds it.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/cpp/asr/test_diar_identity_handoff.cpp
git commit -m "$(cat <<'MSG'
test: assert the handoff offset excludes the provisional preview tail

Deliberately takes a forced preview via flush_available() before
extraction, then requires committed_frames() to be strictly less than
n_frames(). Fails to compile until committed_frames() exists.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01V5yFwwzvUHBrTZsXs2v25Z
MSG
)"
```

---

### Task 3: Remove the force-drain; derive the offset from committed frames

**Files:**
- Modify: `src/asr/diar/diar_pipeline.h` (add `committed_frames()`)
- Modify: `src/asr/recognizer.cpp:327-328` (offset source)
- Modify: `src/asr/recognizer.cpp` (`finish()` — drop the force-drain)
- Test: `tests/cpp/asr/test_diar_identity_handoff.cpp` (from Task 2)

**Interfaces:**
- Consumes: Task 2's failing assertion.
- Produces: `int64_t DiarStream::committed_frames() const` — the frontier of persistent state, excluding any provisional preview tail. Task 4 does not depend on it.

- [ ] **Step 1: Declare `committed_frames()` next to `stable_frames()`**

In `src/asr/diar/diar_pipeline.h`, immediately after the `stable_frames()` declaration in the public section, add:

```cpp
    // Frontier of persistent state: emitted frames minus any provisional
    // preview tail. Unlike stable_frames(), this is NOT clamped to the
    // birth gate -- it answers "how much audio has been permanently
    // consumed" (a clock), not "which labels are immutable" (an identity
    // question). Used for the adoption offset when a DiarStream is handed
    // to a fresh RecognitionStream across a commit boundary.
    int64_t committed_frames() const;
```

Declaration only, defined in the .cpp in the next step — matching how
`stable_frames()` is split. `diar_pipeline.h` does not include
`<algorithm>`, so an inline body using `std::max` would depend on a
transitive include; the .cpp already has it.

- [ ] **Step 1b: Define `committed_frames()` in the .cpp**

In `src/asr/diar/diar_pipeline.cpp`, immediately after the `stable_frames()` definition, add:

```cpp
int64_t
DiarStream::committed_frames() const {
    return std::max<int64_t>(0, n_frames() - provisional_frames_);
}
```

- [ ] **Step 2: Point the adoption offset at it**

In `src/asr/recognizer.cpp`, replace:

```cpp
            diar_time_offset_sec_ =
                existing_diar->n_frames() * existing_diar->seconds_per_frame();
```

with:

```cpp
            // committed_frames(), not n_frames(): a provisional preview tail
            // is replaced on replay, so including it would overshoot this
            // stream's clock against the diarizer's own timeline.
            diar_time_offset_sec_ =
                existing_diar->committed_frames() * existing_diar->seconds_per_frame();
```

- [ ] **Step 3: Delete the force-drain from the commit-boundary path**

In `src/asr/recognizer.cpp`'s `finish()`, replace the whole `else` branch body with:

```cpp
            // Commit boundary within a longer session: tag the tail via
            // the same on-demand mechanism next() already uses for
            // endpointed finals, without closing the diarizer. That flush
            // is a provisional preview and is discarded with this stream.
            //
            // Deliberately NO force-drain here. feed_audio() already runs
            // every whole chunk persistently, so the committed frontier is
            // already correct. The only way to force the sub-chunk tail
            // into persistent state is run_one_chunk(force=true,
            // final_flush=true), which bakes a truncated-right-context
            // chunk into the AOSC state the *next* stream adopts --
            // acceptable at a true end of stream, not at a boundary where
            // audio continues. The tail stays in mel_buf_ and the adopting
            // stream labels it with full right context.
            flush_diar_deficit_(u);
```

- [ ] **Step 4: Build and run the handoff test**

Substitute the real model paths (see SliqSpeech's README "Verified environment"; typically under `~/.cache/nemo-speech/models/`) and the committed multi-speaker fixture:

```bash
cmake --build --preset cuda-asr -j
./build/cuda-asr/tests/cpp/asr/test_diar_identity_handoff \
  ~/.cache/nemo-speech/models/nvidia/parakeet-tdt-0.6b-v3/*/parakeet-tdt-0.6b-v3.q8_0.gguf \
  ~/.cache/nemo-speech/models/nvidia/diar_streaming_sortformer_4spk-v2/*/diar_streaming_sortformer_4spk-v2.q8_0.gguf \
  test_files/asr/wav/test/scotus_08-1314_excerpt.wav --gpu 0
```

Expected: prints a `frames at split: committed=… total=… (provisional tail=…)` line with a non-zero tail, then `[identity-handoff] OK`. If it reports `committed == n_frames()`, re-run with a different `--split-sec` so the split lands mid-chunk.

- [ ] **Step 5: Confirm nothing else regressed**

```bash
ctest --preset cuda-asr --output-on-failure
```

Expected: all registered tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/asr/diar/diar_pipeline.h src/asr/recognizer.cpp
git commit -m "$(cat <<'MSG'
fix: derive the handoff offset from committed frames, not n_frames()

Upstream's flush_available() went from a drain loop to a single forced
chunk that is provisional (replaced on replay), so divergence #3's
commit-boundary force-drain no longer drained -- and the adoption offset
was read off a frame count that was both incomplete and about to be
rewound.

Drops the force-drain entirely rather than rebuilding it. feed_audio()
already persists every whole chunk, and the only way to force the
sub-chunk tail into persistent state bakes truncated right context into
the AOSC state the next stream adopts. The tail now stays in mel_buf_ and
the adopting stream labels it with full right context.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01V5yFwwzvUHBrTZsXs2v25Z
MSG
)"
```

---

### Task 4: Gate the speaker-change event on the stability frontier

**Files:**
- Modify: `src/asr/diar/diar_pipeline.h` (declare `segments_before()`, rewire `poll_speaker_change()`)
- Modify: `src/asr/diar/diar_pipeline.cpp` (define `segments_before()`)
- Test: `tests/cpp/asr/test_diar_speaker_change.cpp`

**Interfaces:**
- Consumes: upstream's `DiarStream::stable_frames()` from Task 1.
- Produces: `std::vector<DiarSegment> segments_before(const std::vector<DiarSegment>& segments, double stable_time)` — a free function in `nemo_speech::asr`, keeping only segments whose onset is strictly before `stable_time`.

Gating on `t0 < stable_time` (onset settled), not `t1 <= stable_time` (segment fully ended). Requiring the whole segment to be settled would mean never reporting a speaker who is still talking, which is the only case the event exists for.

- [ ] **Step 1: Write the failing tests**

In `tests/cpp/asr/test_diar_speaker_change.cpp`, add these three functions in the anonymous namespace, after the existing test functions:

```cpp
bool
test_segments_before_keeps_settled_onsets() {
    // Onsets at 0.0 and 2.0 are both before the frontier at 3.0; the one
    // at 4.0 is not yet immutable and must be withheld.
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}, {2.0, 5.0, 1}, {4.0, 6.0, 2}};
    const auto got = segments_before(segs, /*stable_time=*/3.0);
    if (got.size() != 2 || got[0].speaker != 0 || got[1].speaker != 1) {
        std::fprintf(
            stderr, "[FAIL] expected 2 settled segments (speakers 0,1), got %zu\n", got.size());
        return false;
    }
    return true;
}

bool
test_segments_before_withholds_everything_at_a_zero_frontier() {
    // A fresh stream whose frontier has not advanced yet: nothing is
    // immutable, so nothing may be reported.
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}, {2.0, 5.0, 1}};
    const auto got = segments_before(segs, /*stable_time=*/0.0);
    if (!got.empty()) {
        std::fprintf(
            stderr, "[FAIL] expected no segments at a zero frontier, got %zu\n", got.size());
        return false;
    }
    return true;
}

bool
test_unsettled_change_is_not_reported_through_the_gate() {
    // The false-positive case the gate exists to kill: speaker 1's segment
    // starts past the frontier, so the change is not yet confirmable.
    // Gated, the tracker must still see only speaker 0 and report nothing.
    const std::vector<DiarSegment> segs = {{0.0, 2.0, 0}, {4.0, 6.0, 1}};
    const auto gated = segments_before(segs, /*stable_time=*/3.0);
    const auto got = detect_speaker_change(gated, /*last_reported=*/0);
    if (got) {
        std::fprintf(
            stderr, "[FAIL] expected no change for a segment starting past the frontier, "
            "got speaker %d at t=%.2f\n", got->speaker, got->start_time);
        return false;
    }
    return true;
}
```

Then register them in `main()` alongside the existing calls, following the file's existing pattern (each test's result AND-ed into the pass flag).

- [ ] **Step 2: Run to verify the tests fail**

```bash
cmake --build --preset cuda-asr --target test_diar_speaker_change -j
```

Expected: FAIL — `error: 'segments_before' was not declared in this scope`.

- [ ] **Step 3: Declare `segments_before()` in the header**

In `src/asr/diar/diar_pipeline.h`, immediately after the `detect_speaker_change()` declaration and **before** `class SpeakerChangeTracker`, add:

```cpp
// Keep only segments whose onset is in immutable territory (t0 strictly
// before stable_time). Gates the speaker-change event on
// DiarStream::stable_frames(): without it the tracker reports off
// segments().back() with no stability guarantee at all, which is the
// mechanism behind the documented false positives -- a "confirmed" change
// whose words the very next final still tags with the old speaker.
//
// Deliberately gates on the onset, not on t1 <= stable_time: requiring a
// segment to have ended would mean never reporting a speaker who is
// currently talking, which is the only case this event exists for.
std::vector<DiarSegment> segments_before(
    const std::vector<DiarSegment>& segments, double stable_time);
```

- [ ] **Step 4: Define `segments_before()` in the .cpp**

In `src/asr/diar/diar_pipeline.cpp`, immediately after the `detect_speaker_change()` definition, add:

```cpp
std::vector<DiarSegment>
nemo_speech::asr::segments_before(
    const std::vector<DiarSegment>& segments, double stable_time) {
    std::vector<DiarSegment> out;
    out.reserve(segments.size());
    for (const auto& s : segments)
        if (s.t0 < stable_time)
            out.push_back(s);
    return out;
}
```

- [ ] **Step 5: Run to verify the tests pass**

```bash
cmake --build --preset cuda-asr --target test_diar_speaker_change -j
ctest --preset cuda-asr -R diar_speaker_change --output-on-failure
```

Expected: PASS, including the three new cases.

- [ ] **Step 6: Wire the gate into `poll_speaker_change()`**

In `src/asr/diar/diar_pipeline.h`, replace the body of `poll_speaker_change()`:

```cpp
    std::optional<DiarSpeakerChange> poll_speaker_change() {
        return speaker_change_tracker_.observe(
            segments_before(segments(), stable_frames() * sec_per_frame_));
    }
```

- [ ] **Step 7: Rebuild and run the full suite plus the handoff test**

```bash
cmake --build --preset cuda-asr -j
ctest --preset cuda-asr --output-on-failure
```

Then re-run the Task 3 Step 4 handoff command. Expected: still `[identity-handoff] OK`. Its `speaker_changes2 >= 1` assertion now runs through the gate, so a frontier that is too conservative would surface here as a failure.

- [ ] **Step 8: Commit**

```bash
git add src/asr/diar/diar_pipeline.h src/asr/diar/diar_pipeline.cpp \
        tests/cpp/asr/test_diar_speaker_change.cpp
git commit -m "$(cat <<'MSG'
fix: gate the speaker-change event on the stability frontier

SpeakerChangeTracker reported off segments().back() with no stability
guarantee, which is the mechanism behind the documented false positives:
SliqSpeech cross-referenced .changed events against the next .completed
and repeatedly found every word still tagged with the old speaker.

Adds segments_before(), gating on upstream's new stable_frames() frontier
so a change is reported only once its onset is immutable. Gates on the
onset rather than requiring the segment to have ended -- the latter would
never report a speaker who is still talking.

The event now fires later but truthfully. Nothing downstream commits on
it any more, so the added latency is free.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01V5yFwwzvUHBrTZsXs2v25Z
MSG
)"
```

---

### Task 5: Real-audio verification

**Files:**
- Create: `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-results.md`

**Interfaces:**
- Consumes: the built binary from Tasks 1–4.
- Produces: a results doc recording measured outcomes. No source changes.

Record actual numbers. A negative result is a valid outcome and must be written down rather than retried until it passes.

- [ ] **Step 1: Divergence #1 regression — the long-file collapse**

This is the defect the fork exists for, and upstream still ships it. Transcribe the 2h39m two-speaker interview with diarization and bucket speaker time per 10-minute window:

```bash
./build/cuda-asr/nemo-speech transcribe --diarize \
  --model ~/.cache/nemo-speech/models/nvidia/parakeet-tdt-0.6b-v3/*/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --diar-model ~/.cache/nemo-speech/models/nvidia/diar_streaming_sortformer_4spk-v2/*/diar_streaming_sortformer_4spk-v2.q8_0.gguf \
  --gpu 0 --word-timestamps --output-format json "$LONG_INTERVIEW" \
  > /tmp/sync-verify-longfile.json
```

`$LONG_INTERVIEW` is the 2h39m two-speaker recording that originally
exposed divergence #1. It is **not** committed to either repo (size), so
set it to your local copy first. If it is unavailable, this step cannot be
substituted with the short committed fixture — the defect only appears
past the ~20 min compaction horizon. In that case record the step as
un-run rather than as passed, and treat `test_diar_frame_lookup` (which
covers the same logic synthetically) as the only #1 evidence.

Expected: both speakers appear in every 10-minute window, alternating
roughly proportionally. FAIL condition: one speaker holding ~140 of 160
minutes — that is the original collapse, and it means divergence #1 was
lost in the merge.

- [ ] **Step 2: Divergence #2 — count false positives before vs after the gate**

Use the SliqSpeech virtual-sink replay (reproducible; no live microphone):

```bash
cd ~/Projects/SliqSpeech
./scripts/replay_live_clip.sh <a multi-speaker clip>
```

For each `conversation.item.speaker_diarization.changed` event, cross-reference the speaker tags on the very next `.completed`. A false positive is an event whose claimed new speaker does not appear in the following final's word tags. Record the count for the gated build, and against the pre-merge binary for comparison.

Expected: the gated count is lower. Record both numbers even if the improvement is small or absent.

- [ ] **Step 3: Divergence #3 — identity across real commit boundaries**

In the same replay, confirm a speaker keeps its number across silence-triggered commits for the whole clip. Expected: no renumbering at a commit boundary.

- [ ] **Step 4: AMI fixture — continuity observation only**

```bash
./build/cuda-asr/tests/cpp/asr/test_diar_identity_handoff \
  <asr.gguf> <diar.gguf> test_files/diar/ami_en2002d_2132.wav --gpu 0 --split-sec 20
```

Upstream's new fixture is the first multi-speaker audio with ground-truth RTTM this repo has had. Use it for continuity and false-positive observation only. Do **not** run `tune_diarizer.compute_der` against it — heavy overlap triggers the known order-dependence defect, so any DER number from it would be meaningless.

- [ ] **Step 5: HTTP protocol conformance — the divergence #3 proof**

`tests/integration/http_conformance_test.py` asserts *"speaker identity
persists across a commit boundary"* (lines 399-405) — the protocol-level
proof of exactly what Task 3 changed — and separately requires the
speaker-change event to fire at all (line 423). It is a standalone script,
not a pytest module, and needs a multi-speaker fixture:

```bash
python tests/integration/http_conformance_test.py \
  --binary ./build/cuda-asr/nemo-speech \
  --asr-model ~/.cache/nemo-speech/models/nvidia/parakeet-tdt-0.6b-v3/*/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --diar-model ~/.cache/nemo-speech/models/nvidia/diar_streaming_sortformer_4spk-v2/*/diar_streaming_sortformer_4spk-v2.q8_0.gguf \
  --audio test_files/asr/wav/test/scotus_08-1314_excerpt.wav
```

Expected: all `require(...)` checks pass.

**Watch line 423 specifically.** Task 4's gating delays the event until its
onset is immutable, so on a short fixture the frontier may never advance
far enough for any event to fire — turning `require(speaker_changes,
"WebSocket speaker-change event")` into a failure. If that happens it is a
real finding about the gate being too conservative, not a flaky test:
record it, and re-run against a longer multi-speaker clip to confirm the
event fires there. Do not weaken the assertion to make it pass.

- [ ] **Step 6: Companion-repo checks**

```bash
cd ~/Projects/SliqSpeech
python -m pytest tests/test_live_bridge_client.py -k speaker_change -v
```

Expected: `test_client_does_not_commit_on_speaker_change_event` and `test_speaker_change_event_leaves_silence_trigger_state_untouched` pass. Also confirm the debug view's status panel still displays speaker changes during the Step 2 replay.

- [ ] **Step 7: Write and commit the results doc**

Create `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-results.md` with one section per step above, each recording the command run and the measured outcome (not "passed" — the actual per-window speaker split, the actual false-positive counts). Then:

```bash
git add docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-results.md
git commit -m "$(cat <<'MSG'
docs: real-audio verification results for the Nemotron 3 sync

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01V5yFwwzvUHBrTZsXs2v25Z
MSG
)"
```

---

### Task 6: Update `FORK_CHANGES.md` and open the PR

**Files:**
- Modify: `FORK_CHANGES.md`

**Interfaces:**
- Consumes: the commits and measured results from Tasks 1–5.
- Produces: the divergence log entry and a PR into `bsips/main`.

- [ ] **Step 1: Append the new divergence entry**

Add to the end of `FORK_CHANGES.md`, matching the existing entry format exactly:

```markdown
### 2026-09-25 — Upstream sync onto the Nemotron 3 diarization base

- **Upstream base:** `97a15af` (`main`, "make Nemotron 3 Diarization the
  default diarizer" #52).
- **PR:** [bsips/NeMo-Speech.cpp#4](https://github.com/bsips/NeMo-Speech.cpp/pull/4)
- **Files:** `src/asr/diar/diar_pipeline.{h,cpp}`, `src/asr/recognizer.cpp`,
  `tests/cpp/asr/CMakeLists.txt`,
  `tests/cpp/asr/test_diar_identity_handoff.cpp`,
  `tests/cpp/asr/test_diar_speaker_change.cpp`,
  `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-design.md` (new),
  `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-results.md` (new),
  `docs/superpowers/plans/2026-09-25-upstream-nemotron3-sync.md` (new).
- **What:** merged 4 upstream commits that reworked the whole diarization
  subsystem for Nemotron 3 (v3, 10 ms native cadence vs v2's 80 ms).
  Merged rather than rebased: 4 commits upstream vs 22 ours, all three
  divergences living in the reworked subsystem.
- **Semantic break fixed:** upstream's `flush_available()` changed from a
  drain loop to a single forced chunk that is *provisional* (replaced on
  replay, tracked by the new `provisional_frames_`). Divergence #3's
  commit-boundary force-drain therefore stopped draining, and the
  adoption offset was being read off an incomplete, about-to-be-rewound
  frame count. Dropped the force-drain entirely and added
  `DiarStream::committed_frames()` for the offset -- `feed_audio()`
  already persists every whole chunk, and forcing the sub-chunk tail into
  persistent state would bake truncated right context into the AOSC state
  the next stream adopts.
- **Divergence #2 improved:** gated the speaker-change event on upstream's
  new `stable_frames()` frontier via a new `segments_before()` filter,
  fixing the false positives documented in `SliqSpeech`'s README.
- **Superseded by upstream:** our hardcoded 2-frame word anchor in
  `speaker_for_word_time()` is replaced by upstream's model-derived
  `word_anchor_frames_`. Ours was a latent bug -- 2 frames is 160 ms at
  v2's cadence but 20 ms at v3's.
- **Still diverged:** divergence #1's `probs_base_` clamp fix remains
  absent upstream as of `97a15af`, comment claiming the clamp is "inert in
  practice" included. The 2h39m interview disproves it.
- **Verified against:** see
  `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-results.md`.
- **Upstreaming status:** not proposed to `NVIDIA/NeMo-Speech.cpp`.
```

- [ ] **Step 2: Amend the three existing entries' upstream base**

For each of the `2026-09-10`, `2026-09-13` (live speaker-change event) and `2026-09-13` (persistent diarizer identity) entries, append one line to its bullet list:

```markdown
- **Rebased onto:** `97a15af` on 2026-09-25 (see the 2026-09-25 sync entry).
```

- [ ] **Step 3: Commit**

```bash
git add FORK_CHANGES.md
git commit -m "$(cat <<'MSG'
docs: log the Nemotron 3 upstream sync divergence

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01V5yFwwzvUHBrTZsXs2v25Z
MSG
)"
```

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin sync/upstream-nemotron3
gh pr create --repo bsips/NeMo-Speech.cpp --base main \
  --title "Sync onto the Nemotron 3 diarization base" \
  --body "$(cat <<'MSG'
Merges `NVIDIA/NeMo-Speech.cpp@97a15af` (4 commits, Nemotron 3
diarization) into the fork, preserving all three divergences.

The merge is textually small (4 files, ~40 lines) but hid a semantic
break: `flush_available()` went from a drain loop to a single *provisional*
forced chunk, so divergence #3's adoption offset was read off an
incomplete, about-to-be-rewound frame count. Fixed by dropping the
force-drain and adding `committed_frames()`.

Also gates divergence #2's speaker-change event on upstream's new
`stable_frames()` frontier, fixing its documented false positives, and
lets upstream's `word_anchor_frames_` supersede our hardcoded 2-frame
anchor (which was wrong at v3's 10 ms cadence).

Design: `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-design.md`
Results: `docs/superpowers/specs/2026-09-25-upstream-nemotron3-sync-results.md`

Divergence #1 remains unfixed upstream.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01V5yFwwzvUHBrTZsXs2v25Z
MSG
)"
```

Note: `bsips/NeMo-Speech.cpp` is our own fork, so this PR is in-bounds. Nothing is pushed to `NVIDIA/*`.

---

## Deferred (tracked, not in this plan)

- Workstream **B**: purge our own stale `NeMoSpeech` cross-references (starting `FORK_CHANGES.md:7`, a dead link to `github.com/bsips/NeMoSpeech`). Upstream's `NeMoSpeech::` identifiers stay.
- Workstream **C**: SliqSpeech README published results (RTFx, DER) plus demo media with placeholders.
- Evaluating v3 / Nemotron 3 as SliqSpeech's diarizer (it pins v2; the tuning corpus is v2-calibrated).
- Wiring `set_interim_words`/`refresh_speaker_tags` into SliqSpeech's live path.
- Fixing `tune_diarizer.compute_der`'s overlap order-dependence.
- Cleaning the dirty `ggml` submodule pointer.
