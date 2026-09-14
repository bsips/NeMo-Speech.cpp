# Persistent Diarizer Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the realtime WebSocket handler from wiping the diarizer's speaker identity on every commit, so a given speaker number refers to the same real person across an entire live session, not just within one commit's audio.

**Architecture:** Reuse the existing `flush_available()` on-demand tagging mechanism (already used for mid-stream endpointed finals) instead of the harder `DiarStream::finish()` at commit time, so the diarizer can be detached from a dying `RecognitionStream` and handed to the next one instead of destroyed. A previously-undiscovered wrinkle, found while planning: the adopted diarizer's internal clock never resets, but each new stream's ASR clock does — word-to-speaker tagging needs an automatically-derived time offset to stay correctly aligned across the handoff.

**Tech Stack:** C++17, `nemo_speech::asr` library (`src/asr/`), CTest, the existing Python `websockets`-based integration test (`tests/integration/http_conformance_test.py`).

**Spec:** `docs/superpowers/specs/2026-09-13-persistent-diarizer-identity-design.md`

## Global Constraints

- Every new parameter (`finish(bool finish_diarizer = true)`, `existing_diar = nullptr` on both the `RecognitionStream` constructor and `Recognizer::streaming_recognize`) defaults to today's exact behavior — every existing call site must compile and behave identically without modification.
- The `diar_time_offset_sec_` correction is entirely internal to `RecognitionStream`, derived automatically from the adopted `DiarStream`'s own state (`n_frames() * seconds_per_frame()`) at the moment of adoption. No caller (including `http_server.cpp`) computes or passes it explicitly.
- `ww.start_time`/`ww.end_time` (the `Word` struct's own reported/returned timestamps) are never modified by the offset — only the values passed into `diar_->speaker_for_word_time(...)`/`flush_diar_deficit_`'s target-frame computation get the offset added, at the query call site only.
- `SpeakerChangeTracker`'s own class and its existing unit tests (`tests/cpp/asr/test_diar_speaker_change.cpp`) are unchanged — only *where an instance of it lives* moves (from `RecognitionStream` to `DiarStream`).
- No change to `NemoSpeech`/`live_bridge.py` or the wire protocol.
- Every new/modified `.cpp`/`.h` file keeps the existing SPDX header, copied verbatim from a neighboring file in the same directory.
- Build/test with `cmake --build --preset cpu-asr` + `ctest --test-dir build/cpu-asr --output-on-failure` (per `CONTRIBUTING.md`); the new end-to-end test additionally needs real ASR/diarizer GGUF models and is not part of default `ctest` (matching `test_diar_recognizer`'s existing convention).

---

## Task 1: Core plumbing — persist the diarizer across a commit boundary

**Files:**
- Modify: `src/asr/diar/diar_pipeline.h` (move `SpeakerChangeTracker` ownership onto `DiarStream`)
- Modify: `src/asr/recognizer.h` (constructor signature, `finish(bool)`, remove `speaker_change_tracker_`, add `diar_time_offset_sec_`)
- Modify: `src/asr/recognizer.cpp` (constructor body, `finish()` body, `poll_speaker_change()`, `flush_diar_deficit_()`, per-word tagging, `Recognizer::streaming_recognize`)
- Create: `test_files/asr/wav/test/scotus_08-1314_excerpt.wav` (new fixture)
- Create: `tests/cpp/asr/test_diar_identity_handoff.cpp`
- Modify: `tests/cpp/asr/CMakeLists.txt` (register the new test)

**Interfaces:**
- Produces: `DiarStream::poll_speaker_change() -> std::optional<DiarSpeakerChange>` (new, on `DiarStream`); `RecognitionStream::finish(bool finish_diarizer = true) -> Result` (changed signature, default-compatible); `RecognitionStream::extract_diar_stream() -> std::unique_ptr<DiarStream>` (new); `Recognizer::streaming_recognize(AsrRequestOptions, const std::string&, bool coordinate_ingress = false, std::unique_ptr<DiarStream> existing_diar = nullptr) -> std::unique_ptr<RecognitionStream>` (changed signature, default-compatible).
- Consumes: nothing from other tasks (this is the foundational task).

- [ ] **Step 1: Move `SpeakerChangeTracker` onto `DiarStream`**

In `src/asr/diar/diar_pipeline.h`, the class currently reads (starting at the line `class DiarStream {`):
```cpp
// Per-stream streaming state + timeline.
class DiarStream {
   public:
    DiarStream(DiarModel& model, const DiarGeometry& geometry);

    // Feed 16 kHz mono samples; runs any chunks whose right context is
    // covered. Call finish() once at end-of-stream to flush the tail.
    void feed_audio(const float* samples, size_t n_samples);
    void finish();
    void reset();
```
Leave this block unchanged. Find where `segments()` is declared:
```cpp
    using Segment = DiarSegment;
    std::vector<Segment> segments(const DiarSegmentationCfg& cfg = DiarSegmentationCfg()) const;

   private:
```
Insert a new public method immediately after the `segments()` declaration, before `private:`:
```cpp
    using Segment = DiarSegment;
    std::vector<Segment> segments(const DiarSegmentationCfg& cfg = DiarSegmentationCfg()) const;

    // Poll for a confirmed speaker change since the last call on THIS
    // DiarStream instance (or since its construction, for the first
    // call). Lives here (not on RecognitionStream) so that when a
    // RecognitionStream hands this DiarStream to a fresh one via
    // extract_diar_stream()/existing_diar, the "have I already announced
    // a baseline speaker" state travels with it -- otherwise every fresh
    // stream would spuriously re-swallow the first real change after
    // every commit.
    std::optional<DiarSpeakerChange> poll_speaker_change() {
        return speaker_change_tracker_.observe(segments());
    }

   private:
```

Then find the class's data members (ending with `int64_t compact_retain_frames_ = 7500;` just before the closing `};` of the class). Add the tracker as a new private member right after it:
```cpp
    // ~20 min trigger / ~10 min retained at 80 ms frames.
    int64_t compact_trigger_frames_ = 15000;
    int64_t compact_retain_frames_ = 7500;
    SpeakerChangeTracker speaker_change_tracker_;
};
```

- [ ] **Step 2: Update `recognizer.h`**

Change the constructor declaration from:
```cpp
    RecognitionStream(
        Recognizer* recognizer, std::unique_ptr<AsrRunner> runner, AsrRequestOptions options,
        bool coordinate_ingress);
```
to:
```cpp
    RecognitionStream(
        Recognizer* recognizer, std::unique_ptr<AsrRunner> runner, AsrRequestOptions options,
        bool coordinate_ingress, std::unique_ptr<DiarStream> existing_diar = nullptr);
```

Change:
```cpp
    // No more audio: flush the tail and return the end-of-stream final Result.
    Result finish();
    // Poll for a confirmed speaker change since the last call (or since
    // stream start, for the first call). Always returns nullopt if
    // diarization isn't enabled for this stream. See detect_speaker_change()
    // in diar_pipeline.h for the comparison semantics.
    std::optional<DiarSpeakerChange> poll_speaker_change();
```
to:
```cpp
    // No more audio: flush the tail and return the end-of-stream final
    // Result. finish_diarizer=true (default) is a genuine end of audio:
    // the diarizer is closed via DiarStream::finish() and can never
    // accept more (matches every existing caller's expectation). Pass
    // false for a commit boundary within a longer session -- the tail is
    // still correctly tagged (via the same on-demand flush next() already
    // uses for endpointed finals), but the diarizer stays alive and
    // extractable via extract_diar_stream() for a future stream to adopt.
    Result finish(bool finish_diarizer = true);
    // Detaches and returns this stream's diarizer (nullptr if diarization
    // wasn't enabled), leaving this stream's own copy null. Intended to
    // be called right after finish(/*finish_diarizer=*/false), just
    // before this (now-dying) stream is destroyed, so the caller can hand
    // the still-alive DiarStream to the next RecognitionStream via its
    // existing_diar constructor parameter.
    std::unique_ptr<DiarStream> extract_diar_stream() { return std::move(diar_); }
    // Poll for a confirmed speaker change since the last call (or since
    // stream start, for the first call). Always returns nullopt if
    // diarization isn't enabled for this stream. See
    // DiarStream::poll_speaker_change() for the comparison semantics.
    std::optional<DiarSpeakerChange> poll_speaker_change();
```

Change:
```cpp
    // Optional sidecar over the same model-rate audio as ASR.
    std::unique_ptr<DiarStream> diar_;
    SpeakerChangeTracker speaker_change_tracker_;
    AsrRequestOptions opts_;
```
to:
```cpp
    // Optional sidecar over the same model-rate audio as ASR.
    std::unique_ptr<DiarStream> diar_;
    // Seconds of diarizer-timeline time that had already elapsed when
    // diar_ was adopted from a prior stream (0 for a freshly-created
    // diar_). Added to every word time before it's used to query diar_,
    // since diar_'s own internal clock never resets across a handoff but
    // this stream's own AsrRunner always reports word times relative to
    // its own start at 0.
    double diar_time_offset_sec_ = 0.0;
    AsrRequestOptions opts_;
```

Update `Recognizer::streaming_recognize`'s declaration from:
```cpp
    std::unique_ptr<RecognitionStream> streaming_recognize(
        AsrRequestOptions opts, const std::string& language_code, bool coordinate_ingress = false);
```
to:
```cpp
    std::unique_ptr<RecognitionStream> streaming_recognize(
        AsrRequestOptions opts, const std::string& language_code, bool coordinate_ingress = false,
        std::unique_ptr<DiarStream> existing_diar = nullptr);
```

- [ ] **Step 3: Update `recognizer.cpp`'s constructor**

Change:
```cpp
RecognitionStream::RecognitionStream(
    Recognizer* recognizer, std::unique_ptr<AsrRunner> runner, AsrRequestOptions opts,
    bool coordinate_ingress)
    : recognizer_(recognizer), runner_(std::move(runner)), opts_(std::move(opts)),
      coordinate_ingress_(coordinate_ingress) {
    runner_->set_request_options(opts_);
    if (opts_.enable_speaker_diarization) {
        if (recognizer_->diar_model() == nullptr) {
            throw std::invalid_argument(
                "speaker diarization requested but no diarizer model is loaded "
                "(start the server with --diar-model)");
        }
        diar_ = std::make_unique<DiarStream>(
            *recognizer_->diar_model(), recognizer_->config().diar.resolved_geometry());
    }
    if (coordinate_ingress_)
        recognizer_->register_streaming_ingress();
}
```
to:
```cpp
RecognitionStream::RecognitionStream(
    Recognizer* recognizer, std::unique_ptr<AsrRunner> runner, AsrRequestOptions opts,
    bool coordinate_ingress, std::unique_ptr<DiarStream> existing_diar)
    : recognizer_(recognizer), runner_(std::move(runner)), opts_(std::move(opts)),
      coordinate_ingress_(coordinate_ingress) {
    runner_->set_request_options(opts_);
    if (opts_.enable_speaker_diarization) {
        if (existing_diar) {
            diar_time_offset_sec_ =
                existing_diar->n_frames() * existing_diar->seconds_per_frame();
            diar_ = std::move(existing_diar);
        } else {
            if (recognizer_->diar_model() == nullptr) {
                throw std::invalid_argument(
                    "speaker diarization requested but no diarizer model is loaded "
                    "(start the server with --diar-model)");
            }
            diar_ = std::make_unique<DiarStream>(
                *recognizer_->diar_model(), recognizer_->config().diar.resolved_geometry());
        }
    }
    if (coordinate_ingress_)
        recognizer_->register_streaming_ingress();
}
```

- [ ] **Step 4: Update `poll_speaker_change()`**

Change:
```cpp
std::optional<DiarSpeakerChange>
RecognitionStream::poll_speaker_change() {
    if (!diar_)
        return std::nullopt;
    return speaker_change_tracker_.observe(diar_->segments());
}
```
to:
```cpp
std::optional<DiarSpeakerChange>
RecognitionStream::poll_speaker_change() {
    if (!diar_)
        return std::nullopt;
    return diar_->poll_speaker_change();
}
```

- [ ] **Step 5: Apply the time offset at both diarizer query sites**

In `flush_diar_deficit_`, change:
```cpp
void
RecognitionStream::flush_diar_deficit_(const StreamingUpdate& u) {
    if (!diar_ || u.words.empty())
        return;
    const double end_sec = u.words.back().end_frame * recognizer_->ms_per_enc_frame() / 1000.0;
    const auto target = static_cast<int64_t>(std::ceil(end_sec / diar_->seconds_per_frame()));
```
to:
```cpp
void
RecognitionStream::flush_diar_deficit_(const StreamingUpdate& u) {
    if (!diar_ || u.words.empty())
        return;
    const double end_sec = u.words.back().end_frame * recognizer_->ms_per_enc_frame() / 1000.0
                            + diar_time_offset_sec_;
    const auto target = static_cast<int64_t>(std::ceil(end_sec / diar_->seconds_per_frame()));
```

In the per-word tagging inside `make_alt` (`build_result_`), change:
```cpp
                if (tag_speakers && diar_) {
                    // Transducer punctuation can extend a word timestamp into
                    // the next turn. Anchor attribution to the word onset and
                    // average two diar frames to reject single-frame noise.
                    const int spk =
                        diar_->speaker_for_word_time(ww.start_time / 1000.0, ww.end_time / 1000.0);
                    ww.speaker_tag = spk >= 0 ? spk + 1 : 0;
                }
```
to:
```cpp
                if (tag_speakers && diar_) {
                    // Transducer punctuation can extend a word timestamp into
                    // the next turn. Anchor attribution to the word onset and
                    // average two diar frames to reject single-frame noise.
                    // diar_time_offset_sec_ re-aligns this stream's own
                    // (0-based) word clock with diar_'s timeline, which
                    // keeps running across a handoff from a prior stream
                    // (see RecognitionStream's constructor).
                    const int spk = diar_->speaker_for_word_time(
                        ww.start_time / 1000.0 + diar_time_offset_sec_,
                        ww.end_time / 1000.0 + diar_time_offset_sec_);
                    ww.speaker_tag = spk >= 0 ? spk + 1 : 0;
                }
```

- [ ] **Step 6: Update `finish()`**

Change:
```cpp
Result
RecognitionStream::finish() {
    const ScopedBatchCohort cohort_scope(
        pending_cohort_target_ > 0 ? pending_cohort_target_ : current_batch_cohort_target());
    pending_cohort_target_ = 0;
    if (resampler_ && !resampler_flushed_) {
        resampled_audio_.clear();
        resampler_->finish(&resampled_audio_);
        if (!resampled_audio_.empty()) {
            runner_->feed_audio(resampled_audio_.data(), resampled_audio_.size());
            if (diar_)
                diar_->feed_audio(resampled_audio_.data(), resampled_audio_.size());
        }
        resampler_flushed_ = true;
    }
    if (diar_)
        diar_->finish();  // flush the diarizer tail before tagging final words
    auto u = runner_->finalize();
    return build_result_(u, /*is_final=*/true);
}
```
to:
```cpp
Result
RecognitionStream::finish(bool finish_diarizer) {
    const ScopedBatchCohort cohort_scope(
        pending_cohort_target_ > 0 ? pending_cohort_target_ : current_batch_cohort_target());
    pending_cohort_target_ = 0;
    if (resampler_ && !resampler_flushed_) {
        resampled_audio_.clear();
        resampler_->finish(&resampled_audio_);
        if (!resampled_audio_.empty()) {
            runner_->feed_audio(resampled_audio_.data(), resampled_audio_.size());
            if (diar_)
                diar_->feed_audio(resampled_audio_.data(), resampled_audio_.size());
        }
        resampler_flushed_ = true;
    }
    auto u = runner_->finalize();
    if (diar_) {
        if (finish_diarizer) {
            diar_->finish();  // true end of audio: flush the tail, close for good
        } else {
            // Commit boundary within a longer session: tag the tail via
            // the same on-demand mechanism next() already uses for
            // endpointed finals, without closing the diarizer.
            flush_diar_deficit_(u);
        }
    }
    return build_result_(u, /*is_final=*/true);
}
```
(Note: this moves `auto u = runner_->finalize();` to before the diarizer branch — it previously ran after `diar_->finish()`. This reordering is behaviorally neutral: neither `DiarStream::finish()` nor `flush_diar_deficit_` depends on ASR results, only on `diar_`'s own state and, for the new `false` branch, on `u.words` — verified by existing tests staying green in Step 8.)

- [ ] **Step 7: Update `Recognizer::streaming_recognize`**

Change:
```cpp
std::unique_ptr<RecognitionStream>
Recognizer::streaming_recognize(
    AsrRequestOptions opts, const std::string& language_code, bool coordinate_ingress) {
    opts.language_code = language_code;
    auto runner = make_runner();
    log_execution_status(/*streaming=*/true);
    if (model_->has_prompt())
        runner->set_prompt_index(model_->prompt_index_for_lang(language_code));
    return std::make_unique<RecognitionStream>(
        this, std::move(runner), std::move(opts), coordinate_ingress);
}
```
to:
```cpp
std::unique_ptr<RecognitionStream>
Recognizer::streaming_recognize(
    AsrRequestOptions opts, const std::string& language_code, bool coordinate_ingress,
    std::unique_ptr<DiarStream> existing_diar) {
    opts.language_code = language_code;
    auto runner = make_runner();
    log_execution_status(/*streaming=*/true);
    if (model_->has_prompt())
        runner->set_prompt_index(model_->prompt_index_for_lang(language_code));
    return std::make_unique<RecognitionStream>(
        this, std::move(runner), std::move(opts), coordinate_ingress, std::move(existing_diar));
}
```

- [ ] **Step 8: Build and confirm existing tests are unaffected**

```bash
git submodule update --init ggml llama.cpp   # first time only, if build/cpu-asr doesn't exist yet
scripts/configure.sh cpu-asr -DNEMO_SPEECH_BUILD_TESTS=ON   # first time only
cmake --build --preset cpu-asr --target test_diar_speaker_change
ctest --test-dir build/cpu-asr --output-on-failure
```
Expected: all existing tests pass unchanged (this is the primary evidence that every default-parameter path — `finish()`, `finish(true)`, `streaming_recognize` without `existing_diar` — behaves exactly as before, including through the `finish()` reordering in Step 6).

- [ ] **Step 9: Add the real multi-speaker fixture**

No genuinely multi-speaker audio fixture exists anywhere in this repo or the companion `NemoSpeech` project (confirmed during the previous PR's work — every existing wav in both repos turned out to be a byte-identical duplicate of the same single-speaker `jfk.wav`). Add one: a 90-second excerpt of public-domain U.S. Supreme Court oral argument audio (case 08-1314), sourced from archive.org's official `SCOTUSOralArugments` collection (U.S. government work product, no copyright restriction). Verified during planning: streamed through the standalone `diarize` CLI, it produces 3 distinct, stable speakers with a long (7.2s-81.4s) continuous middle segment from one speaker — ideal for a handoff test that needs the same real speaker active well before and after a chosen split point.

```bash
mkdir -p test_files/asr/wav/test
ffmpeg -hide_banner -loglevel warning \
  -i "https://archive.org/download/SCOTUSOralArugments/08-1314.mp3" \
  -t 90 -ar 16000 -ac 1 -c:a pcm_s16le \
  test_files/asr/wav/test/scotus_08-1314_excerpt.wav
```
Verify it: `ffprobe test_files/asr/wav/test/scotus_08-1314_excerpt.wav` should report `Duration: 00:01:30.00`, `pcm_s16le, 16000 Hz, 1 channels`.

- [ ] **Step 10: Write the core regression test**

Create `tests/cpp/asr/test_diar_identity_handoff.cpp`:

```cpp
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Regression test for persistent diarizer identity across a realtime
// commit boundary: split one real multi-speaker recording into two
// RecognitionStreams (mirroring what http_server.cpp does on every
// input_audio_buffer.commit), hand the first stream's DiarStream to the
// second via extract_diar_stream()/existing_diar, and confirm the
// speaker actively talking at the split point keeps the same speaker
// number across it -- the property this fix establishes. Also confirms
// poll_speaker_change() (now living on DiarStream) still correctly
// detects a genuine later transition after the handoff.
//
// Usage: test_diar_identity_handoff <asr.gguf> <diar.gguf> <audio.wav> [--gpu N] [--split-sec N]

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "fe.h"
#include "recognizer.h"

using namespace nemo_speech::asr;

namespace {

struct TaggedWord {
    std::string word;
    int speaker_tag;
};

void
collect(const Result& r, std::vector<TaggedWord>& out) {
    if (r.is_final && !r.alternatives.empty())
        for (const auto& w : r.alternatives[0].words) out.push_back({w.word, w.speaker_tag});
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(
            stderr, "usage: %s <asr.gguf> <diar.gguf> <audio.wav> [--gpu N] [--split-sec N]\n",
            argv[0]);
        return 2;
    }
    RecognizerConfig cfg;
    cfg.model.path = argv[1];
    cfg.diar.model_path = argv[2];
    const std::string wav_path = argv[3];
    cfg.backend.gpu = -1;
    double split_sec = 40.0;
    for (int i = 4; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--gpu" && i + 1 < argc)
            cfg.backend.gpu = std::atoi(argv[++i]);
        else if (a == "--split-sec" && i + 1 < argc)
            split_sec = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    std::vector<float> audio;
    int sr = 0;
    if (!read_wav_mono_16k(wav_path, audio, sr) || sr != 16000) {
        std::fprintf(stderr, "failed to read 16 kHz mono wav: %s\n", wav_path.c_str());
        return 1;
    }
    const size_t split_sample = static_cast<size_t>(split_sec * sr);
    if (split_sample == 0 || split_sample >= audio.size()) {
        std::fprintf(
            stderr, "--split-sec %.2f is out of range for a %.2fs file\n", split_sec,
            static_cast<double>(audio.size()) / sr);
        return 2;
    }

    Recognizer rec(cfg);
    AsrRequestOptions opts;
    opts.enable_speaker_diarization = true;

    const size_t push = 160 * 16;  // 160 ms

    // ---- Stream 1: audio[0, split) -----------------------------------
    auto stream1 = rec.streaming_recognize(opts, "");
    std::vector<TaggedWord> words1;
    for (size_t off = 0; off < split_sample; off += push) {
        stream1->push(audio.data() + off, std::min(push, split_sample - off));
        while (auto r = stream1->next()) {
            collect(*r, words1);
            if (!r->is_final)
                break;
        }
    }
    // finish_diarizer=false: this is a commit boundary within a longer
    // session, not the true end of audio -- the diarizer must survive it.
    collect(stream1->finish(/*finish_diarizer=*/false), words1);

    auto extracted = stream1->extract_diar_stream();
    if (!extracted) {
        std::fprintf(stderr, "[identity-handoff] FAIL: extract_diar_stream() returned null\n");
        return 1;
    }
    stream1.reset();

    if (words1.empty() || words1.back().speaker_tag <= 0) {
        std::fprintf(stderr, "[identity-handoff] FAIL: no tagged words before the split\n");
        return 1;
    }
    const int speaker_before = words1.back().speaker_tag;

    // ---- Stream 2: audio[split, end), adopting stream 1's diarizer ---
    auto stream2 =
        rec.streaming_recognize(opts, "", /*coordinate_ingress=*/false, std::move(extracted));
    std::vector<TaggedWord> words2;
    int speaker_changes2 = 0;
    for (size_t off = split_sample; off < audio.size(); off += push) {
        stream2->push(audio.data() + off, std::min(push, audio.size() - off));
        while (auto r = stream2->next()) {
            collect(*r, words2);
            if (!r->is_final)
                break;
        }
        if (stream2->poll_speaker_change())
            speaker_changes2++;
    }
    collect(stream2->finish(), words2);  // true end of audio: default finish_diarizer=true

    if (words2.empty() || words2.front().speaker_tag <= 0) {
        std::fprintf(stderr, "[identity-handoff] FAIL: no tagged words after the split\n");
        return 1;
    }
    const int speaker_after = words2.front().speaker_tag;

    std::printf(
        "[identity-handoff] speaker at split: before=%d after=%d, %d speaker change(s) "
        "detected after the split\n",
        speaker_before, speaker_after, speaker_changes2);

    if (speaker_before != speaker_after) {
        std::printf(
            "[identity-handoff] FAIL: speaker identity did not survive the handoff (%d -> %d)\n",
            speaker_before, speaker_after);
        return 1;
    }
    if (speaker_changes2 < 1) {
        std::printf(
            "[identity-handoff] FAIL: expected at least one genuine speaker change after the "
            "split (the fixture's later speaker transition), got none\n");
        return 1;
    }
    std::printf("[identity-handoff] OK\n");
    return 0;
}
```

Register it in `tests/cpp/asr/CMakeLists.txt`, right after the `test_diar_recognizer` block:
```cmake
add_executable(test_diar_identity_handoff test_diar_identity_handoff.cpp)
target_link_libraries(test_diar_identity_handoff PRIVATE nemo_speech_asr)
```
(No `add_test`: like `test_diar_recognizer`, this needs real model/wav arguments and is a manual/local tool, not part of default `ctest`.)

- [ ] **Step 11: Build and run the regression test**

```bash
cmake --build --preset cpu-asr --target test_diar_identity_handoff
./build/cpu-asr/tests/cpp/asr/test_diar_identity_handoff \
  <asr.gguf> <diar.gguf> test_files/asr/wav/test/scotus_08-1314_excerpt.wav
```
Real ASR/diarizer GGUF paths: check `~/.cache/nemo-speech/models/` (e.g.
`nvidia/nemotron-3.5-asr-streaming-0.6b/.../nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` and
`nvidia/diar_streaming_sortformer_4spk-v2/.../diar_streaming_sortformer_4spk-v2.q8_0.gguf`
were both confirmed present and working during planning). Expected: `[identity-handoff] OK`,
with the printed `before=`/`after=` speaker numbers equal.

**If it fails with `before != after`:** do not weaken the assertion or the fixture choice — this is exactly the bug this task exists to fix. Re-check Steps 3-7 against this plan text precisely (a missed `+ diar_time_offset_sec_` at either query site, or a constructor path that doesn't actually adopt `existing_diar`, are the most likely causes).

- [ ] **Step 12: Commit**

```bash
git add src/asr/diar/diar_pipeline.h src/asr/recognizer.h src/asr/recognizer.cpp \
        test_files/asr/wav/test/scotus_08-1314_excerpt.wav \
        tests/cpp/asr/test_diar_identity_handoff.cpp tests/cpp/asr/CMakeLists.txt
git commit -m "feat(asr): persist diarizer identity across a realtime commit boundary"
```

---

## Task 2: Wire the server to use the persistent diarizer

**Files:**
- Modify: `server/http/http_server.cpp` (realtime handler: `ensure_stream`, `input_audio_buffer.commit`, `input_audio_buffer.clear`/`response.cancel`)
- Modify: `tests/integration/http_conformance_test.py` (new cross-commit identity assertion)

**Interfaces:**
- Consumes: `RecognitionStream::finish(bool)`, `extract_diar_stream()`, `Recognizer::streaming_recognize(..., existing_diar)` (Task 1).
- Produces: no new public interface — this task makes the existing wire protocol's word-level `speaker` tags finally consistent across commits within one connection. No event shape changes.

- [ ] **Step 1: Add connection-scoped persistent-diarizer state**

In `server/http/http_server.cpp`'s realtime handler, find:
```cpp
            auto recognizer = this->models.asr();
            asr::AsrRequestOptions options;
            options.enable_automatic_punctuation = true;
            int sample_rate = recognizer->sample_rate();
            std::unique_ptr<asr::RecognitionStream> stream;
            std::string partial_transcript;
            size_t audio_bytes = 0;
```
Add one line:
```cpp
            auto recognizer = this->models.asr();
            asr::AsrRequestOptions options;
            options.enable_automatic_punctuation = true;
            int sample_rate = recognizer->sample_rate();
            std::unique_ptr<asr::RecognitionStream> stream;
            std::unique_ptr<asr::DiarStream> persistent_diar;
            std::string partial_transcript;
            size_t audio_bytes = 0;
```

- [ ] **Step 2: Thread it through `ensure_stream()`**

Change:
```cpp
            auto ensure_stream = [&] {
                if (!stream)
                    stream = recognizer->streaming_recognize(
                        options, options.language_code, /*coordinate_ingress=*/true);
            };
```
to:
```cpp
            auto ensure_stream = [&] {
                if (!stream)
                    stream = recognizer->streaming_recognize(
                        options, options.language_code, /*coordinate_ingress=*/true,
                        std::move(persistent_diar));
            };
```

- [ ] **Step 3: Extract before resetting on commit**

Change:
```cpp
                    } else if (type == "input_audio_buffer.commit") {
                        ensure_stream();
                        if (!emit(stream->finish()))
                            break;
                        // poll_speaker_change() is deliberately not called after finish():
                        // any speaker change confirmed only in the flushed tail goes
                        // unreported by this event, but is still captured correctly by
                        // the completed event's own word-level speaker tags above.
                        stream.reset();
                        audio_bytes = 0;
```
to:
```cpp
                    } else if (type == "input_audio_buffer.commit") {
                        ensure_stream();
                        if (!emit(stream->finish(/*finish_diarizer=*/false)))
                            break;
                        // poll_speaker_change() is deliberately not called after finish():
                        // any speaker change confirmed only in the flushed tail goes
                        // unreported by this event, but is still captured correctly by
                        // the completed event's own word-level speaker tags above.
                        // finish_diarizer=false + extract_diar_stream(): the diarizer's
                        // speaker identity survives this commit and is handed to the
                        // next stream via ensure_stream() above, instead of being
                        // destroyed and rebuilt from scratch on every single commit.
                        persistent_diar = stream->extract_diar_stream();
                        stream.reset();
                        audio_bytes = 0;
```

- [ ] **Step 4: Extract on clear/cancel too**

Change:
```cpp
                    } else if (type == "input_audio_buffer.clear" || type == "response.cancel") {
                        stream.reset();
                        partial_transcript.clear();
                        audio_bytes = 0;
```
to:
```cpp
                    } else if (type == "input_audio_buffer.clear" || type == "response.cancel") {
                        // Discarding buffered *transcript* audio doesn't mean discarding
                        // *voice identity* tracking -- extract here too, same as commit.
                        if (stream)
                            persistent_diar = stream->extract_diar_stream();
                        stream.reset();
                        partial_transcript.clear();
                        audio_bytes = 0;
```

- [ ] **Step 5: Build**

```bash
cmake --build --preset cpu-asr --target nemo_speech_cli
```
(If this fails because HTTP support isn't compiled into this build config, reconfigure with `-DNEMO_SPEECH_BUILD_HTTP=ON` first, then rebuild — this was needed once before during this project's prior work on the same file.)

- [ ] **Step 6: Write the failing protocol-level assertion**

In `tests/integration/http_conformance_test.py`, find the WebSocket section that streams `pcm` and waits for `input_audio_buffer.committed` (search for `websocket_url = f"ws://127.0.0.1:{port}/v1/audio/transcriptions/realtime`). After that existing block, add a new one specifically exercising a two-commit handoff, using the new fixture from Task 1 (not `args.audio`, since this needs that fixture's specific known multi-speaker structure):

```python
        if args.diar_model:
            fixture_path = (
                Path(__file__).resolve().parent.parent.parent
                / "test_files" / "asr" / "wav" / "test" / "scotus_08-1314_excerpt.wav"
            )
            with wave.open(str(fixture_path), "rb") as audio:
                sample_rate2 = audio.getframerate()
                pcm2 = audio.readframes(audio.getnframes())
            split_byte = int(40.0 * sample_rate2) * 2  # 40s, matches the C++ test's default
            chunk2 = max(2, sample_rate2 * 2 // 5)
            events2 = []
            with connect(
                websocket_url, open_timeout=30, close_timeout=10, ping_interval=None
            ) as websocket:
                events2.append(json.loads(websocket.recv()))
                websocket.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "sample_rate": sample_rate2,
                                "word_timestamps": True,
                                "speaker_diarization": True,
                            },
                        }
                    )
                )
                for offset in range(0, split_byte, chunk2):
                    websocket.send(pcm2[offset : offset + chunk2])
                websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
                while True:
                    event = json.loads(websocket.recv())
                    events2.append(event)
                    if event.get("type") == "input_audio_buffer.committed":
                        break
                for offset in range(split_byte, len(pcm2), chunk2):
                    websocket.send(pcm2[offset : offset + chunk2])
                websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
                while True:
                    event = json.loads(websocket.recv())
                    events2.append(event)
                    if event.get("type") == "input_audio_buffer.committed":
                        break
            completed2 = [
                e for e in events2 if e.get("type", "").endswith("transcription.completed")
            ]
            require(len(completed2) >= 2, "two commits produced two completed events")
            words_before = completed2[0].get("words") or []
            words_after = completed2[1].get("words") or []
            require(words_before and words_before[-1].get("speaker", 0) > 0,
                    "first commit has a tagged word before the split")
            require(words_after and words_after[0].get("speaker", 0) > 0,
                    "second commit has a tagged word after the split")
            require(
                words_before[-1]["speaker"] == words_after[0]["speaker"],
                "speaker identity persists across a commit boundary",
            )
```

Add `from pathlib import Path` to the file's imports if not already present (check the existing import block first).

- [ ] **Step 7: Run it to verify it fails on the pre-fix code, then passes**

`git stash` Step 3 and Step 4's edits to `http_server.cpp` (the `ensure_stream`/`commit`/`clear` wiring — leave Step 1's `persistent_diar` declaration in place, or stash that too, whichever `git stash` groups naturally), rebuild (`cmake --build --preset cpu-asr --target nemo_speech_cli`), and run the conformance test below — the new `require(...) speaker identity persists...` line should fail, since `http_server.cpp` still destroys the diarizer every commit regardless of what `RecognitionStream` now supports. Then `git stash pop`, rebuild again, and confirm it passes:
```bash
python3 tests/integration/http_conformance_test.py \
  --binary build/cpu-asr/bin/nemo-speech \
  --asr-model <asr.gguf> \
  --diar-model <diar.gguf> \
  --audio test_files/asr/wav/test/jfk.wav
```
(The pre-existing `--audio` argument is still required by the script for its other checks; the new section reads the SCOTUS fixture itself via `Path(__file__)`, independent of `--audio`.) Expected: all checks pass, including the new one.

- [ ] **Step 8: Commit**

```bash
git add server/http/http_server.cpp tests/integration/http_conformance_test.py
git commit -m "feat(server): keep the diarizer alive across realtime commits"
```

---

## Task 3: Fork divergence log

**Files:**
- Modify: `FORK_CHANGES.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing consumed by later tasks — this is the last task.

- [ ] **Step 1: Add a new divergence entry**

Append to `FORK_CHANGES.md`, following the exact structure of the existing entries (run `git merge-base feat/persistent-diarizer-identity main` to confirm the upstream base commit before filling it in):

```markdown
### 2026-09-13 — Persistent diarizer identity across realtime commits

- **Upstream base:** (fill in from `git merge-base feat/persistent-diarizer-identity main`)
- **PR:** (pending — opened by the controller after this task)
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
```

- [ ] **Step 2: Commit**

```bash
git add FORK_CHANGES.md
git commit -m "docs: log the persistent-diarizer-identity divergence"
```

(Push, PR, and merge are handled by the controller after this plan completes, per this project's established process for this repo — not part of this task.)
