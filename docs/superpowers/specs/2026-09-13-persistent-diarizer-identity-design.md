# Persistent diarizer identity across realtime commits

## Problem

`NemoSpeech` (the companion project driving this server for live captions)
recently added client-side logic that commits immediately on a confirmed
speaker change (`conversation.item.speaker_diarization.changed`,
`bsips/NeMo-Speech.cpp#2`). This surfaced a pre-existing problem in the
realtime WebSocket handler that used to be rare (commits only every ~15s)
and is now common (commits on every genuine speaker turn, often every
1-2 seconds): **speaker identity does not survive a commit.**

`server/http/http_server.cpp`'s `input_audio_buffer.commit` handler does:

```cpp
} else if (type == "input_audio_buffer.commit") {
    ensure_stream();
    if (!emit(stream->finish())) break;
    stream.reset();   // destroys the whole RecognitionStream, including diar_
```

`stream.reset()` destroys the entire `RecognitionStream`, including its
owned `DiarStream` — the diarizer's speaker-embedding cache and timeline.
The next commit's `ensure_stream()` builds a **brand-new** `DiarStream`
from scratch, with no memory of who "speaker 1/2/3" were a moment ago. So
"Speaker 2" in one caption line has no relationship to "Speaker 2" in the
next: two different real people can both get labeled "Speaker 1" (each
being the first voice heard in their own fresh stream), or the same
person can silently become "Speaker 2" moments later.

**Verified this is not a model-quality problem.** A real 90-second,
public-domain multi-speaker recording (Supreme Court oral argument audio,
case 08-1314, from archive.org's official SCOTUS collection — see
Verification below), run as **one continuous stream** through the
standalone `diarize` CLI under both `streaming` and `offline` presets,
produced nearly identical segment boundaries (within ~0.1-0.2s of each
other) and consistent speaker numbering throughout. The model correctly
tracks distinct voices within a continuous session, regardless of preset.
The bug is entirely in the realtime handler's per-commit stream teardown.

## Why simply "don't destroy the stream" doesn't work

`RecognitionStream::finish()` (`src/asr/recognizer.cpp:524`, called by the
commit handler above) does this unconditionally:

```cpp
if (diar_)
    diar_->finish();  // flush the diarizer tail before tagging final words
```

`DiarStream::finish()` (`src/asr/diar/diar_pipeline.cpp:119`) is not a
harmless flush — it sets `finished_ = true` (after which "pushes after
finish are silent no-ops", per `diar_pipeline.h`'s own doc comment) and
applies genuine end-of-audio edge-padding math (a geometric-decay tail
matching NeMo's `torch.stft(center=True)` semantics) that is only correct
when there truly is no more audio coming. Simply keeping the same
`DiarStream` object alive and feeding it more audio after `finish()`
would either silently drop that audio (no-op) or, if that guard were
removed, create a real encoder-input discontinuity at every single commit
boundary. This isn't a shortcut we can skip past — the fix has to change
*what* the commit handler asks the diarizer to do, not just how long the
object lives.

## Approach

**The tool already exists.** `flush_diar_deficit_()`
(`src/asr/recognizer.cpp:481`) already solves "tag the last few words
correctly without ending the stream" — it's used today for mid-stream
endpointed finals — by calling `DiarStream::flush_available(target)`
instead of `finish()`. `flush_available`'s own doc comment
(`diar_pipeline.h`) confirms this: "label already-arrived audio early...
the stream continues normally afterwards."

The fix gives the realtime handler a way to opt into that same mechanism
at commit time, and a way to carry the still-alive `DiarStream` forward
into the next `RecognitionStream`.

**Why this shape, not alternatives considered:**

- *Never destroy `RecognitionStream` at all, keep one instance for the
  whole connection* — rejected: the ASR side (`AsrRunner`) genuinely
  needs a fresh session per utterance/commit (streaming
  transducer/CTC decoder hypothesis state resets between utterances);
  only the diarizer's identity needs to survive.
- *Call `DiarStream::reset()` instead of `finish()`* — rejected: `reset()`
  wipes the timeline/speaker-cache back to blank, which is the exact
  problem this fix removes, not a solution to it.
- *Give `DiarStream` its own detach-and-resume API distinct from
  `flush_available`* — rejected: `flush_available` already does exactly
  the needed "tag the tail, stay alive" operation and is already
  exercised by existing endpointing tests; reusing it means no new
  diarizer-level logic, only new plumbing one layer up.

## API changes (all additive / default-parameter-preserving)

All existing call sites (offline/batch `Recognizer::recognize()`,
`test_diar_recognizer.cpp`, any other caller) keep compiling and behaving
identically without changes, because every new parameter defaults to
today's behavior.

1. **`RecognitionStream::finish(bool finish_diarizer = true)`**
   (`recognizer.h:85`, implementation `recognizer.cpp:524`) — new
   parameter, computed **after** `runner_->finalize()` produces the final
   words (reordered from today's before-finalize call; behaviorally
   neutral since `DiarStream::finish()`/`flush_available()` don't depend
   on ASR results, only verified by existing test coverage staying
   green):
   ```cpp
   Result
   RecognitionStream::finish(bool finish_diarizer) {
       ... (resampler flush unchanged) ...
       auto u = runner_->finalize();
       if (diar_) {
           if (finish_diarizer) {
               diar_->finish();
           } else {
               flush_diar_deficit_(u);
           }
       }
       return build_result_(u, /*is_final=*/true);
   }
   ```
   `true` (default): identical to today. `false`: tags the tail words via
   the same on-demand mechanism `next()` already uses for endpointed
   finals, and leaves `diar_` alive and accepting more audio.

2. **`RecognitionStream::extract_diar_stream()`** — new method, detaches
   and returns `std::move(diar_)` (type `std::unique_ptr<DiarStream>`),
   leaving this (about-to-be-destroyed) stream's own `diar_` null. Named
   to avoid confusion with `std::unique_ptr::release()`'s different
   raw-pointer-returning semantics.

3. **`RecognitionStream`'s constructor** (`recognizer.h:67`,
   `recognizer.cpp:318`) and **`Recognizer::streaming_recognize(...)`**
   (`recognizer.h:128`, `recognizer.cpp:545`) both gain a new optional
   trailing parameter: `std::unique_ptr<DiarStream> existing_diar =
   nullptr`. When non-null, the constructor adopts it directly instead of
   building a fresh `DiarStream` from `recognizer_->diar_model()`:
   ```cpp
   if (opts_.enable_speaker_diarization) {
       if (existing_diar) {
           diar_ = std::move(existing_diar);
       } else {
           if (recognizer_->diar_model() == nullptr) throw ...;
           diar_ = std::make_unique<DiarStream>(...);
       }
   }
   ```

   **Critical correctness detail, found during implementation planning
   (not in the original approach above): word timestamps and diarizer
   timestamps will desync across a handoff unless corrected.** The
   adopted `DiarStream`'s internal frame timeline keeps counting from its
   true start (that's the whole point — it never resets), but a brand new
   `RecognitionStream`'s `AsrRunner` always reports word timestamps
   relative to *that stream's own* start at 0. Word-level diarization
   tagging (`recognizer.cpp:454-460`, `diar_->speaker_for_word_time(...)`)
   and on-demand tail-flushing (`flush_diar_deficit_`, `recognizer.cpp:481`)
   both feed ASR-relative word times directly into the diarizer's
   absolute timeline today — correct only because, until now, both clocks
   always started together at stream construction.

   Fix: when adopting `existing_diar`, derive and store how much
   diarizer-time had already elapsed, and add it to every word time
   before it reaches the diarizer (not to the word's own reported
   `start_time`/`end_time` — those stay stream-relative, matching
   existing API/display behavior downstream):
   ```cpp
   // New private member, recognizer.h: double diar_time_offset_sec_ = 0.0;
   if (opts_.enable_speaker_diarization) {
       if (existing_diar) {
           diar_time_offset_sec_ = existing_diar->n_frames() * existing_diar->seconds_per_frame();
           diar_ = std::move(existing_diar);
       } else {
           ...
       }
   }
   ```
   Then at both `diar_` query sites, add `diar_time_offset_sec_` to the
   word time(s) passed in (only at the query call, `ww.start_time`/
   `ww.end_time` themselves are untouched):
   ```cpp
   // recognizer.cpp:458-459
   const int spk = diar_->speaker_for_word_time(
       ww.start_time / 1000.0 + diar_time_offset_sec_,
       ww.end_time / 1000.0 + diar_time_offset_sec_);
   ```
   ```cpp
   // recognizer.cpp:484, flush_diar_deficit_
   const double end_sec = u.words.back().end_frame * recognizer_->ms_per_enc_frame() / 1000.0
                           + diar_time_offset_sec_;
   ```
   No caller-facing change: this is entirely internal to
   `RecognitionStream`, derived automatically from the adopted
   `DiarStream`'s own state at the moment of adoption. Neither
   `http_server.cpp` nor `streaming_recognize`'s other callers need to
   know or pass anything extra.

4. **`SpeakerChangeTracker` moves from `RecognitionStream` onto
   `DiarStream`.** Currently (`recognizer.h`) `RecognitionStream` owns a
   `SpeakerChangeTracker speaker_change_tracker_` member and
   `poll_speaker_change()` (`recognizer.cpp:385`) calls
   `speaker_change_tracker_.observe(diar_->segments())`. This has to move
   because if the diarizer's speaker cache now persists across commits,
   the tracker's "don't announce the very first observation" state must
   persist right alongside it — otherwise **every new stream would
   spuriously re-swallow the first real change after every single
   commit**, silently defeating the speaker-change-commit feature this
   fix is meant to make finally work correctly. Concretely:
   - `DiarStream` (`diar_pipeline.h:182`) gains a private
     `SpeakerChangeTracker speaker_change_tracker_;` member and a public
     `std::optional<DiarSpeakerChange> poll_speaker_change() { return
     speaker_change_tracker_.observe(segments()); }`.
   - `RecognitionStream` loses its own `speaker_change_tracker_` member;
     `RecognitionStream::poll_speaker_change()` becomes:
     ```cpp
     std::optional<DiarSpeakerChange>
     RecognitionStream::poll_speaker_change() {
         if (!diar_) return std::nullopt;
         return diar_->poll_speaker_change();
     }
     ```
   - The standalone `SpeakerChangeTracker` class and its existing unit
     tests (`tests/cpp/asr/test_diar_speaker_change.cpp`) are unchanged —
     this only changes *where an instance of it lives*, not its own
     logic or tests.

## Server wiring (`server/http/http_server.cpp`)

The realtime handler's per-connection state (currently `stream`,
`partial_transcript`, `audio_bytes`, etc., all local to the WS handler's
closure — see `http_server.cpp` around line 1026 onward) gains one more
variable:

```cpp
std::unique_ptr<asr::DiarStream> persistent_diar;
```

- **`ensure_stream()`** (`http_server.cpp:1033`) passes it through:
  ```cpp
  auto ensure_stream = [&] {
      if (!stream)
          stream = recognizer->streaming_recognize(
              options, options.language_code, /*coordinate_ingress=*/true,
              std::move(persistent_diar));
  };
  ```
- **`input_audio_buffer.commit`** (`http_server.cpp:1173`): call
  `stream->finish(/*finish_diarizer=*/false)` instead of `stream->finish()`,
  then extract before resetting:
  ```cpp
  ensure_stream();
  if (!emit(stream->finish(/*finish_diarizer=*/false))) break;
  persistent_diar = stream->extract_diar_stream();
  stream.reset();
  ```
- **`input_audio_buffer.clear` / `response.cancel`**
  (`http_server.cpp:1187`): discarding the buffered *transcript* doesn't
  mean discarding *voice identity* tracking — extract before resetting
  here too:
  ```cpp
  if (stream) persistent_diar = stream->extract_diar_stream();
  stream.reset();
  ```
  (`extract_diar_stream()` on a non-diarized stream's `diar_ == nullptr`
  just returns `nullptr`, a harmless no-op reassignment.)

No changes to `session.update`, `emit()`, the `delta`/`completed` event
construction, or anything in `NemoSpeech`/`live_bridge.py` — the wire
protocol and event shapes are completely unchanged. This is a pure
server-side correctness fix.

## Testing

- **Unit tests for `finish(bool)`'s two branches**: extend
  `tests/cpp/asr/test_diar_recognizer.cpp`'s pattern (or a focused new
  test) to confirm `finish()`/`finish(true)` behaves identically to
  today (existing tests passing unchanged is the primary evidence here),
  and that `finish(false)` correctly tags tail words while leaving
  `diar_` non-null and still willing to accept `push()`.
- **The core regression test**: a new end-to-end test proving
  cross-instance speaker identity — construct a `RecognitionStream`, feed
  real audio from one real speaker, `finish(false)`, `extract_diar_stream()`,
  construct a **second** `RecognitionStream` adopting that extracted
  stream, feed more audio continuing from the **same** speaker (well past
  the handoff point, in both timestamp-seconds and audio-content terms),
  and assert the speaker number is unchanged across that boundary (this is
  the property that did not hold before this fix, and is exactly what a
  plain before/after diff of this test would show). This same test is
  also the natural place to verify the timestamp-offset correction above:
  if it were missing or wrong, word-to-speaker tagging right after the
  handoff would silently point at the wrong slice of the diarizer's
  timeline, which a real multi-speaker fixture spanning the handoff (see
  below) will expose as a wrong or missing tag, not just a wrong number.
- **Fixture**: commit the verified real clip used above — a 90-second,
  16kHz mono PCM16 excerpt of public-domain SCOTUS oral argument audio
  (case 08-1314, sourced from archive.org's official
  `SCOTUSOralArugments` collection, U.S. government work product) — to
  `test_files/asr/wav/test/` alongside the existing `jfk.wav`, replacing
  reliance on ad hoc `/tmp` files for this and future multi-speaker
  tests. This also finally gives this repo a real multi-speaker fixture,
  closing a gap flagged during the previous PR (`#2`)'s work: every
  existing wav in both this repo and `NemoSpeech` turned out to be
  byte-identical duplicates of the same single-speaker `jfk.wav`.
- **Protocol-level test**: extend `tests/integration/http_conformance_test.py`'s
  WebSocket section to send audio, commit, send more audio from the
  fixture's same speaker, commit again, and assert the `speaker` field in
  both `completed` events' `words` refers to the same person (using the
  new fixture makes this assertion meaningful for the first time — it
  was not previously possible to write given the fixture-duplication gap).

## Non-goals / explicitly out of scope

- Any change to `NemoSpeech`/`live_bridge.py` or the wire protocol —
  purely a server-side fix; the client already just reads the words'
  `speaker` tags it's always read.
- The separate, already-documented known limitation that
  `SpeakerChangeTracker`/`detect_speaker_change` compare against
  `segments().back()` (the most recently-*started* segment, which can
  misattribute the active speaker during genuinely overlapping speech) —
  unrelated to identity persistence, still an open item from the
  previous spec.
- Tuning the diarizer's own accuracy knobs (`chunk_len`, `right_context`,
  `onset`/`offset`, `min_duration_on`/`off`) — this fix corrects an
  identity-tracking bug that exists at any geometry/threshold setting;
  it's a prerequisite for any future tuning work to even be measurable
  (before this fix, apparent "bad diarization" during live testing was
  conflating this bug with actual model/threshold behavior).
