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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

    // The adoption offset must track audio actually fed, not any
    // persistently-committed subset of it -- frame indices are absolute
    // over fed audio, so an offset derived from committed_frames() (which
    // excludes the unconsumed tail) would resolve every word in the
    // adopting stream too early, into the previous turn at a commit
    // boundary. Confirm fed_audio_sec() tracks the split point itself,
    // independent of chunk/right-context geometry.
    const double fed_sec = extracted->fed_audio_sec();
    const double frame_sec = extracted->seconds_per_frame();
    if (std::fabs(fed_sec - split_sec) > frame_sec) {
        std::fprintf(
            stderr,
            "[identity-handoff] FAIL: fed_audio_sec() %.4f is more than one frame (%.4f) away "
            "from split_sec %.4f -- the handoff offset would not track fed audio\n",
            fed_sec, frame_sec, split_sec);
        return 1;
    }
    std::printf(
        "[identity-handoff] timeline at split: fed_audio_sec=%.4f split_sec=%.4f "
        "(seconds_per_frame=%.4f)\n",
        fed_sec, split_sec, frame_sec);
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
