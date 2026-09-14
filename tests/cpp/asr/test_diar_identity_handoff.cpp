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
