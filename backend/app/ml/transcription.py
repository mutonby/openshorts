"""faster-whisper transcription: CPU-optimized (INT8 quantization) with word timestamps."""


def transcribe_video(video_path):
    print("🎙️  Transcribing video with Faster-Whisper (CPU Optimized)...")
    from faster_whisper import WhisperModel

    # Run on CPU with INT8 quantization for speed
    model = WhisperModel("base", device="cpu", compute_type="int8")

    segments, info = model.transcribe(video_path, word_timestamps=True)

    print(f"   Detected language '{info.language}' with probability {info.language_probability:.2f}")

    # Convert to openai-whisper compatible format
    transcript_segments = []
    full_text = ""

    for segment in segments:
        # Print progress to keep user informed (and prevent timeouts feeling)
        print(f"   [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        seg_dict = {
            'text': segment.text,
            'start': segment.start,
            'end': segment.end,
            'words': []
        }

        if segment.words:
            for word in segment.words:
                seg_dict['words'].append({
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'probability': word.probability
                })

        transcript_segments.append(seg_dict)
        full_text += segment.text + " "

    return {
        'text': full_text.strip(),
        'segments': transcript_segments,
        'language': info.language
    }
