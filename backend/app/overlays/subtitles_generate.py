"""SRT subtitle generation: transcription and word-level grouping into short lines."""

import os
import subprocess


def transcribe_audio(video_path):
    """
    Transcribe audio from a video file using faster-whisper.
    Returns transcript in the same format as main.py for compatibility.
    """
    from faster_whisper import WhisperModel

    print(f"🎙️  Transcribing audio from: {video_path}")

    # Run on CPU with INT8 quantization for speed
    model = WhisperModel("base", device="cpu", compute_type="int8")

    segments, info = model.transcribe(video_path, word_timestamps=True)

    transcript = {
        "segments": [],
        "language": info.language
    }

    for segment in segments:
        seg_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": []
        }
        if segment.words:
            for word in segment.words:
                seg_data["words"].append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                })
        transcript["segments"].append(seg_data)

    print(f"✅ Transcription complete. Language: {info.language}")
    return transcript


def _apply_text_case(text: str, text_case: str) -> str:
    """Brand-kit case transform: 'upper' / 'lower' / anything else = original."""
    if text_case == "upper":
        return text.upper()
    if text_case == "lower":
        return text.lower()
    return text


def generate_srt_from_video(video_path, output_path, max_chars=20, max_duration=2.0, max_words=None, text_case="original"):
    """
    Transcribe a video and generate SRT directly.
    Used for dubbed videos that don't have a pre-existing transcript.
    """
    transcript = transcribe_audio(video_path)

    # Get video duration to use as clip_end
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0
    cap.release()

    return generate_srt(transcript, 0, duration, output_path, max_chars, max_duration, max_words=max_words, text_case=text_case)


def generate_srt(transcript, clip_start, clip_end, output_path, max_chars=20, max_duration=2.0, max_words=None, text_case="original"):
    """
    Generates an SRT file from the transcript for a specific time range.
    Groups words into short lines suitable for vertical video.

    ``max_words`` (optional) overrides character-based grouping with a fixed
    words-per-line cap — set from the brand kit. None = use char heuristic only.
    ``text_case`` applies the brand-kit casing: "original" | "upper" | "lower".
    """

    words = []
    # 1. Extract and flatten words within range
    for segment in transcript.get('segments', []):
        for word_info in segment.get('words', []):
            # Check overlap
            if word_info['end'] > clip_start and word_info['start'] < clip_end:
                words.append(word_info)

    if not words:
        return False

    srt_content = ""
    index = 1

    current_block = []
    block_start = None

    for i, word in enumerate(words):
        # Adjust times relative to clip
        start = max(0, word['start'] - clip_start)
        end = max(0, word['end'] - clip_start)

        # Clip to video duration logic handled by ffmpeg usually, but good to be safe

        if not current_block:
            current_block.append(word)
            block_start = start
        else:
            # Decide whether to close block
            current_text_len = sum(len(w['word']) + 1 for w in current_block)
            duration = end - block_start

            # Honor explicit words-per-line cap (brand kit) if set; falls back
            # to character heuristic otherwise.
            words_exceeded = max_words is not None and len(current_block) >= max_words
            chars_exceeded = max_words is None and (current_text_len + len(word['word']) > max_chars)

            if words_exceeded or chars_exceeded or duration > max_duration:
                # Finalize current block
                # End time of block is start of this word (gap) or end of last word?
                # Usually end of last word.
                block_end = current_block[-1]['end'] - clip_start

                text = " ".join([w['word'] for w in current_block]).strip()
                text = _apply_text_case(text, text_case)
                srt_content += format_srt_block(index, block_start, block_end, text)
                index += 1

                current_block = [word]
                block_start = start
            else:
                current_block.append(word)

    # Final block
    if current_block:
        block_end = current_block[-1]['end'] - clip_start
        text = " ".join([w['word'] for w in current_block]).strip()
        text = _apply_text_case(text, text_case)
        srt_content += format_srt_block(index, block_start, block_end, text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)

    return True


def format_srt_block(index, start, end, text):
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    return f"{index}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"
