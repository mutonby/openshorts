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


def generate_srt_from_video(video_path, output_path, max_chars=20, max_duration=2.0):
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

    return generate_srt(transcript, 0, duration, output_path, max_chars, max_duration)


def generate_srt(transcript, clip_start, clip_end, output_path, max_chars=20, max_duration=2.0):
    """
    Generates an SRT file from the transcript for a specific time range.
    Groups words into short lines suitable for vertical video.
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
            
            if current_text_len + len(word['word']) > max_chars or duration > max_duration:
                # Finalize current block
                # End time of block is start of this word (gap) or end of last word?
                # Usually end of last word.
                block_end = current_block[-1]['end'] - clip_start
                
                text = " ".join([w['word'] for w in current_block]).strip()
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

def burn_subtitles(video_path, srt_path, output_path, alignment=2, fontsize=16):
    """
    Burns subtitles into the video using FFmpeg.
    Alignment: 2 (Bottom), 5 (Middle), 8 (Top)
    """
    # Convert styling options to ASS format
    # FontSize is roughly pixels for 720p? It scales.
    # For 1080x1920, fontsize 16 is tiny. 
    # Let's assume standard vertical resolution (1080w). 
    # Try a larger default or let it be scaled.
    # We will accept a 'scale' factor or just big font.
    # Default 16 in ASS is small. 
    # Let's use a reasonable default if user says "small/medium/large".
    
    # Mapping alignment:
    # Top: 6 or 10? ASS: 8 = Top Center. 2 = Bottom Center. 5 = Middle Center.
    
    # Position mapping (Numpad)
    ass_alignment = 2 # Default Bottom Center
    align_lower = str(alignment).lower()
    if align_lower == 'top': 
        ass_alignment = 6 # 6 is Top-Center in libass (legacy mode usually 6, standard is 8. Let's force 2 (bottom) 10 (center) ? No. 
        # Actually libass follows SSA/ASS V4+.
        # 1=Left, 2=Center, 3=Right (Subtitles Filter treats these as "Bottom")
        # 5=Top-Left, 6=Top-Center, 7=Top-Right ??
        # 9=Mid-Left, 10=Mid-Center, 11=Mid-Right ??
        # Standard: 2=Bottom, 6=Top, 10=Middle
        ass_alignment = 6
    elif align_lower == 'middle': 
        ass_alignment = 10
    elif align_lower == 'bottom': 
        ass_alignment = 2

    # Font size logic
    # Scale: Libass uses 384x288 virtual resolution unless PlayResX/Y set.
    # The frontend sends a value like 24 (pixels).
    # In 288p land, 24 is HUGE (approx 1/12th of screen height).
    # We want it to be smaller, around 10-12 units.
    # Factor: 0.5 is safe.
    final_fontsize = int(fontsize * 0.5) 
    if final_fontsize < 8: final_fontsize = 8

    # Path handling for filter string
    try:
        # Use absolute path but replace special chars for FFmpeg filter syntax
        # : -> \: and \ -> / (forward slash is safer on windows too usually in ffmpeg filters if escaped)
        # But for standard os paths on linux/mac: /path/to/file.srt
        # FFmpeg expects: subtitles='/path/to/file.srt'
        # If there are colons (e.g. C:), they need escaping: C\:
        safe_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')
    except:
        safe_srt_path = srt_path

    # Style String
    # BorderStyle=3 (Opaque Box)
    # OutlineColour is Box Background. Alpha 60 (approx 40% opacity) -> &H60000000
    # Fontname: 'Verdana' or 'Arial' are safe. 'Verdana' is slightly more "modern/web".
    # Bold=1
    style_string = f"Alignment={ass_alignment},Fontname=Verdana,Fontsize={final_fontsize},PrimaryColour=&H00FFFFFF,OutlineColour=&H60000000,BackColour=&H00000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=25,Bold=1"
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f"subtitles='{safe_srt_path}':force_style='{style_string}'",
        '-c:a', 'copy',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        output_path
    ]
    
    print(f"🎬 Burning subtitles: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f"❌ FFmpeg Subtitle Error: {result.stderr.decode()}")
        raise Exception(f"FFmpeg failed: {result.stderr.decode()}")

    return True

