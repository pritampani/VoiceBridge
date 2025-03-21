import os
import wave
import collections
import contextlib
import subprocess
import time
import uuid
import logging
from pydub import AudioSegment
# Set ffmpeg path globally for pydub
ffmpeg_path = "ffmpeg.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
AudioSegment.converter = ffmpeg_path
from inaSpeechSegmenter import Segmenter
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import ffmpeg
import whisper
from gtts import gTTS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import yt_dlp
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="C:\\coding\\python\\INFOSYS\\Speech-to-speech streaming\\FlaskUI\\.env")
api_key = os.getenv("API_KEY")

# Configure paths and settings
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv'}

from gtts.lang import tts_langs
AVAILABLE_LANGUAGES = tts_langs()

SUPPORTED_LANGUAGES = {
    'hi': 'Hindi',
    'bn': 'Bengali',
    'te': 'Telugu',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'ur': 'Urdu',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh-CN': 'Chinese (Simplified)',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ar': 'Arabic',
    'pt': 'Portuguese',
    'it': 'Italian',
    'ko': 'Korean',
}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = "uploads"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----- Step 1: Video Download and Audio Extraction -----
def download_youtube_video(URL, output_folder=None):
    if output_folder is None:
        output_folder = app.config['UPLOAD_FOLDER']
    unique_id = uuid.uuid4().hex
    output_template = os.path.join(output_folder, f"{unique_id}_%(title)s.%(ext)s")
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_template,
        'quiet': False,
        'no_warnings': False
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([URL])
        downloaded_files = [f for f in os.listdir(output_folder) if f.startswith(unique_id)]
        if not downloaded_files:
            raise FileNotFoundError("No files were downloaded.")
        return os.path.join(output_folder, downloaded_files[0])
    except Exception as e:
        app.logger.error("Error downloading video: %s", e)
        return None

def extract_audio(video_path, audio_path):
    ffmpeg_path = "ffmpeg.exe"
    command = [
        ffmpeg_path,
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        audio_path
    ]
    subprocess.run(command, check=True)
    if not os.path.exists(audio_path):
        raise Exception("Audio extraction failed.")

# ----- Step 2: Speech Segmentation using inaSpeechSegmenter -----
def get_speech_segments(audio_path):
    ffmpeg_bin = "ffmpeg.exe"
    if not os.path.exists(ffmpeg_bin):
        raise Exception(f"ffmpeg binary not found at {ffmpeg_bin}")
    ffmpeg_dir = os.path.dirname(ffmpeg_bin)
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    seg = Segmenter()
    segments = seg(audio_path)
    speech_segments = [(start, end) for (label, start, end) in segments if label == 'speech']
    app.logger.info("Detected speech segments (inaSpeechSegmenter): %s", speech_segments)
    return speech_segments

# ----- Step 3: TTS Generation and Audio Mixing -----
def generate_tts_for_segment(text, language, temp_output="temp_tts.mp3"):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(temp_output)
    return AudioSegment.from_file(temp_output)

def overlay_tts_on_audio(audio_path, segments_with_text, language, output_path):
    # Load the original audio
    original_audio = AudioSegment.from_file(audio_path)
    
    # Process audio: replace detected speech segments with TTS output
    modified_audio = original_audio
    for (start, end), text in segments_with_text:
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        app.logger.info("Replacing audio from %.2f to %.2f with TTS", start, end)
        tts_audio = generate_tts_for_segment(text, language)
        # Replace the segment with TTS audio
        modified_audio = modified_audio[:start_ms] + tts_audio + modified_audio[end_ms:]
    
    # Export the modified audio
    modified_audio.export(output_path, format="mp3")
    app.logger.info("Final mixed audio exported to %s", output_path)

# ----- Utility Functions -----
def get_video_duration(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        app.logger.error("Error getting video duration: %s", e)
        return None

def transcribe_audio(audio_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result
    except Exception as e:
        app.logger.error("Error in transcribe_audio (Whisper): %s", e)
        raise

def translate_text(text, target_language):
    if target_language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language code: {target_language}")
    try:
        prompt_template = ("Translate the following text into {language} in a friendly, conversational tone: {sentence}")
        prompt = PromptTemplate(input_variables=["sentence", "language"], template=prompt_template)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
        chain = prompt | llm | StrOutputParser()
        translated_text = chain.invoke({
            "sentence": text,
            "language": SUPPORTED_LANGUAGES[target_language]
        })
        return translated_text
    except Exception as e:
        app.logger.error("Error in translate_text: %s", e)
        raise

def embed_subtitles(video_path, subtitle_file_path, output_path):
    try:
        ffmpeg_path = "ffmpeg.exe"
        normalized_path = os.path.normpath(subtitle_file_path).replace("\\", "/")
        escaped_subtitle_path = normalized_path.replace(":", "\\:")
        subtitles_filter = f"subtitles='{escaped_subtitle_path}':force_style='Fontsize=24,FontName=Arial'"
        command = [
            ffmpeg_path,
            '-i', video_path,
            '-vf', subtitles_filter,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-y',
            output_path
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode != 0:
            app.logger.error("FFmpeg Error in embed_subtitles: %s", process.stderr)
            raise Exception(f"FFmpeg failed: {process.stderr}")
        if not os.path.exists(output_path):
            raise Exception(f"Output file was not created at: {output_path}")
        return True
    except Exception as e:
        app.logger.error("Subtitle embedding error: %s", e)
        return False

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(secs):02}.{milliseconds:03}"

def generate_vtt(transcript, total_duration, num_blocks, offset=0.0):
    sentences = transcript.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    num_blocks = min(num_blocks, len(sentences))
    block_duration = total_duration / num_blocks
    vtt_lines = ["WEBVTT", ""]
    current_time = 0.0
    for i in range(num_blocks):
        start_time = max(0, current_time - offset)
        end_time = start_time + block_duration
        vtt_lines.append(f"{format_time(start_time)} --> {format_time(end_time)}")
        vtt_lines.append(sentences[i])
        vtt_lines.append("")
        current_time += block_duration
    return "\n".join(vtt_lines)

def generate_vtt_from_segments(segments):
    vtt_lines = ["WEBVTT", ""]
    for seg in segments:
        start_time = seg["start"] / 1000.0  
        end_time = seg["end"] / 1000.0
        text = seg["text"].strip()
        vtt_lines.append(f"{format_time(start_time)} --> {format_time(end_time)}")
        vtt_lines.append(text)
        vtt_lines.append("")
    return "\n".join(vtt_lines)

def generate_summary(transcript):
    try:
        prompt_template = """Provide a concise summary of the following transcript in 3-4 sentences, capturing the main points: {transcript}"""
        prompt = PromptTemplate(input_variables=["transcript"], template=prompt_template)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"transcript": transcript})
        return summary
    except Exception as e:
        app.logger.error("Error generating summary: %s", e)
        raise

def text_to_speech(text, output_file, language_code):
    try:
        tts = gTTS(text=text, lang=language_code, slow=False)
        tts.save(output_file)
        if not os.path.exists(output_file):
            raise Exception("Speech file was not created")
    except Exception as e:
        app.logger.error("Error in text_to_speech: %s", e)
        raise

def tune_audio(input_file, output_file, pitch_factor=1.0, tempo_factor=1.0):
    ffmpeg_path = "ffmpeg.exe"
    command = [
        ffmpeg_path,
        '-i', input_file,
        '-filter:a', f"asetrate=44100*{pitch_factor},aresample=44100,atempo={tempo_factor}",
        '-y',
        output_file
    ]
    try:
        subprocess.run(command, check=True)
    except Exception as e:
        app.logger.error("Error in tune_audio: %s", e)
        raise

def replace_audio_in_video(video_with_subtitles, audio_path, final_output_path):
    ffmpeg_path = "ffmpeg.exe"
    command = [
        ffmpeg_path,
        '-i', video_with_subtitles,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-y',
        final_output_path
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        app.logger.error("FFmpeg Error in replace_audio_in_video: %s", process.stderr)
        raise Exception(f"FFmpeg failed with error: {process.stderr}")
    if not os.path.exists(final_output_path):
        raise Exception(f"Output file was not created at: {final_output_path}")
    app.logger.info("Successfully created output video at: %s", final_output_path)
    return True

@app.route("/", methods=["GET", "POST"])
def index():
    project_data = {
        "meta": {
            "ogDescription": "Your default project description",
            "robotsMeta": "index, follow",
            "keywords": "speech-to-speech, AI translation, multilingual",
            "ogTitle": "Speech-to-Speech AI Translator",
            "ogImage": "default_image_url.jpg",
        },
        "seoTitle": "AI Speech Translator - Break Language Barriers",
        "seoDescription": "Translate and dub videos in 12 languages using AI.",
        "name": "LangStream",
    }

    subtitle_file_path = None
    video_save_path = None
    final_video_path = None
    transcript = None
    translated_text = None
    subtitle_text = None
    summary = None
    original_video_url = None
    translated_video_url = None
    subtitle_download_url = None

    if request.method == "POST":
        youtube_url = request.form.get("youtube_url")
        video_language = request.form.get("video_language")
        subtitle_language = request.form.get("subtitle_language")

        if not youtube_url:
            return jsonify({"error": "No URL provided"}), 400
        if video_language not in SUPPORTED_LANGUAGES:
            return jsonify({"error": "Invalid video language selection"}), 400
        if subtitle_language not in SUPPORTED_LANGUAGES:
            return jsonify({"error": "Invalid subtitle language selection"}), 400

        try:
            video_path = download_youtube_video(youtube_url)
            if not video_path:
                return jsonify({"error": "Failed to download video"}), 500

            unique_id = uuid.uuid4().hex
            base_name = os.path.basename(video_path)
            video_save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{base_name}")
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.wav")
            mixed_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_mixed.mp3")
            output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_withsubs.mp4")
            subtitle_file_path = os.path.splitext(video_save_path)[0] + ".vtt"

            os.rename(video_path, video_save_path)
            app.logger.info("Video saved at: %s", video_save_path)

            app.logger.info("Extracting audio...")
            extract_audio(video_save_path, audio_path)

            app.logger.info("Detecting speech segments using inaSpeechSegmenter...")
            segments = get_speech_segments(audio_path)

            app.logger.info("Starting transcription using Whisper...")
            transcript_obj = transcribe_audio(audio_path)
            transcript = transcript_obj.get("text", "") if isinstance(transcript_obj, dict) else transcript_obj

            app.logger.info("Translating video content to %s...", video_language)
            translated_text = translate_text(transcript, video_language) or "Translation failed."
            app.logger.info("Translating subtitles to %s...", subtitle_language)
            subtitle_text = translate_text(transcript, subtitle_language) or "Subtitle translation failed."
            app.logger.info("Generating summary...")
            summary = generate_summary(translated_text) or "Summary generation failed."

            if segments:
                split_text = [s.strip() for s in translated_text.split('.') if s.strip()]
                num_parts = min(len(segments), len(split_text))
                segments_with_text = []
                for i in range(num_parts):
                    segments_with_text.append((segments[i], split_text[i] + "."))
            else:
                total_duration = get_video_duration(video_save_path) or 90
                segments_with_text = [((0, total_duration), translated_text)]

            app.logger.info("Overlaying TTS on audio...")
            overlay_tts_on_audio(audio_path, segments_with_text, video_language, mixed_audio_path)

            if subtitle_text and subtitle_text != "Subtitle translation failed.":
                if isinstance(transcript_obj, dict) and transcript_obj.get("segments"):
                    app.logger.info("Generating VTT content using Whisper segments...")
                    whisper_segments = transcript_obj.get("segments", [])
                    vtt_segments = [{"start": seg["start"] * 1000, "end": seg["end"] * 1000, "text": seg["text"]} for seg in whisper_segments]
                    vtt_content = generate_vtt_from_segments(vtt_segments)
                else:
                    total_duration = get_video_duration(video_save_path) or 90
                    num_blocks = 5
                    app.logger.info("Generating VTT content with even segmentation...")
                    vtt_content = generate_vtt(subtitle_text, total_duration, num_blocks)
                with open(subtitle_file_path, 'w', encoding='utf-8') as f:
                    f.write(vtt_content)
                app.logger.info("Subtitle file written to: %s", subtitle_file_path)
                if os.path.exists(subtitle_file_path):
                    embed_success = embed_subtitles(video_save_path, subtitle_file_path, output_video_path)
                    if not embed_success:
                        shutil.copy2(video_save_path, output_video_path)
                else:
                    shutil.copy2(video_save_path, output_video_path)
            else:
                shutil.copy2(video_save_path, output_video_path)

            final_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_final.mp4")
            app.logger.info("Replacing audio in video with mixed audio...")
            replace_audio_in_video(output_video_path, mixed_audio_path, final_video_path)

            if not os.path.exists(final_video_path):
                return jsonify({"error": "Failed to create translated video"}), 500

            original_video_url = url_for('uploaded_file', filename=os.path.basename(video_save_path), _external=True)
            translated_video_url = url_for('uploaded_file', filename=os.path.basename(final_video_path), _external=True)
            subtitle_download_url = url_for('uploaded_file', filename=os.path.basename(subtitle_file_path), _external=True) if os.path.exists(subtitle_file_path) else None

            response_data = {
                "status": "success",
                "message": "Video processed successfully!",
                "download_link": translated_video_url,
                "original_video_url": original_video_url,
                "translated_video_url": translated_video_url,
                "subtitle_download_url": subtitle_download_url,
                "original_transcript": transcript,
                "translated_transcript": translated_text,
                "subtitle_transcript": subtitle_text,
                "summary": summary
            }
            app.logger.info("Response data: %s", response_data)
            return jsonify(response_data)

        except Exception as e:
            import traceback
            app.logger.error("Error processing video: %s", e, exc_info=True)
            return jsonify({"error": str(e)}), 500

    return render_template("index.html",
                           project=project_data,
                           supported_languages=SUPPORTED_LANGUAGES,
                           original_video_url=original_video_url,
                           translated_video_url=translated_video_url,
                           subtitle_download_url=subtitle_download_url,
                           original_transcript=transcript,
                           translated_transcript=translated_text,
                           subtitle_transcript=subtitle_text,
                           summary=summary)

@app.route("/progress")
def progress():
    def generate():
        last_progress = 0
        while True:
            current_progress = app.config.get('current_progress', 0)
            if current_progress != last_progress:
                yield f"data: {current_progress}\n\n"
                last_progress = current_progress
            if current_progress >= 100:
                break
            time.sleep(0.5)
    return Response(generate(), content_type='text/event-stream')

@app.route('/uploaded_file/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        app.logger.error("Error serving file: %s", e)
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=False)
