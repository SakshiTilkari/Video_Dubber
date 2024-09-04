import streamlit as st
import cv2
from tempfile import NamedTemporaryFile
import os
import uuid
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
from google.oauth2 import service_account
import google.auth
from google.auth.transport.requests import Request
from google.cloud import texttospeech_v1 as texttospeech
import whisper
import spacy
from spacy_syllables import SpacySyllables
from tqdm import tqdm
import re
from en_hi_trans import translate

AudioSegment.converter = "C:\Program Files\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"

# Define the spacy models and abbreviations
spacy_models = {
    "english": "en_core_web_sm",
    "german": "de_core_news_sm",
    "french": "fr_core_news_sm",
    "italian": "it_core_news_sm",
    "catalan": "ca_core_news_sm",
    "chinese": "zh_core_web_sm",
    "croatian": "hr_core_news_sm",
    "danish": "da_core_news_sm",
    "dutch": "nl_core_news_sm",
    "finnish": "fi_core_news_sm",
    "greek": "el_core_news_sm",
    "japanese": "ja_core_news_sm",
    "korean": "ko_core_news_sm",
    "lithuanian": "lt_core_news_sm",
    "macedonian": "mk_core_news_sm",
    "polish": "pl_core_news_sm",
    "portuguese": "pt_core_news_sm",
    "romanian": "ro_core_news_sm",
    "russian": "ru_core_news_sm",
    "spanish": "es_core_news_sm",
    "swedish": "sv_core_news_sm",
    "ukrainian": "uk_core_news_sm"
}

ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "No.": "Number",
    "Dr.": "Doctor",
    "Ms.": "Miss",
    "Ave.": "Avenue",
    "Blvd.": "Boulevard",
    "Ln.": "Lane",
    "Rd.": "Road",
    "a.m.": "before noon",
    "p.m.": "after noon",
    "ft.": "feet",
    "hr.": "hour",
    "min.": "minute",
    "sq.": "square",
    "St.": "street",
    "Asst.": "assistant",
    "Corp.": "corporation"
}

ISWORD = re.compile(r'.*\w.*')

# Define the functions for extracting audio, transcribing, translating, creating audio from text, and merging audio files
def extract_audio_from_video(video_file):
    try:
        video = VideoFileClip(video_file)
        audio = video.audio
        audio_file = os.path.splitext(video_file)[0] + ".wav"
        audio.write_audiofile(audio_file)
        print("audio_file", audio_file)
        return audio_file
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None


def transcribe_audio(audio_file, source_language):
    try:
        model = whisper.load_model("small")
        trans = model.transcribe(audio_file, language=source_language, verbose=False, word_timestamps=True)
        print("reached trans")
        return trans
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def translate_text(texts):
    try:
        # Initialize an empty list to store translated texts
        translated_texts = []
        
        # Loop through each text in the input list
        for text in texts:
            # Translate the text using your translation function
            translated_text = translate(text)  # Assuming translate function is defined elsewhere
            
            # Append the translated text to the list
            translated_texts.append(translated_text)
        
        return translated_texts
    except Exception as e:
        print(f"Error translating texts: {e}")
        return None


def create_audio_from_text(text, target_language, target_voice, api_key):
    print("reached create audio")
    audio_file = "translated_" + str(uuid.uuid4()) + ".wav"
    try:
        client = texttospeech.TextToSpeechClient(client_options={"api_key": api_key})
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=target_language,
            name="hi-IN-Standard-C"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=1.1
        )
        response = client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        with open(audio_file, "wb") as out:
            print("reach weite audio")
            out.write(response.audio_content)
            print("reach weite audio")
        return audio_file
    except Exception as e:
        if os.path.isfile(audio_file):
            os.remove(audio_file)
        raise Exception(f"Error creating audio from text: {e}")
   

def merge_audio_files(transcription, source_language, target_language, target_voice, audio_file):
    temp_files = []
    print("reach merge audio")
    try:
        print("reach merge audio")
        ducked_audio = AudioSegment.from_wav(audio_file)
        if spacy_models[source_language] not in spacy.util.get_installed_models():
            spacy.cli.download(spacy_models[source_language])
        nlp = spacy.load(spacy_models[source_language])
        nlp.add_pipe("syllables", after="tagger")
        merged_audio = AudioSegment.silent(duration=0)
        sentences = []
        sentence_starts = []
        sentence_ends = []
        sentence = ""
        sent_start = 0
        for segment in tqdm(transcription["segments"]):
            if segment["text"].isupper():
                continue
            for i, word in enumerate(segment["words"]):
                if not ISWORD.search(word["word"]):
                    continue
                word["word"] = ABBREVIATIONS.get(word["word"].strip(), word["word"])
                if word["word"].startswith("-"):
                    sentence = sentence[:-1] + word["word"] + " "
                else:
                    sentence += word["word"] + " "
                word_syllables = sum(token._.syllables_count for token in nlp(word["word"]) if token._.syllables_count)
                segment_syllables = sum(token._.syllables_count for token in nlp(segment["text"]) if token._.syllables_count)
                if i == 0 or sent_start == 0:
                    word_speed = word_syllables / (word["end"] - word["start"])
                    if word_speed < 3:
                        sent_start = word["end"] - word_syllables / 3
                    else:
                        sent_start = word["start"]
                if i == len(segment["words"]) - 1:
                    word_speed = word_syllables / (word["end"] - word["start"])
                    segment_speed = segment_syllables / (segment["end"] - segment["start"])
                    if word_speed < 1.0 or segment_speed < 2.0:
                        word["word"] += "."

                if word["word"].endswith("."):
                    sentences.append(sentence)
                    sentence_starts.append(sent_start)
                    sentence_ends.append(word["end"])
                    sent_start = 0
                    sentence = ""
        translated_texts = []
        print("translate")
        for i in tqdm(range(0, len(sentences), 128)):
            chunk = sentences[i:i + 128]
            translated_chunk = translate_text(chunk)
            if translated_chunk is None:
                raise Exception("Translation failed")
            translated_texts.extend(translated_chunk)
        prev_end_time = 0
        print("translated_text")
        for i, translated_text in enumerate(tqdm(translated_texts)):
            translated_audio_file = create_audio_from_text(translated_text, target_language, target_voice, api_key)
            if translated_audio_file is None:
                raise Exception("Audio creation failed")
            temp_files.append(translated_audio_file)
            translated_audio = AudioSegment.from_wav(translated_audio_file)
            start_time = int(sentence_starts[i] * 1000)
            end_time = start_time + len(translated_audio)
            next_start_time = int(sentence_starts[i+1] * 1000) if i < len(translated_texts) - 1 else len(ducked_audio)
            ducked_segment = ducked_audio[start_time:end_time].apply_gain(-10)
            fade_out_duration = min(500, max(1, start_time - prev_end_time))
            fade_in_duration = min(500, max(1, next_start_time - end_time))
            prev_end_time = end_time
            if start_time == 0:
                ducked_audio = ducked_segment + ducked_audio[end_time:].fade_in(fade_in_duration)
            elif end_time == len(ducked_audio):
                ducked_audio = ducked_audio[:start_time].fade_out(fade_out_duration) + ducked_segment
            else:
                ducked_audio = ducked_audio[:start_time].fade_out(fade_out_duration) + ducked_segment + ducked_audio[end_time:].fade_in(fade_in_duration)
            ducked_audio = ducked_audio.overlay(translated_audio, position=start_time)
            original_duration = int(sentence_ends[i] * 1000)
            new_duration = len(translated_audio) + len(merged_audio)
            padding_duration = max(0, original_duration - new_duration)
            print("padding_duration")
            padding = AudioSegment.silent(duration=padding_duration)
            merged_audio += padding + translated_audio
        return merged_audio, ducked_audio
    except Exception as e:
        print(f"Error merging audio files: {e}")
        return None
    finally:
        for file in temp_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing temporary file {file}: {e}")

def save_audio_to_file(audio, filename):
    try:
        print(f"Audio track ")
        audio.export(filename, format="wav")
        print(f"Audio track with translation only saved to {filename}")
    except Exception as e:
        print(f"Error saving audio to file: {e}")

import tempfile

def replace_audio_in_video(video_file, new_audio):
    try:
        video = VideoFileClip(video_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            new_audio.export(temp_audio_file.name, format="wav")
        new_audio.export("dubbed.wav", format="wav")
        try:
            new_audio_clip = AudioFileClip(temp_audio_file.name)
        except Exception as e:
            print(f"Error loading new audio into an AudioFileClip: {e}")
            return
        if new_audio_clip.duration < video.duration:
            print("Warning: The new audio is shorter than the video. The remaining video will have no sound.")
        elif new_audio_clip.duration > video.duration:
            print("Warning: The new audio is longer than the video. The extra audio will be cut off.")
            new_audio_clip = new_audio_clip.subclip(0, video.duration)
        video = video.set_audio(new_audio_clip)
        output_filename = os.path.splitext(video_file)[0] + "_translated.mp4"
        try:
            video.write_videofile(output_filename, audio_codec='aac')
        except Exception as e:
            print(f"Error writing the new video file: {e}")
            return
        print(f"Translated video saved as {output_filename}")
    except Exception as e:
        print(f"Error replacing audio in video: {e}")
    finally:
        if os.path.isfile(temp_audio_file.name):
            os.remove(temp_audio_file.name)

# Streamlit UI
st.title("Video Processing and Translation App")

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

# Process the video and display it
if uploaded_file is not None:
    st.write("Original Video:")
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Use st.video to display the video
    st.video(temp_file_path)

    st.write("Processing...")
    
    api_key = "Your Google Text-To-Speech API Key"
    target_voice = "hi-IN-Standard-C"
    source_language = "english"

    # Extract, transcribe, translate, and merge audio files
    audio_file = extract_audio_from_video(temp_file_path)
    if audio_file is not None:
        transcription = transcribe_audio(audio_file, source_language)
        if transcription is not None:
            merged_audio, ducked_audio = merge_audio_files(transcription, source_language, target_voice[:5], target_voice, audio_file)
            if merged_audio is not None:
                replace_audio_in_video(temp_file_path, ducked_audio)
                output_filename = os.path.splitext(temp_file_path)[0] + "_translated.mp4"
                st.write("Processed Video:")
                st.video(output_filename)
            else:
                st.write("Error merging audio files.")
        else:
            st.write("Error transcribing audio.")
    else:
        st.write("Error extracting audio from video.")
