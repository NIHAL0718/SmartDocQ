"""Utility functions for audio processing in the frontend."""

import os
import tempfile
from io import BytesIO
import base64
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play


def text_to_speech(text, language="english"):
    """Convert text to speech and return audio data.
    
    Args:
        text (str): Text to convert to speech
        language (str): Language code for speech synthesis
        
    Returns:
        bytes: Audio data
    """
    # Map language to gTTS language code
    language_map = {
        "english": "en",
        "hindi": "hi",
        "telugu": "te",
        "tamil": "ta"
    }
    
    # Default to English if language not supported
    lang_code = language_map.get(language.lower(), "en")
    
    try:
        # Use BytesIO instead of temporary file
        mp3_fp = BytesIO()
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.write_to_fp(mp3_fp)
        
        # Get the audio bytes
        mp3_fp.seek(0)
        audio_bytes = mp3_fp.read()
        
        return audio_bytes
    
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None


def speech_to_text(audio_bytes, language="english"):
    """Convert speech to text.
    
    Args:
        audio_bytes (bytes): Audio data
        language (str): Language code for speech recognition
        
    Returns:
        str: Recognized text
    """
    # Map language to speech recognition language code
    language_map = {
        "english": "en-US",
        "hindi": "hi-IN",
        "telugu": "te-IN",
        "tamil": "ta-IN"
    }
    
    # Default to English if language not supported
    lang_code = language_map.get(language.lower(), "en-US")
    
    # Note: SpeechRecognition library requires a file path for AudioFile
    # We still need to use a temporary file, but with better error handling
    temp_file = None
    try:
        # Create a temporary file with a unique name
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(audio_bytes)
        temp_file.flush()
        temp_file.close()  # Close the file before using it
        
        # Initialize speech recognizer
        recognizer = sr.Recognizer()
        
        # Recognize speech using Google Speech Recognition
        with sr.AudioFile(temp_file.name) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language=lang_code)
                return text
            except sr.UnknownValueError:
                return "Speech could not be understood"
            except sr.RequestError:
                return "Could not request results from speech recognition service"
    except Exception as e:
        return f"Error processing audio: {str(e)}"
    finally:
        # Clean up the temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass  # Ignore errors during cleanup


def get_audio_player_html(audio_bytes):
    """Generate HTML for an audio player with the given audio data.
    
    Args:
        audio_bytes (bytes): Audio data
        
    Returns:
        str: HTML for audio player
    """
    # Encode audio bytes as base64
    b64 = base64.b64encode(audio_bytes).decode()
    
    # Create HTML for audio player
    audio_html = f"""
    <audio controls autoplay>\n
      <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">\n
      Your browser does not support the audio element.\n
    </audio>\n
    """
    
    return audio_html


def convert_audio_format(audio_bytes, input_format="webm", output_format="wav"):
    """Convert audio from one format to another.
    
    Args:
        audio_bytes (bytes): Audio data
        input_format (str): Input audio format
        output_format (str): Output audio format
        
    Returns:
        bytes: Converted audio data
    """
    # Note: pydub requires file paths for some operations
    # We need to use temporary files but with better error handling
    input_file = None
    output_file = None
    
    try:
        # Create temporary files for input and output
        input_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_format}")
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}")
        
        # Write input audio to file
        input_file.write(audio_bytes)
        input_file.flush()
        input_file.close()  # Close the file before using it
        
        # Load audio with pydub
        audio = AudioSegment.from_file(input_file.name, format=input_format)
        
        # Export to output format
        output_file.close()  # Close the file before using it
        audio.export(output_file.name, format=output_format)
        
        # Read the output file
        with open(output_file.name, "rb") as f:
            output_bytes = f.read()
        
        return output_bytes
    
    except Exception as e:
        st.error(f"Error converting audio format: {str(e)}")
        return None
    
    finally:
        # Clean up temporary files
        if input_file and os.path.exists(input_file.name):
            try:
                os.unlink(input_file.name)
            except Exception:
                pass  # Ignore errors during cleanup
        
        if output_file and os.path.exists(output_file.name):
            try:
                os.unlink(output_file.name)
            except Exception:
                pass  # Ignore errors during cleanup