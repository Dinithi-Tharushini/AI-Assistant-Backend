import os
from typing import Optional
from openai import OpenAI


class AudioService:
    """Handles Speech-to-Text (STT) and Text-to-Speech (TTS) using OpenAI APIs."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)

    def speech_to_text(self, audio_file_path: str) -> str:
        """Transcribe audio to text using Whisper API.
        Language is fixed to English ('en').
        """
        lang = 'en'
        with open(audio_file_path, "rb") as af:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=af,
                language=lang,
            )
        return transcript.text

    def text_to_speech(self, text: str, voice: str = "alloy", audio_format: str = "mp3") -> bytes:
        """Synthesize speech audio bytes from text using TTS model."""
        # Prefer explicit response_format to match current SDK API
        try:
            resp = self.client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text,
                response_format=audio_format,
            )
            # Some SDK versions expose .content, others require .read()
            return getattr(resp, "content", None) or resp.read()
        except TypeError:
            # Fallback for SDKs that expect 'format' instead of 'response_format'
            resp = self.client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text,
                format=audio_format,
            )
            return getattr(resp, "content", None) or resp.read()


