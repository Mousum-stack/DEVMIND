import os
import time
import asyncio
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import tempfile

import httpx
from deepgram import DeepgramClient, PrerecordedOptions
from elevenlabs.client import ElevenLabs
from elevenlabs import play

load_dotenv()

DEEPGRAM_API_KEY  = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEVMIND_URL       = "http://127.0.0.1:8000/query"
PROJECT           = "chroma"

dg_client = DeepgramClient(DEEPGRAM_API_KEY)
el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


def record_audio(seconds: int = 5) -> str:
    import pyaudio
    import wave

    CHUNK      = 1024
    FORMAT     = pyaudio.paInt16
    CHANNELS   = 1
    RATE       = 16000

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print(f"  Recording for {seconds} seconds — speak now...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wf = wave.open(tmp.name, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    print(f"  Saved to {tmp.name}")
    return tmp.name


def transcribe_audio(audio_path: str) -> str:
    print("  Transcribing with Deepgram...")
    t0 = time.time()

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    options = PrerecordedOptions(
        model="nova-2",
        language="en",
        smart_format=True,
    )

    response = dg_client.listen.rest.v("1").transcribe_file(
        {"buffer": audio_data, "mimetype": "audio/wav"},
        options
    )

    transcript = response.results.channels[0].alternatives[0].transcript
    stt_time = round(time.time() - t0, 2)
    print(f"  Transcript: '{transcript}' ({stt_time}s)")
    return transcript, stt_time


def ask_devmind(question: str) -> tuple:
    print(f"  Asking DevMind: {question}")
    t0 = time.time()

    response = httpx.post(
        DEVMIND_URL,
        json={"question": question, "project": PROJECT},
        timeout=30
    )
    data = response.json()
    rag_time = round(time.time() - t0, 2)

    answer = data.get("answer", "I could not find an answer.")
    sources = data.get("sources", [])
    metrics = data.get("metrics", {})
    print(f"  Answer received ({rag_time}s)")
    return answer, sources, metrics, rag_time


def speak_answer(text: str) -> float:
    print("  Speaking answer with Mac TTS...")
    t0 = time.time()
    clean_text = text[:400].replace("'", "").replace('"', "")
    import subprocess
    subprocess.run(["say", "-v", "Samantha", clean_text])
    tts_time = round(time.time() - t0, 2)
    print(f"  Spoken ({tts_time}s)")
    return tts_time


def run_voice_pipeline(record_seconds: int = 5):
    print("\n" + "=" * 55)
    print("  DevMind — Voice Pipeline")
    print("=" * 55)

    total_start = time.time()

    print("\n[1/4] Recording your question...")
    audio_path = record_audio(seconds=record_seconds)

    print("\n[2/4] Transcribing speech to text...")
    question, stt_time = transcribe_audio(audio_path)

    if not question.strip():
        print("  No speech detected. Try again.")
        return

    print("\n[3/4] Getting answer from DevMind...")
    answer, sources, metrics, rag_time = ask_devmind(question)

    print("\n[4/4] Speaking the answer...")
    tts_time = speak_answer(answer)

    total_time = round(time.time() - total_start, 2)

    print("\n" + "=" * 55)
    print("  LATENCY BREAKDOWN")
    print("=" * 55)
    print(f"  STT (Deepgram):     {stt_time}s")
    print(f"  RAG pipeline:       {rag_time}s")
    print(f"    - Retrieval:      {metrics.get('retrieval_time', '?')}s")
    print(f"    - Generation:     {metrics.get('generation_time', '?')}s")
    print(f"  TTS (ElevenLabs):   {tts_time}s")
    print(f"  Total end-to-end:   {total_time}s")
    print(f"\n  Sources: {', '.join([s.split('/')[-1] for s in sources[:3]])}")

    if total_time < 10:
        print(f"\n  Good latency for CPU-only setup!")
    print("=" * 55)

    Path(audio_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import sys
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_voice_pipeline(record_seconds=seconds)