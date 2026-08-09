#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" #БЕЗ ТЕСЛЫ
os.environ["PYANNOTE_DISABLE_TORCHCODEC"] = "1"

import sys
import queue
import tempfile
import threading
import argparse
import datetime as dt
import subprocess
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

import torch
import whisper

from pyannote.audio import Model, Inference
from pyannote.core import Segment

import argostranslate.package
import argostranslate.translate

SAMPLE_RATE = 16000
CHANNELS = 1


def fmt_time(sec: float) -> str:
    return str(dt.timedelta(seconds=int(sec)))


def parse_device(device_str: str):
    if device_str.startswith("cuda"):
        if ":" in device_str:
            return "cuda", int(device_str.split(":")[1])
        return "cuda", 0
    return "cpu", 0


class StreamingProcessor:
    def __init__(
        self,
        whisper_dir: str,
        embedding_model_path: str,
        whisper_device: str = "cuda:0",
        pyannote_device: str = "cuda:2",
        similarity_threshold: float = 0.5,
        min_seg_dur: float = 0.5,
        forced_language: str = None,
    ):
        print(f"🎙  Whisper -> {whisper_device}, model={whisper_dir}")
        self.whisper = whisper.load_model(
            whisper_dir,
            device=whisper_device,
        )

        print(f"👥 Embedding-model -> {pyannote_device}, path={embedding_model_path}")
        emb_model = Model.from_pretrained(embedding_model_path)

        if pyannote_device.startswith("cuda"):
            inf_device = torch.device(pyannote_device)
        else:
            inf_device = torch.device("cpu")

        self.inference = Inference(
            emb_model,
            window="whole",
            device=inf_device,
        )

        self.sim_thr = similarity_threshold
        self.min_seg = min_seg_dur
        self.speakers = []
        self.transcript = []
        self.language = forced_language
        self.forced_language = forced_language
        self._print_lock = threading.Lock()

        self._warmup()

    def _warmup(self):
        try:
            silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            wav_write(tmp.name, SAMPLE_RATE, silence)
            self.whisper.transcribe(tmp.name, beam_size=1)
            os.remove(tmp.name)
            print("🔥 Модели прогреты.")
        except Exception as e:
            print(f"⚠️  warmup: {e}")

    @staticmethod
    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _assign(self, emb: np.ndarray) -> str:
        if not self.speakers:
            sid = "SPEAKER_00"
            self.speakers.append({"id": sid, "centroid": emb.copy(), "count": 1})
            return sid

        best_sim, best = -1.0, None
        for s in self.speakers:
            sim = self._cos(emb, s["centroid"])
            if sim > best_sim:
                best_sim, best = sim, s

        if best_sim >= self.sim_thr:
            n = best["count"]
            best["centroid"] = (best["centroid"] * n + emb) / (n + 1)
            best["count"] = n + 1
            return best["id"]

        sid = f"SPEAKER_{len(self.speakers):02d}"
        self.speakers.append({"id": sid, "centroid": emb.copy(), "count": 1})
        return sid

    def process_chunk(self, chunk_wav_path: str, chunk_start_time: float):
        try:
            use_lang = self.forced_language if self.forced_language else self.language

            with self._print_lock:
                print(f"🔍 Чанк: lang={use_lang}, файл={chunk_wav_path}")

            # ВАЖНО: берём language=None если не определён ещё, 
            # чтобы модель сама определила язык, а не переключилась на английский
            if use_lang is None:
                result = self.whisper.transcribe(
                    chunk_wav_path,
                    beam_size=1,
                    language=None,  # auto-detect
                    task="transcribe",
                    condition_on_previous_text=False,
                    initial_prompt="Речь будет только на русском!",
                    #   temperature=0.0,                   # Запрещает модели "фантазировать" (строгий выбор токенов)
                    #   no_speech_threshold=0.6,           # Если вероятность отсутствия речи > 60%, чанк игнорируется
                    #   suppress_blank=True,    
                    #   suppress_tokens=[50358] 
                )
                if self.language is None:
                    self.language = result.get("language")
                    with self._print_lock:
                        print(f"🌐 Язык определён: {self.language}")
            else:
                # Если язык уже определён — используем его
                result = self.whisper.transcribe(
                    chunk_wav_path,
                    beam_size=1,
                    language=use_lang,
                    task="transcribe",
                    condition_on_previous_text=False,
                    #   temperature=0.0,                   # Запрещает модели "фантазировать" (строгий выбор токенов)
                    #   no_speech_threshold=0.6,           # Если вероятность отсутствия речи > 60%, чанк игнорируется
                    #   suppress_blank=True,    
                    #   suppress_tokens=[50358] 
                    initial_prompt="Речь будет только на русском!",  # можно раскомментировать если нужно
                )

            segments = result.get("segments", [])

            for seg in segments:
                text = seg["text"].strip()
                if not text:
                    continue

                start = seg["start"]
                end = seg["end"]
                dur = end - start
                if dur < self.min_seg:
                    speaker = "SPEAKER_?"
                else:
                    try:
                        emb = self.inference.crop(
                            chunk_wav_path, Segment(start, end)
                        )
                        emb_arr = np.asarray(emb).flatten()
                        speaker = self._assign(emb_arr)
                    except Exception:
                        speaker = "SPEAKER_?"

                s_g = chunk_start_time + start
                e_g = chunk_start_time + end
                self.transcript.append({
                    "start": s_g,
                    "end": e_g,
                    "speaker": speaker,
                    "text": text,
                })
                with self._print_lock:
                    print(f"[{fmt_time(s_g)}] {speaker}: {text}")

        except Exception as e:
            print(f"⚠️  chunk error: {e}")
            import traceback
            traceback.print_exc()


def record_and_stream(
    session_dir: Path,
    processor: StreamingProcessor,
    stop_event: threading.Event,
    chunk_seconds: float,
):
    audio_buffer = []
    all_audio = []
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=callback,
    )
    stream.start()

    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    global_offset_samples = 0
    proc_q = queue.Queue()

    def worker():
        while True:
            item = proc_q.get()
            if item is None:
                break
            path, start_t = item
            processor.process_chunk(path, start_t)
            try:
                os.remove(path)
            except OSError:
                pass

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    print("🔴 Запись пошла. Enter — стоп.\n")

    try:
        while not stop_event.is_set():
            try:
                data = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            audio_buffer.append(data)
            all_audio.append(data)

            total = sum(len(a) for a in audio_buffer)
            if total >= chunk_samples:
                chunk_data = np.concatenate(audio_buffer, axis=0)
                audio_buffer.clear()

                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                wav_write(tmp.name, SAMPLE_RATE, chunk_data)

                start_t = global_offset_samples / SAMPLE_RATE
                global_offset_samples += len(chunk_data)
                proc_q.put((tmp.name, start_t))

    finally:
        stream.stop()
        stream.close()

    if audio_buffer:
        chunk_data = np.concatenate(audio_buffer, axis=0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wav_write(tmp.name, SAMPLE_RATE, chunk_data)
        proc_q.put((tmp.name, global_offset_samples / SAMPLE_RATE))

    proc_q.put(None)
    worker_thread.join()

    full = np.concatenate(all_audio, axis=0)
    audio_path = session_dir / "audio.wav"
    wav_write(str(audio_path), SAMPLE_RATE, full)
    return audio_path


def merge_consecutive(transcript: list[dict]) -> list[dict]:
    out = []
    for s in transcript:
        if out and out[-1]["speaker"] == s["speaker"]:
            out[-1]["end"] = s["end"]
            out[-1]["text"] += " " + s["text"]
        else:
            out.append(dict(s))
    return out


def format_transcript(segments: list[dict]) -> str:
    return "\n".join(
        f"[{fmt_time(s['start'])} - {fmt_time(s['end'])}] {s['speaker']}: {s['text']}"
        for s in segments
    )


def install_argos(argos_dir: str):
    p = Path(argos_dir)
    if not p.exists():
        print(f"⚠️  директория argos не найдена: {argos_dir}")
        return
    for pkg in p.glob("*.argosmodel"):
        try:
            argostranslate.package.install_from_path(str(pkg))
            print(f"✅ Установлен пакет: {pkg.name}")
        except Exception as e:
            print(f"⚠️  argos ({pkg.name}): {e}")


def translate_segments(segments: list[dict], src_lang: str) -> list[dict]:
    installed = argostranslate.translate.get_installed_languages()
    src = next((l for l in installed if l.code == src_lang), None)
    tgt = next((l for l in installed if l.code == "en"), None)
    if not src or not tgt:
        raise RuntimeError(f"нет argos-пакета {src_lang}->en")
    tr = src.get_translation(tgt)
    return [{**s, "text": tr.translate(s["text"])} for s in segments]


def full_pipeline(
    audio_path: Path,
    whisper_model_dir: str,
    pyannote_config: str,
    whisper_device: str,
    pyannote_device: str,
):
    print(f"🎙  [full] Whisper -> {whisper_device}")
    whisper_model = whisper.load_model(
        whisper_model_dir,
        device=whisper_device,
    )

    print("📝 Полная транскрипция...")
    result = whisper_model.transcribe(
        str(audio_path),
        beam_size=5,
        language=None,
        task="transcribe",
    )
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result.get("segments", [])]
    lang = result.get("language")
    print(f"🌐 Язык: {lang}")

    print(f"👥 [full] Pyannote diarization -> {pyannote_device}")
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(pyannote_config)
    if pyannote_device.startswith("cuda"):
        pipeline.to(torch.device(pyannote_device))
    diar = pipeline(str(audio_path))

    turns = [(t.start, t.end, spk) for t, _, spk in diar.itertracks(yield_label=True)]

    def assign(a: float, b: float) -> str:
        best, ov_max = "SPEAKER_?", 0.0
        for s, e, spk in turns:
            ov = max(0, min(b, e) - max(a, s))
            if ov > ov_max:
                ov_max, best = ov, spk
        return best

    labeled = [{**s, "speaker": assign(s["start"], s["end"])} for s in segs]
    return lang, merge_consecutive(labeled)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time interview transcriber (offline, multi-GPU)"
    )
    parser.add_argument("--input", "-i", help="Готовый аудиофайл. Иначе — запись с микрофона.")
    parser.add_argument("--name", "-n", help="Имя сессии.")
    parser.add_argument("--out-dir", default="interviews")

    parser.add_argument(
        "--whisper-model",
        default=os.getenv("WHISPER_MODEL_DIR", "/media/biorp/lnd/CODE26/DSMN/msys/models/large-v3/large-v3.pt"),
    )
    parser.add_argument(
        "--pyannote-config",
        default=os.getenv("PYANNOTE_CONFIG", "models/pyannote-diarization-3.1/config.yaml"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "models/pyannote-wespeaker/pytorch_model.bin"),
    )
    parser.add_argument(
        "--argos-dir",
        default=os.getenv("ARGOS_DIR", "models/argos"),
    )

    parser.add_argument("--whisper-device", default="cuda:0")
    parser.add_argument("--pyannote-device", default="cuda:2")

    parser.add_argument("--chunk-sec", type=float, default=5.0)
    parser.add_argument("--sim-threshold", type=float, default=0.5)
    parser.add_argument("--refine", action="store_true")

    parser.add_argument(
        "--language",
        default=None,
        help="Принудительно указать язык (напр. 'ru'). Если не указан — определяется автоматически.",
    )
    args = parser.parse_args()

    name = args.name or dt.datetime.now().strftime("interview_%Y%m%d_%H%M%S")
    session_dir = Path(args.out_dir) / name
    session_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        audio_path = session_dir / "audio.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", args.input,
                "-ar", str(SAMPLE_RATE), "-ac", "1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
        print(f"✅ Аудио: {audio_path}")
        lang, labeled = full_pipeline(
            audio_path, args.whisper_model, args.pyannote_config,
            args.whisper_device, args.pyannote_device,
        )
    else:
        processor = StreamingProcessor(
            whisper_dir=args.whisper_model,
            embedding_model_path=args.embedding_model,
            whisper_device=args.whisper_device,
            pyannote_device=args.pyannote_device,
            similarity_threshold=args.sim_threshold,
            forced_language=args.language,
        )

        stop_event = threading.Event()

        def wait_for_enter():
            try:
                input()
            except EOFError:
                pass
            stop_event.set()

        threading.Thread(target=wait_for_enter, daemon=True).start()
        audio_path = record_and_stream(session_dir, processor, stop_event, args.chunk_sec)
        print("\n⏹  Запись остановлена.")

    if args.refine:
        print("🛠  Финальный проход...")
        lang, labeled = full_pipeline(
            audio_path, args.whisper_model, args.pyannote_config,
            args.whisper_device, args.pyannote_device,
        )
    else:
        lang = processor.language or args.language or "en"
        labeled = merge_consecutive(processor.transcript)

    original = format_transcript(labeled)
    (session_dir / "transcript_original.txt").write_text(original, encoding="utf-8")
    print(f"✅ Оригинал: {session_dir / 'transcript_original.txt'}")

    if lang == "en":
        (session_dir / "transcript_english.txt").write_text(original, encoding="utf-8")
        print(f"✅ Английский (оригинал): {session_dir / 'transcript_english.txt'}")
    else:
        install_argos(args.argos_dir)
        try:
            en_segments = translate_segments(labeled, lang)
            en_text = format_transcript(en_segments)
            (session_dir / "transcript_english.txt").write_text(en_text, encoding="utf-8")
            print(f"✅ Английский: {session_dir / 'transcript_english.txt'}")
        except Exception as e:
            print(f"⚠️  перевод: {e}")

    print(f"\n🎉 Готово: {session_dir.resolve()}")


if __name__ == "__main__":
    main()