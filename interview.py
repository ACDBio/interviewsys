#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time interview transcriber (offline, multi-GPU)
ДОРАБОТКА: у КАЖДОЙ фразы/реплики есть точное время, чтобы мгновенно найти её в audio.wav
- В процессе записи (StreamingProcessor) и в постобработке (full_pipeline)
- Время в терминале: ОДИН формат HH:MM:SS.mmm (было два), остальное — в файлах
- Сохранение во все форматы: .txt, .srt, .vtt, .json, .csv, Audacity labels, .html (тихо, без спама в терминал)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # БЕЗ ТЕСЛЫ
os.environ["PYANNOTE_DISABLE_TORCHCODEC"] = "1"
import warnings

# Скрывает конкретное предупреждение от pyannote
warnings.filterwarnings(
    "ignore", 
    message=".*TensorFloat-32.*", 
    category=UserWarning # или попробуйте Warning, если не сработает
)


import sys
import queue
import tempfile
import threading
import argparse
import datetime as dt
import subprocess
import json
import csv
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

# ============================================================
# ===  ФУНКЦИИ ВРЕМЕНИ  ======================================
# ============================================================

def sec_to_hms(sec: float, with_ms: bool = True) -> str:
    """00:00:00.000 — удобно для поиска в плеере / ffmpeg -ss"""
    if sec < 0:
        sec = 0
    total_seconds = int(sec)
    ms = int(round((sec - total_seconds) * 1000))
    if ms == 1000:
        total_seconds += 1
        ms = 0
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    if with_ms:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    else:
        return f"{h:02d}:{m:02d}:{s:02d}"

def sec_to_srt(sec: float) -> str:
    """00:00:00,000 — формат SRT"""
    return sec_to_hms(sec, with_ms=True).replace(".", ",")

def sec_to_vtt(sec: float) -> str:
    """00:00:00.000 — формат VTT"""
    return sec_to_hms(sec, with_ms=True)

def fmt_time(sec: float) -> str:
    return sec_to_hms(sec, with_ms=True)

def parse_device(device_str: str):
    if device_str.startswith("cuda"):
        if ":" in device_str:
            return "cuda", int(device_str.split(":")[1])
        return "cuda", 0
    return "cpu", 0


# ============================================================
# ===  ЭКСПОРТ ТРАНСКРИПТА ВО ВСЕ ФОРМАТЫ  ====================
# ============================================================

def enrich_segments(segments: list[dict]) -> list[dict]:
    out = []
    for s in segments:
        start = float(s["start"])
        end = float(s["end"])
        out.append({
            "start": start,
            "end": end,
            "duration": round(end - start, 3),
            "start_hms": sec_to_hms(start),
            "end_hms": sec_to_hms(end),
            "start_srt": sec_to_srt(start),
            "end_srt": sec_to_srt(end),
            "speaker": s["speaker"],
            "text": s["text"].strip(),
        })
    return out

def format_transcript(segments: list[dict]) -> str:
    enriched = enrich_segments(segments)
    header = [
        f"# Транскрипт — {len(enriched)} реплик",
        f"# Аудио: audio.wav (начало = 00:00:00.000)",
        f"# Как найти в аудио: ffmpeg -ss {enriched[0]['start_hms'] if enriched else '00:00:00.000'} -i audio.wav -t 10 preview.wav",
        f"# Или кликни по времени в transcript.html",
        f"# Формат строки: [START_HMS - END_HMS] SPEAKER: TEXT",
        "",
    ]
    lines = []
    for s in enriched:
        lines.append(f"[{s['start_hms']} - {s['end_hms']}] {s['speaker']}: {s['text']}")
    return "\n".join(header + lines)

def save_srt(segments: list[dict], path: Path):
    enriched = enrich_segments(segments)
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(enriched, 1):
            f.write(f"{i}\n")
            f.write(f"{s['start_srt']} --> {s['end_srt']}\n")
            f.write(f"{s['speaker']}: {s['text']}\n\n")

def save_vtt(segments: list[dict], path: Path):
    enriched = enrich_segments(segments)
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in enriched:
            f.write(f"{s['start_hms']} --> {s['end_hms']}\n")
            f.write(f"<v {s['speaker']}>{s['text']}\n\n")

def save_json(segments: list[dict], path: Path):
    enriched = enrich_segments(segments)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

def save_csv(segments: list[dict], path: Path):
    enriched = enrich_segments(segments)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","start","end","duration","start_hms","end_hms","speaker","text"])
        w.writeheader()
        for i, s in enumerate(enriched):
            w.writerow({
                "id": i,
                "start": f"{s['start']:.3f}",
                "end": f"{s['end']:.3f}",
                "duration": f"{s['duration']:.3f}",
                "start_hms": s['start_hms'],
                "end_hms": s['end_hms'],
                "speaker": s['speaker'],
                "text": s['text']
            })

def save_audacity_labels(segments: list[dict], path: Path):
    enriched = enrich_segments(segments)
    with open(path, "w", encoding="utf-8") as f:
        for s in enriched:
            label = f"{s['speaker']}: {s['text']}".replace("\t"," ").replace("\n"," ")
            f.write(f"{s['start']:.3f}\t{s['end']:.3f}\t{label}\n")

def save_html_player(segments: list[dict], audio_filename: str, path: Path, title: str = "Транскрипт"):
    enriched = enrich_segments(segments)
    rows = ""
    for s in enriched:
        safe_text = s['text'].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        rows += f"""
        <tr class="row" data-start="{s['start']:.3f}" data-end="{s['end']:.3f}">
          <td><a href="#" class="time" onclick="seek({s['start']:.3f}); return false;">{s['start_hms']}</a><br><span class="time2">{s['end_hms']}</span></td>
          <td><span class="spk">{s['speaker']}</span></td>
          <td>{safe_text}</td>
          <td><button onclick="seek({s['start']:.3f})">▶️</button></td>
        </tr>
        """
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
 body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; max-width: 950px; margin: 24px auto; padding: 0 16px; background:#0f1115; color:#e6e6e6; }}
 h1 {{ font-size: 22px; }}
 audio {{ width: 100%; margin: 16px 0; }}
 table {{ width:100%; border-collapse: collapse; font-size: 14px; }}
 th {{ text-align:left; border-bottom: 1px solid #333; padding: 8px; color:#aaa; position: sticky; top:0; background:#0f1115; }}
 td {{ padding: 8px; border-bottom: 1px solid #222; vertical-align: top; }}
 tr:hover {{ background:#1a1d24; }}
 .time {{ font-family: ui-monospace, monospace; color:#7eb8ff; text-decoration:none; font-weight:600; }}
 .time2 {{ font-family: ui-monospace, monospace; color:#8a9bb5; font-size:12px; }}
 .spk {{ background:#222a3a; color:#9ec1ff; padding:2px 6px; border-radius:6px; font-family: ui-monospace, monospace; font-size:12px; white-space:nowrap; }}
 button {{ background:#2a6bff; color:white; border:0; border-radius:6px; padding:4px 8px; cursor:pointer; }}
 .hint {{ color:#9aa6bf; font-size:13px; background:#1a1d24; padding:10px 12px; border-radius:8px; }}
</style>
</head>
<body>
<h1>{title} — {len(enriched)} реплик</h1>
<div class="hint">Нажми на время или ▶️ чтобы прыгнуть в аудио. Время <b>HH:MM:SS.mmm</b> от начала <code>{audio_filename}</code></div>
<audio id="player" controls preload="metadata" src="{audio_filename}"></audio>
<table>
<thead><tr><th style="width:160px">Время</th><th>Спикер</th><th>Текст</th><th></th></tr></thead>
<tbody>{rows}</tbody>
</table>
<script>
const player = document.getElementById('player');
function seek(t){{ player.currentTime = t; player.play(); }}
player.addEventListener('timeupdate', () => {{
  const t = player.currentTime;
  document.querySelectorAll('tr.row').forEach(r=>{{
    const s=parseFloat(r.dataset.start), e=parseFloat(r.dataset.end);
    r.style.background = (t>=s && t<e) ? '#1e2a44' : '';
  }});
}});
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")

def save_all_formats(segments: list[dict], session_dir: Path, basename: str, audio_filename: str = "audio.wav"):
    """Сохраняет во все форматы — тихо, без вывода в терминал (как ты просила)"""
    if not segments:
        print(f"⚠️  Нет сегментов для {basename}, пропускаем экспорт")
        return
    (session_dir / f"{basename}.txt").write_text(format_transcript(segments), encoding="utf-8")
    save_srt(segments, session_dir / f"{basename}.srt")
    save_vtt(segments, session_dir / f"{basename}.vtt")
    save_json(segments, session_dir / f"{basename}.json")
    save_csv(segments, session_dir / f"{basename}.csv")
    save_audacity_labels(segments, session_dir / f"{basename}_audacity.txt")
    save_html_player(segments, audio_filename, session_dir / f"{basename}.html", title=basename)
    # намеренно без print — чтобы не спамить форматами в терминал


# ============================================================
# ===  STREAMING  =============================================
# ============================================================

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
        session_dir: Path | None = None,
    ):
        print(f"🎙  Whisper -> {whisper_device}, model={whisper_dir}")
        self.whisper = whisper.load_model(whisper_dir, device=whisper_device)

        print(f"👥 Embedding-model -> {pyannote_device}, path={embedding_model_path}")
        emb_model = Model.from_pretrained(embedding_model_path)
        if pyannote_device.startswith("cuda"):
            inf_device = torch.device(pyannote_device)
        else:
            inf_device = torch.device("cpu")
        self.inference = Inference(emb_model, window="whole", device=inf_device)

        self.sim_thr = similarity_threshold
        self.min_seg = min_seg_dur
        self.speakers = []
        self.transcript = []
        self.language = forced_language
        self.forced_language = forced_language
        self._print_lock = threading.Lock()
        self.session_dir = Path(session_dir) if session_dir else None
        self.session_start_wall = dt.datetime.now()
        self._live_counter = 0
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

    def _save_live(self):
        if not self.session_dir or not self.transcript:
            return
        try:
            merged = merge_consecutive(self.transcript)
            save_all_formats(merged, self.session_dir, "transcript_live", audio_filename="audio.wav")
        except Exception as e:
            print(f"⚠️  live save: {e}")

    def process_chunk(self, chunk_wav_path: str, chunk_start_time: float):
        try:
            use_lang = self.forced_language if self.forced_language else self.language

            # ОСТАВЛЯЕМ информирование как было, но без дубля формата времени
            with self._print_lock:
                print(f"🔍 Чанк: lang={use_lang}, файл={chunk_wav_path} | оффсет {sec_to_hms(chunk_start_time)}")

            if use_lang is None:
                result = self.whisper.transcribe(
                    chunk_wav_path,
                    beam_size=1,
                    language=None,
                    task="transcribe",
                    condition_on_previous_text=False,
                #    initial_prompt="Речь будет только на русском!",
                )
                if self.language is None:
                    self.language = result.get("language")
                    with self._print_lock:
                        print(f"🌐 Язык определён: {self.language}")
            else:
                result = self.whisper.transcribe(
                    chunk_wav_path,
                    beam_size=1,
                    language=use_lang,
                    task="transcribe",
                    condition_on_previous_text=False,
                  #  initial_prompt="Речь будет только на русском!",
                )

            segments = result.get("segments", [])
            new_entries = 0
            for seg in segments:
                text = seg["text"].strip()
                if not text:
                    continue
                start = float(seg["start"])
                end = float(seg["end"])
                dur = end - start
                if dur < self.min_seg:
                    speaker = "SPEAKER_?"
                else:
                    try:
                        emb = self.inference.crop(chunk_wav_path, Segment(start, end))
                        emb_arr = np.asarray(emb).flatten()
                        speaker = self._assign(emb_arr)
                    except Exception:
                        speaker = "SPEAKER_?"

                s_g = chunk_start_time + start
                e_g = chunk_start_time + end

                entry = {
                    "start": s_g,
                    "end": e_g,
                    "speaker": speaker,
                    "text": text,
                    "_chunk_start": chunk_start_time,
                    "_local_start": start,
                    "_local_end": end,
                }
                self.transcript.append(entry)
                new_entries += 1
                with self._print_lock:
                    # ИЗМЕНЕНО: только один формат времени
                    print(f"[{sec_to_hms(s_g)} - {sec_to_hms(e_g)}] {speaker}: {text}")

            if new_entries > 0:
                self._save_live()
                self._live_counter += 1

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
        if status:
            print(f"⚠️  audio status: {status}", file=sys.stderr)
        audio_q.put(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", callback=callback)
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

    # ВОССТАНОВЛЕНО как было
    print("🔴 Запись пошла. Enter — стоп.\n")
    print(f"   Сессия: {session_dir.resolve()}")
    print(f"   Live-транскрипт будет обновляться в: {session_dir / 'transcript_live.txt'}")
    print(f"   Формат времени: [HH:MM:SS.mmm - HH:MM:SS.mmm] — 0.000 = старт audio.wav\n")

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
        if not all_audio:
            print("⚠️  Нет записанного аудио!")
            silence = np.zeros(int(SAMPLE_RATE*0.5), dtype=np.int16)
            all_audio = [silence[:, None] if silence.ndim==1 else silence]
        full = np.concatenate(all_audio, axis=0)
        audio_path = session_dir / "audio.wav"
        wav_write(str(audio_path), SAMPLE_RATE, full)
        print(f"\n💾 Аудио сохранено: {audio_path} ({len(full)/SAMPLE_RATE:.2f}с)")
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
    # ВОССТАНОВЛЕНО информирование
    print(f"🎙  [full] Whisper -> {whisper_device}")
    whisper_model = whisper.load_model(whisper_model_dir, device=whisper_device)
    print("📝 Полная транскрипция...")
    result = whisper_model.transcribe(str(audio_path), beam_size=5, language=None, task="transcribe")
    segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()} for s in result.get("segments", [])]
    lang = result.get("language")
    print(f"🌐 Язык: {lang} — {len(segs)} сегментов до диаризации")
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
    merged = merge_consecutive(labeled)
    # ИЗМЕНЕНО: только один формат времени
    for s in merged:
        print(f"[{sec_to_hms(s['start'])} - {sec_to_hms(s['end'])}] {s['speaker']}: {s['text']}")
    return lang, merged


def main():
    parser = argparse.ArgumentParser(description="Real-time interview transcriber (offline, multi-GPU) — с таймкодами для поиска в аудио")
    parser.add_argument("--input", "-i", help="Готовый аудиофайл. Иначе — запись с микрофона.")
    parser.add_argument("--name", "-n", help="Имя сессии.")
    parser.add_argument("--out-dir", default="interviews")
    parser.add_argument("--whisper-model", default=os.getenv("WHISPER_MODEL_DIR", "/media/biorp/lnd/CODE26/DSMN/msys/models/large-v3/large-v3.pt"))
    parser.add_argument("--pyannote-config", default=os.getenv("PYANNOTE_CONFIG", "models/pyannote-diarization-3.1/config.yaml"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "models/pyannote-wespeaker/pytorch_model.bin"))
    parser.add_argument("--argos-dir", default=os.getenv("ARGOS_DIR", "models/argos"))
    parser.add_argument("--whisper-device", default="cuda:0")
    parser.add_argument("--pyannote-device", default="cuda:2")
    parser.add_argument("--chunk-sec", type=float, default=15.0)
    parser.add_argument("--sim-threshold", type=float, default=0.5)
    parser.add_argument("--refine", action="store_true", help="После записи сделать финальный точный проход (full_pipeline)")
    parser.add_argument("--language", default=None, help="Принудительно указать язык (напр. 'ru')")
    args = parser.parse_args()

    name = args.name or dt.datetime.now().strftime("interview_%Y%m%d_%H%M%S")
    session_dir = Path(args.out_dir) / name
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Сессия: {session_dir.resolve()}")

    if args.input:
        audio_path = session_dir / "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-i", args.input, "-ar", str(SAMPLE_RATE), "-ac", "1", str(audio_path)], check=True, capture_output=True)
        print(f"✅ Аудио: {audio_path}")
        lang, labeled = full_pipeline(audio_path, args.whisper_model, args.pyannote_config, args.whisper_device, args.pyannote_device)
    else:
        processor = StreamingProcessor(
            whisper_dir=args.whisper_model,
            embedding_model_path=args.embedding_model,
            whisper_device=args.whisper_device,
            pyannote_device=args.pyannote_device,
            similarity_threshold=args.sim_threshold,
            forced_language=args.language,
            session_dir=session_dir,
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
            print("🛠  Финальный проход (более точный, медленнее)...")
            lang, labeled = full_pipeline(audio_path, args.whisper_model, args.pyannote_config, args.whisper_device, args.pyannote_device)
        else:
            lang = processor.language or args.language or "en"
            labeled = merge_consecutive(processor.transcript)
            print(f"📝 Streaming-сегментов: {len(labeled)}")

    # Сохранение — тихо
    save_all_formats(labeled, session_dir, "transcript_original", audio_filename="audio.wav")

    if lang == "en":
        save_all_formats(labeled, session_dir, "transcript_english", audio_filename="audio.wav")
    else:
        install_argos(args.argos_dir)
        try:
            en_segments = translate_segments(labeled, lang)
            save_all_formats(en_segments, session_dir, "transcript_english", audio_filename="audio.wav")
        except Exception as e:
            print(f"⚠️  перевод: {e}")

    # Оставлено краткое информирование, без спама форматами
    print(f"\n🎉 Готово: {session_dir.resolve()}")


if __name__ == "__main__":
    main()
