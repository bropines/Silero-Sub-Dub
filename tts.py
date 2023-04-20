import os
import re
import torch
import srt
import numpy as np
from pydub import AudioSegment

device = torch.device('cuda')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                   local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

# Прогрев модели
_ = model.apply_tts("Тестовый текст для прогрева модели", speaker='baya', sample_rate=48000)

def parse_role_and_text(text):
    role = None
    if '[' in text and ']' in text:
        role = text[text.index('[') + 1:text.index(']')]
        text = text[text.index(']') + 1:]
    return role, text

def get_speaker_by_role(role):
    if role == '1role':
        return 'eugene'
    elif role == '2role':
        return 'kseniya'
    elif role == '3role':
        return 'aidar'
    elif role == '4role':
        return 'baya'
    else:
        return 'baya'

def synthesize_speech(srt_file, output_file, pause_duration_ms=500):
    with open(srt_file, "r", encoding="utf-8") as f:
        srt_content = f.read()

    subtitles = list(srt.parse(srt_content))

    full_audio = AudioSegment.silent(duration=0)

    for i, subtitle in enumerate(subtitles):
        text = subtitle.content.strip()
        role, text = parse_role_and_text(text)
        text = text.replace('<i>', '').replace('</i>', '').replace('\n', ' ')
        speaker = get_speaker_by_role(role)

        audio = model.apply_tts(text, speaker=speaker, sample_rate=48000)
        audio = (audio * 2**15).cpu().numpy().astype(np.int16)

        speech_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=48000,
            sample_width=2,
            channels=1
        )

        start_time = int(subtitle.start.total_seconds() * 1000)
        end_time = int(subtitle.end.total_seconds() * 1000)

        # Добавляем смещение времени
        full_audio += AudioSegment.silent(duration=start_time - len(full_audio))
        full_audio += speech_segment

        # Добавляем паузу между репликами
        full_audio += AudioSegment.silent(duration=pause_duration_ms)

    full_audio.export(output_file, format="wav")

# Укажите путь к файлу субтитров и путь для сохранения итогового аудиофайла
srt_file = "name.srt"
output_file = "output.wav"

synthesize_speech(srt_file, output_file)
