import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    device_map="mps",
    dtype=torch.bfloat16,
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(
        device_map="mps",
        dtype=torch.bfloat16,
    ),
)

results = model.transcribe(
    audio=[
        "/Volumes/untitled/karaoker/runs/map2/audio/vocals_dry.wav",
    ],
    language=[
        "Japanese",
    ],  # can also be set to None for automatic language detection
    return_time_stamps=True,
)

for r in results:
    print(r.language, r.text, r.time_stamps)
