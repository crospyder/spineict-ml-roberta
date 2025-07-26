from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "zero-shot-classification",
    model="microsoft/Multilingual-MiniLM-L12-H384",
    device=device  # 0 za GPU, -1 za CPU
)

result = classifier("Ovo je test OCR teksta", candidate_labels=["IZVOD", "UGOVOR", "URA", "IRA", "OSTALO"])

print(result)
