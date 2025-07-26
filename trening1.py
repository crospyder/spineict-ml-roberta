import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate

# 1. Učitaj CSV
df = pd.read_csv('export_cisti.csv', dtype=str)

# 2. Spoji kolone u jedan tekst
def combine_columns(row):
    parts = []
    for col in ['invoice_number', 'supplier_name', 'partner_name']:
        if pd.notna(row[col]) and row[col] != '':
            parts.append(str(row[col]))
    return ' | '.join(parts)

df['text'] = df.apply(combine_columns, axis=1)

# 3. Filter praznih labela i teksta
df = df[df['document_type'].notna()]
df = df[df['text'].notna()]
df = df[df['text'] != '']

# 4. Mapiranje labela
labels = df['document_type'].unique().tolist()
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
df['label'] = df['document_type'].map(label2id)

# 5. Hugging Face Dataset
dataset = Dataset.from_pandas(df[['text', 'label']])

# 6. Tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/Multilingual-MiniLM-L12-H384')

def preprocess(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(preprocess, batched=True)

# 7. Podjela na train/test
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test['train']
eval_dataset = train_test['test']

# 8. Model
model = AutoModelForSequenceClassification.from_pretrained(
    'microsoft/Multilingual-MiniLM-L12-H384',
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# 9. Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

# 10. Metric
metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 11. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 12. Pokreni trening
trainer.train()
