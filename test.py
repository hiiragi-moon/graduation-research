from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch

# 日本語BERT
model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ダミーデータ（4件）
texts = ["海が見たいです", "静かな温泉がいいです", "夜景の綺麗な場所", "歴史的な寺院を回りたい"]
labels = [0, 1, 0, 1]

encodings = tokenizer(texts, truncation=True, padding=True, max_length=32)
dataset = torch.utils.data.TensorDataset(
    torch.tensor(encodings["input_ids"]),
    torch.tensor(encodings["attention_mask"]),
    torch.tensor(labels)
)

# データコラレーター
collator = DataCollatorWithPadding(tokenizer)

# bf16 を有効にした TrainingArguments
args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    logging_steps=1,
    bf16=True,   # ← GPUでbfloat16を使う
    fp16=False,
    evaluation_strategy="no",
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)

# 学習実行
trainer.train()
