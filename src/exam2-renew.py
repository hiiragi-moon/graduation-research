import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.trainer_callback import TrainerCallback
from torch.nn import CrossEntropyLoss

# データセットクラス
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# データ読み込み
def load_data(correct_path, incorrect_path):
    correct_data = pd.read_csv(correct_path)
    incorrect_data = pd.read_csv(incorrect_path)

    correct_data["label"] = 1
    incorrect_data["label"] = 0

    all_data = pd.concat([correct_data, incorrect_data])
    all_data["text"] = all_data["operator_utterance"] + " [SEP] " + all_data["customer_utterance"]
    return all_data[["text", "label"]]

# データの前処理
def preprocess_data(data, tokenizer, max_length=128):
    encodings = tokenizer(
        data["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    return Dataset(encodings, data["label"].tolist())

# カスタムTrainerクラス（クラス重み対応）
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 損失監視用コールバック
class LossMonitorCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs.get("loss") is not None:
            self.train_losses.append(logs["loss"])
        if logs.get("eval_loss") is not None:
            self.eval_losses.append(logs["eval_loss"])

# モデルの推論
def predict_labels(model, dataset):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(model.device) for key, val in batch.items() if key != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            pred_labels.extend(preds)
            true_labels.extend(batch["labels"].tolist())

    return true_labels, pred_labels

# メイン関数
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    correct_path = "positive_2.csv"
    incorrect_path = "negative_2.csv"
    data = load_data(correct_path, incorrect_path)

    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(data["text"], data["label"])):
        print(f"\n=== Fold {fold+1} ===")

        train_data = data.iloc[train_idx]
        validation_data = data.iloc[val_idx]

        train_dataset = preprocess_data(train_data, tokenizer)
        validation_dataset = preprocess_data(validation_data, tokenizer)

        class_counts = train_data["label"].value_counts().sort_index().tolist()
        total_samples = sum(class_counts)
        class_weights = [total_samples / count for count in class_counts]
        class_weights = torch.tensor(class_weights).to(device)

        model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese", num_labels=2)
        model.to(device)

        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold+1}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f"./logs_fold_{fold+1}",
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        loss_monitor = LossMonitorCallback()

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            class_weights=class_weights,
            callbacks=[loss_monitor],
        )

        trainer.train()
        eval_result = trainer.evaluate()

        # 検証データでの評価
        true_labels, pred_labels = predict_labels(model, validation_dataset)

        classification_metrics = classification_report(true_labels, pred_labels, output_dict=True)
        accuracy = classification_metrics["accuracy"]
        precision = classification_metrics["1"]["precision"]
        recall = classification_metrics["1"]["recall"]
        f1_score = classification_metrics["1"]["f1-score"]  # F1スコアを追加

        print(f"Fold {fold+1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
        
        all_metrics.append((accuracy, precision, recall, f1_score))

        # 損失グラフ
        plt.figure(figsize=(10, 5))
        plt.plot(loss_monitor.train_losses, label="Training Loss")
        plt.plot(loss_monitor.eval_losses, label="Validation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss (Fold {fold+1})")
        plt.legend()
        plt.show()

    # 平均評価値を算出
    mean_accuracy = sum(m[0] for m in all_metrics) / len(all_metrics)
    mean_precision = sum(m[1] for m in all_metrics) / len(all_metrics)
    mean_recall = sum(m[2] for m in all_metrics) / len(all_metrics)
    mean_f1_score = sum(m[3] for m in all_metrics) / len(all_metrics)  # 平均F1スコア

    print("\n=== Cross-Validation Results ===")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean F1-score: {mean_f1_score:.4f}")  # 平均F1スコアを出力

if __name__ == "__main__":
    main()