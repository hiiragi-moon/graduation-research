import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

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

    # ラベル付け
    correct_data["label"] = 1
    incorrect_data["label"] = 0

    # データ結合
    all_data = pd.concat([correct_data, incorrect_data])

    # テキスト結合
    all_data["text"] = all_data["previous_operator_utterance"] + " [SEP] " + all_data["current_utterance"]
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

# モデルの推論
def predict_labels(model, tokenizer, dataset):
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

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    correct_path = "annotation.csv"
    incorrect_path = "incorrect_annotation.csv"
    data = load_data(correct_path, incorrect_path)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    train_dataset = preprocess_data(train_data, tokenizer)
    test_dataset = preprocess_data(test_data, tokenizer)

    class_counts = train_data["label"].value_counts().sort_index().tolist()
    total_samples = sum(class_counts)
    class_weights = [total_samples / count for count in class_counts]
    class_weights = torch.tensor(class_weights).to(device)

    model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese", num_labels=2)
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 損失監視コールバックのインスタンス
    loss_monitor = LossMonitorCallback()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        class_weights=class_weights,
        callbacks=[loss_monitor],  # コールバック追加
    )

    print("トレーニング開始...")
    train_result = trainer.train()

    # 損失を取得
    train_loss = train_result.training_loss

    print("評価中...")
    eval_result = trainer.evaluate()

    # 評価損失を取得
    eval_loss = eval_result["eval_loss"]

    # 推論と評価
    true_labels, pred_labels = predict_labels(model, test_dataset)

    test_texts = test_data["text"].tolist()
    test_df = pd.DataFrame({"text": test_texts, "true_label": true_labels})

    # 詳細な評価結果
    classification_metrics = classification_report(true_labels, pred_labels, target_names=["No SpotRequirement", "SpotRequirement"], output_dict=True)
    accuracy = classification_metrics["accuracy"]
    precision = classification_metrics["SpotRequirement"]["precision"]
    recall = classification_metrics["SpotRequirement"]["recall"]

    # 結果出力
    print("\nトレーニング結果:")
    print(f"学習率: {training_args.learning_rate}")
    print(f"訓練バッチサイズ: {training_args.per_device_train_batch_size}")
    print(f"評価バッチサイズ: {training_args.per_device_eval_batch_size}")
    print(f"エポック数: {training_args.num_train_epochs}")
    print(f"重みの減衰: {training_args.weight_decay}")
    print(f"訓練損失: {train_loss}")
    print(f"評価損失: {eval_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # 分類が正しく行われていないデータを抽出
    result_df = pd.DataFrame({
        "text": test_texts,
        "true_label": true_labels,
        "pred_label": pred_labels
    })

    incorrect_predictions = result_df[result_df["true_label"] != result_df["pred_label"]]

    # 不正解データを出力
    print("\n分類が正しく行われていないデータ:")
    print(incorrect_predictions)

    # 必要であればCSVとして保存
    incorrect_predictions.to_csv("incorrect_predictions.csv", index=False, encoding="utf-8")
    print("\n不正解データが 'incorrect_predictions.csv' として保存されました。")

    # 損失グラフの描画
    plt.figure(figsize=(10, 5))
    plt.plot(loss_monitor.train_losses, label="Training Loss")
    plt.plot(loss_monitor.eval_losses, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()
