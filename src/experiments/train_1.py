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

# メイン関数
def main():
    # GPU/CPUの確認
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # データ読み込み
    correct_path = "annotation.csv"
    incorrect_path = "incorrect_annotation.csv"
    data = load_data(correct_path, incorrect_path)

    # データ分割
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # トークナイザー準備
    tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    # データセット準備
    train_dataset = preprocess_data(train_data, tokenizer)
    test_dataset = preprocess_data(test_data, tokenizer)

    # モデル準備
    model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese", num_labels=2)
    model.to(device)  # モデルをGPU/CPUに移動

    # 訓練設定
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Trainerの作成
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # モデル訓練
    print("トレーニング開始...")
    trainer.train()

    # 推論と評価
    print("評価中...")
    true_labels, pred_labels = predict_labels(model, tokenizer, test_dataset)

    # 評価結果を表示
    print("詳細な評価結果:")
    print(classification_report(true_labels, pred_labels, target_names=["No SpotRequirement", "SpotRequirement"]))

if __name__ == "__main__":
    main()
