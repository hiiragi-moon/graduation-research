import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from transformers import (BertJapaneseTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, 
                          DataCollatorWithPadding, TrainerCallback)

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
    all_data["text"] = all_data["customer_utterance"]
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

# ネストされた辞書をフラットにする関数
def flatten_dict(d, parent_key='', sep='_'):
    flattened = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flattened.update(flatten_dict(v, new_key, sep=sep))
        else:
            flattened[new_key] = v
    return flattened

# 交差検証の実行
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    correct_path = "positive_1.csv"
    incorrect_path = "negative_1.csv"
    data = load_data(correct_path, incorrect_path)

    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(data, data["label"])):
        print(f"Fold {fold+1}/5")
        train_data, test_data = data.iloc[train_idx], data.iloc[test_idx]
        
        train_dataset = preprocess_data(train_data, tokenizer)
        test_dataset = preprocess_data(test_data, tokenizer)

        model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese", num_labels=2)
        model.to(device)
        
        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f"./logs_fold_{fold}",
            logging_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        loss_monitor = LossMonitorCallback()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            callbacks=[loss_monitor],
        )
        
        print("トレーニング開始...")
        trainer.train()
        
        print("評価開始...")
        predictions = trainer.predict(test_dataset)
        y_true = predictions.label_ids
        y_pred = predictions.predictions.argmax(axis=1)
        
        report = classification_report(y_true, y_pred, target_names=["Negative", "Positive"], output_dict=True)
        fold_results.append(flatten_dict(report))
        
        print(f"Fold {fold+1} 終了\n")
    
    # 結果の統合と保存
    avg_results = pd.DataFrame(fold_results).mean()
    avg_results.to_csv("cross_validation_results.csv", index=True)
    print("交差検証の平均評価結果を 'cross_validation_results.csv' に保存しました。")