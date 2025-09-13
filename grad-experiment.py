import re
import os
import gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
    precision_recall_curve,
)
from transformers import (
    BertJapaneseTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import TrainerCallback
from torch.nn import CrossEntropyLoss

# ====== しきい値方針 ======
TARGET_PRECISION = 0.90      # P目標（必要なら変更）
F_BETA = 1.0                 # Fβのβ（1.0=F1。P重視なら0.5、R重視なら2.0 など）
RECOMMEND_STRATEGY = "f_beta"  # "f_beta" or "precision_target"
OUTDIR = "artifacts"

# =============== 速度最適化（CUDA） ===============
def set_cuda_fast():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

# =============== 列名ロバスト検出 ===============
OP_CANDIDATES  = ["operator_utterance", "operator", "op_utt", "utterance_operator", "agent", "オペレーター", "オペレータ"]
CUS_CANDIDATES = ["customer_utterance", "customer", "cust_utt", "utterance_customer", "ユーザー", "カスタマー"]

def _find_column(cols, candidates):
    norm = {c: re.sub(r"[^a-z]", "", c.lower()) for c in cols}
    for cand in candidates:
        key = re.sub(r"[^a-z]", "", cand.lower())
        for orig, n in norm.items():
            if key == n or key in n or n in key:
                return orig
    return None

def _read_csv_any(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")

# =============== Dataset ===============
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def _make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    counts = {}; new_cols = []
    for c in map(str, df.columns):
        if c in counts:
            counts[c] += 1; new_cols.append(f"{c}.{counts[c]}")
        else:
            counts[c] = 0; new_cols.append(c)
    out = df.copy(); out.columns = new_cols; return out

def load_data(correct_path, incorrect_path):
    correct = _read_csv_any(correct_path)
    incorrect = _read_csv_any(incorrect_path)
    correct = _make_unique_columns(correct); incorrect = _make_unique_columns(incorrect)

    def map_cols(df):
        cols = list(df.columns)
        op_col  = _find_column(cols, OP_CANDIDATES)
        cus_col = _find_column(cols, CUS_CANDIDATES)
        if op_col is None or cus_col is None:
            raise KeyError(
                f"会話列が見つかりません。\n検出列: {cols}\n"
                f"operator候補={OP_CANDIDATES}\ncustomer候補={CUS_CANDIDATES}\n"
                "CSVヘッダを上記候補に合わせるか候補を増やしてください。"
            )
        sub = df[[op_col, cus_col]].copy()
        sub.columns = ["operator_utterance", "customer_utterance"]
        sub = sub.loc[:, ~sub.columns.duplicated(keep="first")]
        return sub

    correct  = map_cols(correct); incorrect = map_cols(incorrect)
    correct["label"] = 1; incorrect["label"] = 0
    all_data = pd.concat([correct, incorrect], ignore_index=True)
    all_data["operator_utterance"] = all_data["operator_utterance"].fillna("")
    all_data["customer_utterance"] = all_data["customer_utterance"].fillna("")
    return all_data[["operator_utterance", "customer_utterance", "label"]]

def preprocess_data(data, tokenizer, max_length=256):
    texts = (data["operator_utterance"] + f" {tokenizer.sep_token} " + data["customer_utterance"]).tolist()
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
    return Dataset(encodings, data["label"].tolist())

# =============== Custom Trainer / Callback ===============
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs); logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

class LossMonitorCallback(TrainerCallback):
    def __init__(self): self.train_losses = []; self.eval_losses = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        if logs.get("loss") is not None: self.train_losses.append(logs["loss"])
        if logs.get("eval_loss") is not None: self.eval_losses.append(logs["eval_loss"])

# =============== 予測（確率/ラベル） & ユーティリティ ===============
def predict_probs(model, dataset):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0, pin_memory=True)
    probs, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            inputs = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()}
            logits = model(**inputs).logits
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p); labels_all.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(labels_all)

def eval_at_threshold(y_true, probs, t):
    y_pred = (probs >= float(t)).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return acc, p, r, f1, y_pred

def best_threshold_f1(y_true, probs):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.10, 0.90, 81):
        _, _, _, f1, _ = eval_at_threshold(y_true, probs, t)
        if f1 > best_f1: best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def best_threshold_fbeta(y_true, probs, beta=1.0):
    # β<1でPrecision重視、β>1でRecall重視
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.10, 0.90, 81):
        y_pred = (probs >= t).astype(int)
        p, r, fb, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", beta=beta, zero_division=0)
        # sklearnのfβはf1_scoreには直接無いので上のPRFSで取得
        if fb > best_s: best_s, best_t = fb, t
    return float(best_t), float(best_s)

def min_t_for_precision(y_true, probs, p_target):
    prec, rec, th = precision_recall_curve(y_true, probs)   # th: len=N-1
    mask = prec[:-1] >= p_target
    if np.any(mask):
        idx = np.argmax(rec[:-1][mask])     # Precision満たす中でRecall最大
        return float(th[mask][idx])
    # 到達できない場合は最大Precision点
    idx = np.argmax(prec[:-1])
    return float(th[idx])

def misclassified_dataframe(val_df, y_true, y_pred, probs, threshold, fold):
    df_out = val_df.copy()
    df_out["label"] = y_true
    df_out["prob_pos"] = probs
    df_out["pred"] = y_pred
    df_out["correct"] = (df_out["label"] == df_out["pred"]).astype(int)
    df_out["fold"] = fold
    df_out["threshold"] = threshold
    # 誤分類のみ
    return df_out[df_out["correct"] == 0].copy()

# =============== Main ===============
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    set_cuda_fast()

    correct_path = "datas/positive.csv"
    incorrect_path = "datas/negative.csv"
    df_all = load_data(correct_path, incorrect_path)

    MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 集計
    all_metrics_f1, all_metrics_p = [], []
    best_thresholds_f1, best_thresholds_p = [], []
    oof_probs, oof_true, oof_rows = [], [], []   # OOF
    # 誤分類保存
    mis_all_fold = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_all[["operator_utterance","customer_utterance"]], df_all["label"]), start=1):
        print(f"\n=== Fold {fold} ===")
        train_df = df_all.iloc[train_idx].reset_index(drop=True)
        val_df   = df_all.iloc[val_idx].reset_index(drop=True)

        train_dataset = preprocess_data(train_df, tokenizer, max_length=256)
        val_dataset   = preprocess_data(val_df, tokenizer,   max_length=256)

        class_counts = train_df["label"].value_counts().sort_index().tolist()
        total = sum(class_counts)
        weights = torch.tensor([(total/c)**0.5 for c in class_counts], dtype=torch.float32, device=device)

        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold}",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_dir=f"./logs_fold_{fold}",
            logging_steps=50,
            report_to="none",
            bf16=True, fp16=False,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            save_total_limit=2,
            seed=42,
        )

        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        loss_monitor = LossMonitorCallback()
        early_stop = EarlyStoppingCallback(early_stopping_patience=2)

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            class_weights=weights,
            callbacks=[loss_monitor, early_stop],
            compute_metrics=lambda ev: {
                "accuracy": accuracy_score(ev[1], np.argmax(ev[0], axis=-1)),
                "precision": precision_recall_fscore_support(ev[1], np.argmax(ev[0], axis=-1), average="binary", zero_division=0)[0],
                "recall":    precision_recall_fscore_support(ev[1], np.argmax(ev[0], axis=-1), average="binary", zero_division=0)[1],
                "f1":        precision_recall_fscore_support(ev[1], np.argmax(ev[0], axis=-1), average="binary", zero_division=0)[2],
            },
        )

        trainer.train()
        trainer.evaluate()

        # ===== 検証予測 =====
        probs, y_true = predict_probs(model, val_dataset)

        # F1最大しきい値
        t_f1, _ = best_threshold_f1(y_true, probs)
        acc, p, r, f1, y_pred = eval_at_threshold(y_true, probs, t_f1)
        print(f"[ThreshOpt/F1max] t={t_f1:.3f} Acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f}")
        all_metrics_f1.append((acc, p, r, f1)); best_thresholds_f1.append(t_f1)

        # Precision目標
        t_p = min_t_for_precision(y_true, probs, TARGET_PRECISION)
        acc_p, pp, rr, ff, y_pred_p = eval_at_threshold(y_true, probs, t_p)
        print(f"[ThreshOpt/P≥{TARGET_PRECISION:.2f}] t={t_p:.3f} Acc={acc_p:.4f} P={pp:.4f} R={rr:.4f} F1={ff:.4f}")
        all_metrics_p.append((acc_p, pp, rr, ff)); best_thresholds_p.append(t_p)

        # Fβ最適（デフォルトβ=1.0でF1）
        t_fb, fb_score = best_threshold_fbeta(y_true, probs, beta=F_BETA)
        acc_fb, p_fb, r_fb, f1_fb, y_pred_fb = eval_at_threshold(y_true, probs, t_fb)
        print(f"[ThreshOpt/F{F_BETA:.1f}-max] t={t_fb:.3f} Acc={acc_fb:.4f} P={p_fb:.4f} R={r_fb:.4f} F1={f1_fb:.4f}")

        # 誤分類CSV（各fold）— 推奨戦略（RECOMMEND_STRATEGY）で保存
        t_use = {"f_beta": t_fb, "precision_target": t_p} .get(RECOMMEND_STRATEGY, t_fb)
        _, _, _, _, y_pred_use = eval_at_threshold(y_true, probs, t_use)
        mis_df = misclassified_dataframe(val_df, y_true, y_pred_use, probs, t_use, fold)
        if not mis_df.empty:
            mis_path = os.path.join(OUTDIR, f"misclassified_fold_{fold}.csv")
            mis_df.to_csv(mis_path, index=False, encoding="utf-8-sig")
            print(f"[Saved] {mis_path}  ({len(mis_df)} rows)")

        # OOF収集
        oof_probs.append(probs); oof_true.append(y_true)
        tmp = val_df.copy()
        tmp["prob_pos"] = probs; tmp["label"] = y_true; tmp["fold"] = fold
        oof_rows.append(tmp)

        del model; gc.collect(); torch.cuda.empty_cache()

    # ===== Cross-Validation means（参考：F1最適 & P目標）=====
    def _mean(ms): return tuple(sum(x[i] for x in ms)/len(ms) for i in range(4))
    mean_f1 = _mean(all_metrics_f1); mean_p = _mean(all_metrics_p)
    print("\n=== Cross-Validation Results (F1-opt threshold) ===")
    print(f"Mean Acc:{mean_f1[0]:.4f}  P:{mean_f1[1]:.4f}  R:{mean_f1[2]:.4f}  F1:{mean_f1[3]:.4f}")
    print("Per-fold F1-opt thresholds:", [round(float(t), 3) for t in best_thresholds_f1])

    print("\n=== Cross-Validation Results (Precision-target threshold) ===")
    print(f"Mean Acc:{mean_p[0]:.4f}  P:{mean_p[1]:.4f}  R:{mean_p[2]:.4f}  F1:{mean_p[3]:.4f}")
    print("Per-fold P-target thresholds:", [round(float(t), 3) for t in best_thresholds_p])

    # ===== OOFで本番用しきい値を1本決定 =====
    oof_probs = np.concatenate(oof_probs); oof_true = np.concatenate(oof_true)
    oof_df = pd.concat(oof_rows, ignore_index=True)
    # Fβ最適（推奨）
    t_grid = np.linspace(0.10, 0.90, 81)
    best_t_global, best_s = 0.5, -1.0
    for t in t_grid:
        y_pred = (oof_probs >= t).astype(int)
        p, r, f_beta, _ = precision_recall_fscore_support(oof_true, y_pred, average="binary", beta=F_BETA, zero_division=0)
        if f_beta > best_s: best_s, best_t_global = f_beta, t

    # Precision目標（比較用）
    t_global_p = min_t_for_precision(oof_true, oof_probs, TARGET_PRECISION)

    # 推奨閾値
    t_recommend = {"f_beta": best_t_global, "precision_target": t_global_p}.get(RECOMMEND_STRATEGY, best_t_global)

    # OOFスコア at 推奨閾値
    acc_o, po, ro, fo, y_pred_o = eval_at_threshold(oof_true, oof_probs, t_recommend)
    print("\n=== Deployment Threshold ===")
    print(f"Strategy: {RECOMMEND_STRATEGY}  Threshold: {t_recommend:.3f}")
    print(f"OOF Acc:{acc_o:.4f}  P:{po:.4f}  R:{ro:.4f}  F1:{fo:.4f}")
    print(f"(Also: F{F_BETA:.1f}-global={best_t_global:.3f},  P-target-global={t_global_p:.3f})")

    # OOF 誤分類CSV（推奨しきい値）
    mis_oof = oof_df.copy()
    mis_oof["pred"] = y_pred_o; mis_oof["correct"] = (mis_oof["label"] == mis_oof["pred"]).astype(int)
    mis_oof["threshold"] = t_recommend
    mis_oof_err = mis_oof[mis_oof["correct"] == 0].copy()
    if not mis_oof_err.empty:
        path = os.path.join(OUTDIR, "misclassified_oof.csv")
        mis_oof_err.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[Saved] {path}  ({len(mis_oof_err)} rows)")

    # OOF 予測フルログ（分析用）
    path_pred = os.path.join(OUTDIR, "oof_predictions.csv")
    oof_df.to_csv(path_pred, index=False, encoding="utf-8-sig")
    print(f"[Saved] {path_pred}  ({len(oof_df)} rows)")

if __name__ == "__main__":
    main()
