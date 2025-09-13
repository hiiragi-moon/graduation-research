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
    fbeta_score,
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

# ====== 単位スイッチ ======
UNIT = "sentence"     # "sentence" or "pair"

# ★ 話者フィルタ（sentence のときのみ有効）
SPEAKER_FILTER = "customer"   # "customer" / "operator" / None

# ====== しきい値方針 ======
TARGET_PRECISION = 0.90
F_BETA = 1.0
# 追加: equal_pr を標準に
RECOMMEND_STRATEGY = "equal_pr"  # "equal_pr" / "balanced_pr" / "f_beta" / "precision_target"
OUTDIR = "artifacts"

# ====== 学習バランス微調整 ======
WEIGHT_GAMMA = 0.35  # 0.5→ややR寄り、0.35で中庸（P/Rのバランス取りやすい）

# =============== 速度最適化（CUDA） ===============
def set_cuda_fast():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

def print_env():
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"torch built with CUDA: {getattr(torch.version, 'cuda', None)}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

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

# =============== 文分割 & 文単位展開 ===============
_SENT_SPLIT = re.compile(r'(?<=[。．！？!?])\s*')

def sentence_split_ja(text: str):
    if not isinstance(text, str): return []
    text = text.strip()
    if not text: return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    return sents if sents else [text]

def to_sentence_level(df_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in df_pairs.iterrows():
        label = int(row["label"])
        op_sents  = sentence_split_ja(row["operator_utterance"])
        cu_sents  = sentence_split_ja(row["customer_utterance"])
        for j, s in enumerate(op_sents):
            rows.append({"pair_id": idx, "speaker": "operator", "sent_idx": j, "text": s, "label": label})
        for j, s in enumerate(cu_sents):
            rows.append({"pair_id": idx, "speaker": "customer", "sent_idx": j, "text": s, "label": label})
    out = pd.DataFrame(rows)
    out["text"] = out["text"].astype(str)
    out = out[out["text"].str.strip() != ""]
    return out.reset_index(drop=True)

# =============== 前処理（UNIT対応） ===============
def preprocess_data(data, tokenizer, max_length=256, unit="pair"):
    if unit == "sentence":
        texts = data["text"].tolist()
    else:
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

def best_threshold_balanced_pr(y_true, probs, min_f1_ratio=0.95):
    t_grid = np.linspace(0.10, 0.90, 81)
    f1s, ps, rs = [], [], []
    for t in t_grid:
        _, p, r, f1, _ = eval_at_threshold(y_true, probs, t)
        f1s.append(f1); ps.append(p); rs.append(r)
    f1_max = max(f1s)
    candidates = []
    for t, p, r, f1 in zip(t_grid, ps, rs, f1s):
        if f1 >= f1_max * float(min_f1_ratio):
            candidates.append((t, abs(p - r), p, r, f1))
    if not candidates:
        candidates = [(t, abs(p - r), p, r, f1) for t, p, r, f1 in zip(t_grid, ps, rs, f1s)]
    candidates.sort(key=lambda x: (x[1], -x[4], -x[2]))
    return float(candidates[0][0])

def best_threshold_equal_pr(y_true, probs):
    """Precision と Recall の差 |P-R| が最小になるしきい値"""
    t_grid = np.linspace(0.10, 0.90, 81)
    best = (None, 1e9, 0, 0, 0)  # (t, |P-R|, P, R, F1)
    for t in t_grid:
        _, p, r, f1, _ = eval_at_threshold(y_true, probs, t)
        gap = abs(p - r)
        cand = (t, gap, p, r, f1)
        if (gap, -f1, -p) < (best[1], -best[4], -best[2]):
            best = cand
    return float(best[0])

def best_threshold_f1(y_true, probs):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.10, 0.90, 81):
        _, _, _, f1, _ = eval_at_threshold(y_true, probs, t)
        if f1 > best_f1: best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def best_threshold_fbeta(y_true, probs, beta=1.0):
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.10, 0.90, 81):
        y_pred = (probs >= t).astype(int)
        s = fbeta_score(y_true, y_pred, beta=beta, average="binary", zero_division=0)
        if s > best_s: best_s, best_t = s, t
    return float(best_t), float(best_s)

def min_t_for_precision(y_true, probs, p_target):
    prec, rec, th = precision_recall_curve(y_true, probs)
    mask = prec[:-1] >= p_target
    if np.any(mask):
        idx = np.argmax(rec[:-1][mask])
        return float(th[mask][idx])
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
    return df_out[df_out["correct"] == 0].copy()

# =============== Main ===============
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print_env()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    set_cuda_fast()

    correct_path = "datas/positive.csv"
    incorrect_path = "datas/negative.csv"
    df_pairs = load_data(correct_path, incorrect_path)

    # UNIT切替
    if UNIT == "sentence":
        df_all = to_sentence_level(df_pairs)
        if SPEAKER_FILTER is not None:
            df_all = df_all[df_all["speaker"] == SPEAKER_FILTER].reset_index(drop=True)
        Xsplit = df_all["text"]
    else:
        df_all = df_pairs
        Xsplit = df_all[["operator_utterance", "customer_utterance"]]

    MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_metrics_f1, all_metrics_p = [], []
    best_thresholds_f1, best_thresholds_p = [], []
    best_thresholds_eq, best_thresholds_bal = [], []
    oof_probs, oof_true, oof_rows = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(Xsplit, df_all["label"]), start=1):
        print(f"\n=== Fold {fold} ({UNIT}) ===")
        train_df = df_all.iloc[train_idx].reset_index(drop=True)
        val_df   = df_all.iloc[val_idx].reset_index(drop=True)

        train_dataset = preprocess_data(train_df, tokenizer, max_length=256, unit=UNIT)
        val_dataset   = preprocess_data(val_df, tokenizer,   max_length=256, unit=UNIT)

        # class weights（平方根→係数で弱め）
        class_counts = train_df["label"].value_counts().sort_index().tolist()
        total = sum(class_counts)
        # (total/c)**gamma で重み付け（gamma=0なら無重み、0.35は中庸）
        weights = torch.tensor([(total/c)**WEIGHT_GAMMA for c in class_counts], dtype=torch.float32, device=device)

        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

        training_args = TrainingArguments(
            output_dir=f"./results_useronly_fold_{fold}",
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

        # F1最大
        t_f1, _ = best_threshold_f1(y_true, probs)
        acc, p, r, f1, y_pred = eval_at_threshold(y_true, probs, t_f1)
        print(f"[ThreshOpt/F1max] t={t_f1:.3f} Acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f}")
        all_metrics_f1.append((acc, p, r, f1)); best_thresholds_f1.append(t_f1)

        # Precision目標
        t_p = min_t_for_precision(y_true, probs, TARGET_PRECISION)
        acc_p, pp, rr, ff, y_pred_p = eval_at_threshold(y_true, probs, t_p)
        print(f"[ThreshOpt/P≥{TARGET_PRECISION:.2f}] t={t_p:.3f} Acc={acc_p:.4f} P={pp:.4f} R={rr:.4f} F1={ff:.4f}")
        all_metrics_p.append((acc_p, pp, rr, ff)); best_thresholds_p.append(t_p)

        # Fβ最大
        t_fb, fb_score = best_threshold_fbeta(y_true, probs, beta=F_BETA)
        acc_fb, p_fb, r_fb, f1_fb, y_pred_fb = eval_at_threshold(y_true, probs, t_fb)
        print(f"[ThreshOpt/F{F_BETA:.1f}-max] t={t_fb:.3f} Acc={acc_fb:.4f} P={p_fb:.4f} R={r_fb:.4f} F1={f1_fb:.4f}")

        # 追加: P≒R（差最小）
        t_eq = best_threshold_equal_pr(y_true, probs)
        acc_eq, p_eq, r_eq, f1_eq, y_pred_eq = eval_at_threshold(y_true, probs, t_eq)
        print(f"[ThreshOpt/P≈R] t={t_eq:.3f} Acc={acc_eq:.4f} P={p_eq:.4f} R={r_eq:.4f} F1={f1_eq:.4f}")
        best_thresholds_eq.append(t_eq)

        # 追加: balanced_pr（F1maxの98%維持で |P-R| 最小）
        t_bal = best_threshold_balanced_pr(y_true, probs, min_f1_ratio=0.98)
        acc_b, p_b, r_b, f1_b, y_pred_b = eval_at_threshold(y_true, probs, t_bal)
        print(f"[ThreshOpt/balanced_pr] t={t_bal:.3f} Acc={acc_b:.4f} P={p_b:.4f} R={r_b:.4f} F1={f1_b:.4f}")
        best_thresholds_bal.append(t_bal)

        # 誤分類CSV（推奨戦略で保存）
        t_use = {
            "equal_pr": t_eq,
            "balanced_pr": t_bal,
            "f_beta": t_fb,
            "precision_target": t_p,
        }.get(RECOMMEND_STRATEGY, t_eq)
        _, _, _, _, y_pred_use = eval_at_threshold(y_true, probs, t_use)
        mis_df = misclassified_dataframe(val_df, y_true, y_pred_use, probs, t_use, fold)
        if not mis_df.empty:
            path = os.path.join(OUTDIR, f"misclassified_useronly_{fold}.csv")
            mis_df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"[Saved] {path}  ({len(mis_df)} rows)")

        # OOF収集
        oof_probs.append(probs); oof_true.append(y_true)
        tmp = val_df.copy()
        tmp["prob_pos"] = probs; tmp["label"] = y_true; tmp["fold"] = fold
        oof_rows.append(tmp)

        del model; gc.collect(); torch.cuda.empty_cache()

    # ===== Cross-Validation means（参考）=====
    def _mean(ms): return tuple(sum(x[i] for x in ms)/len(ms) for i in range(4))
    mean_f1 = _mean(all_metrics_f1); mean_p = _mean(all_metrics_p)
    print("\n=== Cross-Validation Results (F1-opt threshold) ===")
    print(f"Mean Acc:{mean_f1[0]:.4f}  P:{mean_f1[1]:.4f}  R:{mean_f1[2]:.4f}  F1:{mean_f1[3]:.4f}")
    print("Per-fold F1-opt thresholds:", [round(float(t), 3) for t in best_thresholds_f1])
    print("\n=== Cross-Validation Results (Precision-target threshold) ===")
    print(f"Mean Acc:{mean_p[0]:.4f}  P:{mean_p[1]:.4f}  R:{mean_p[2]:.4f}  F1:{mean_p[3]:.4f}")
    print("Per-fold P-target thresholds:", [round(float(t), 3) for t in best_thresholds_p])
    print("\nPer-fold equal_pr thresholds:", [round(float(t), 3) for t in best_thresholds_eq])
    print("Per-fold balanced_pr thresholds:", [round(float(t), 3) for t in best_thresholds_bal])

    # ===== OOFで本番用しきい値を1本決定 =====
    oof_probs = np.concatenate(oof_probs); oof_true = np.concatenate(oof_true)
    oof_df = pd.concat(oof_rows, ignore_index=True)

    t_grid = np.linspace(0.10, 0.90, 81)

    # Fβ最適
    best_t_global, best_s = 0.5, -1.0
    for t in t_grid:
        y_pred = (oof_probs >= t).astype(int)
        s = fbeta_score(oof_true, y_pred, beta=F_BETA, average="binary", zero_division=0)
        if s > best_s: best_s, best_t_global = s, t

    # Precision目標
    t_global_p = min_t_for_precision(oof_true, oof_probs, TARGET_PRECISION)

    # equal_pr（OOF）
    best_eq = (None, 1e9, 0, 0, 0)  # (t, |P-R|, P, R, F1)
    for t in t_grid:
        _, p, r, f1, _ = eval_at_threshold(oof_true, oof_probs, t)
        gap = abs(p - r)
        cand = (t, gap, p, r, f1)
        if (gap, -f1, -p) < (best_eq[1], -best_eq[4], -best_eq[2]):
            best_eq = cand
    t_global_equal = float(best_eq[0])

    # balanced_pr（OOF）
    t_global_bal = best_threshold_balanced_pr(oof_true, oof_probs, min_f1_ratio=0.98)

    # 推奨閾値
    t_recommend = {
        "equal_pr": t_global_equal,
        "balanced_pr": t_global_bal,
        "f_beta": best_t_global,
        "precision_target": t_global_p,
    }.get(RECOMMEND_STRATEGY, t_global_equal)

    # OOFスコア at 推奨閾値
    acc_o, po, ro, fo, y_pred_o = eval_at_threshold(oof_true, oof_probs, t_recommend)
    print("\n=== Deployment Threshold ===")
    print(f"UNIT: {UNIT}  Strategy: {RECOMMEND_STRATEGY}  Threshold: {t_recommend:.3f}")
    print(f"OOF Acc:{acc_o:.4f}  P:{po:.4f}  R:{ro:.4f}  F1:{fo:.4f}")
    print(f"(Also: F{F_BETA:.1f}-global={best_t_global:.3f},  P-target-global={t_global_p:.3f},  equal_pr-global={t_global_equal:.3f},  balanced_pr-global={t_global_bal:.3f})")

    # OOF 誤分類CSV（推奨しきい値）
    mis_oof = oof_df.copy()
    mis_oof["pred"] = y_pred_o; mis_oof["correct"] = (mis_oof["label"] == mis_oof["pred"]).astype(int)
    mis_oof["threshold"] = t_recommend
    mis_oof_err = mis_oof[mis_oof["correct"] == 0].copy()
    if not mis_oof_err.empty:
        path = os.path.join(OUTDIR, f"misclassified_oof_useronly.csv")
        mis_oof_err.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[Saved] {path}  ({len(mis_oof_err)} rows)")

    # OOF 予測フルログ（分析用）
    path_pred = os.path.join(OUTDIR, "oof_predictions_useronly.csv")
    oof_df.to_csv(path_pred, index=False, encoding="utf-8-sig")
    print(f"[Saved] {path_pred}  ({len(oof_df)} rows)")

if __name__ == "__main__":
    main()
