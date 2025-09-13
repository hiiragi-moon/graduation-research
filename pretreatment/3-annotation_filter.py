import json
import csv
import re

# 入力ファイルと出力ファイル
INPUT_FILE = "cleaned.json"
OUTPUT_FILE = "spot_requirement_pairs.csv"

def tidy_inline(s: str) -> str:
    """軽整形：空白圧縮、句読点連続圧縮、句読点直前の空白除去"""
    if not s:
        return s
    s = re.sub(r"[ \u3000]+", " ", s)
    s = re.sub(r"[、]{2,}", "、", s)
    s = re.sub(r"[。]{2,}", "。", s)
    s = re.sub(r"\s+(?=[、。])", "", s)
    return s.strip()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

pairs = []

# 1) SpotRequirement を含むカスタマー発話と、その直前のオペレーター発話をペア化
for i, entry in enumerate(data):
    if "annotation" not in entry:
        continue

    # SpotRequirement の有無チェック
    if not any(ann.get("specific_tag") == "SpotRequirement" for ann in entry["annotation"]):
        continue

    spot_req_utt = tidy_inline(entry.get("utterance", ""))

    # 直前からさかのぼって最初のオペレーター発話を取得
    operator_utt = ""
    for j in range(i - 1, -1, -1):
        if data[j].get("speaker") == "operator":
            operator_utt = tidy_inline(data[j].get("utterance", ""))
            break

    pairs.append([f"オペレーター: {operator_utt}", f"カスタマー: {spot_req_utt}"])

# 2) 連続して同じオペレーター発話なら、カスタマー側を結合
merged_pairs = []
for op, cu in pairs:
    if merged_pairs and merged_pairs[-1][0] == op:
        # 直前の行と同じオペレーター：カスタマー発話を結合
        prev_op, prev_cu = merged_pairs[-1]
        # 末尾が句点等で終わってなければ読点でつなぐ
        if not re.search(r"[。！？!?]$", prev_cu):
            prev_cu = prev_cu.rstrip("、") + "、"
        # "カスタマー: "のラベルを外して結合→整形→再付与
        new_text = prev_cu.replace("カスタマー: ", "") + cu.replace("カスタマー: ", "")
        new_text = tidy_inline(new_text)
        merged_pairs[-1][1] = f"カスタマー: {new_text}"
    else:
        merged_pairs.append([op, cu])

# 3) CSVに保存
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["operator_utterance", "spot_requirement_utterance"])
    writer.writerows(merged_pairs)

print(f"{OUTPUT_FILE} を出力しました。（同一オペレーター連続を結合済み）")