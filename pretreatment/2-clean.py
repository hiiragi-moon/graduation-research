import json
import re

# ===== 設定 =====
INPUT_FILE = "raw-data/merged.json"
OUTPUT_FILE = "raw-data/cleaned.json"
REMOVE_TAGS = {"AutoPositive", "AutoNegative", "Stalling"}

def tidy_text(s: str) -> str:
    """削除後のテキストを軽く整える（空白・句読点のだぶりを解消）"""
    if not s:
        return s
    # 連続スペース（半角/全角）を1つに
    s = re.sub(r"[ \u3000]+", " ", s)
    # 文頭・文末の空白を除去
    s = s.strip()
    # 句読点や読点が連続したものを1つに圧縮（必要に応じて追加）
    s = re.sub(r"[、]{2,}", "、", s)
    s = re.sub(r"[。]{2,}", "。", s)
    # 句読点の直前/直後のスペース調整（「 、」「 。」→「、」「。」）
    s = re.sub(r"\s+(?=[、。])", "", s)
    return s

def is_empty_annotation(ann_list):
    return not ann_list  # 空リストなら True

def is_empty_utterance(utt: str) -> bool:
    return not (utt or "").strip()

def main():
    # 読み込み
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    for entry in data:
        annotations = entry.get("annotation", [])
        utter = entry.get("utterance", "")

        # --- 不要タグの削除＋対応 segment を utterance から削除 ---
        new_annotations = []
        for ann in annotations:
            tag = ann.get("tag")
            seg = ann.get("segment", "")
            if tag in REMOVE_TAGS:
                if seg:
                    # segment を完全削除（単純置換）
                    utter = utter.replace(seg, "")
            else:
                new_annotations.append(ann)

        # 整形
        utter = tidy_text(utter)

        # 更新
        entry["annotation"] = new_annotations
        entry["utterance"] = utter

        # --- 空要素の削除（annotation も utterance も空のものを除去）---
        if is_empty_annotation(new_annotations) and is_empty_utterance(utter):
            continue  # 追加しない＝削除

        cleaned.append(entry)

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4)

    print(f"{OUTPUT_FILE} を出力しました！（不要タグ除去＋utterance修正＋空要素削除）")

if __name__ == "__main__":
    main()