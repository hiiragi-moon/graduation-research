import json
import csv
import random

# 入力ファイルと出力ファイルを指定
input_file = 'raw-data/cleaned.json'
output_file = 'negative_cases_random.csv'

# 負例を抽出する関数
def extract_negative_cases(conversations):
    negative_cases = []
    processed_utterances = set()

    if not isinstance(conversations, list):
        raise ValueError("Conversations data is not a list. Please check the JSON structure.")

    for i, entry in enumerate(conversations):
        if not isinstance(entry, dict):
            print(f"Skipping non-dictionary entry: {entry}")
            continue

        annotations = entry.get("annotation", [])
        contains_spot_requirement = any(ann.get("specific_tag") == "SpotRequirement" for ann in annotations)

        # SpotRequirement を含まない かつ 発話者がカスタマー の場合のみ対象とする
        if contains_spot_requirement or entry.get("speaker") != "customer":
            continue

        utterance = entry.get('utterance', '')
        if not utterance or utterance in processed_utterances:
            continue
        processed_utterances.add(utterance)

        # 発話情報を収集
        utterance_formatted = f"カスタマー: {utterance}"

        # 遡って最も近いオペレーター発話を探す
        operator_utterance = ""
        j = 0
        for j in range(1, i + 1):
            previous_entry = conversations[i - j]
            if previous_entry.get("speaker") == "operator":
                operator_utterance = f"オペレーター: {previous_entry.get('utterance')}"
                break

        # さらにその前のオペレーター発話を探す
        previous_operator_utterance = ""
        for k in range(j + 1, i + 1):
            earlier_entry = conversations[i - k]
            if earlier_entry.get("speaker") == "operator":
                previous_operator_utterance = f"オペレーター: {earlier_entry.get('utterance')}"
                break

        # 負例をリストに追加
        negative_cases.append((
            previous_operator_utterance,  # さらに前のオペレーター発話
            operator_utterance,          # 最も近いオペレーター発話
            utterance_formatted          # 負例の発話（カスタマー）
        ))

    return negative_cases

# JSONデータの読み込み
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 負例データを抽出
try:
    all_negative_cases = extract_negative_cases(data)
    print(f"Total negative cases extracted: {len(all_negative_cases)}")
except ValueError as e:
    print(f"Error processing data: {e}")
    all_negative_cases = []

# ランダムに1400件を抽出
random_negative_cases = random.sample(all_negative_cases, min(1063, len(all_negative_cases)))

# 結果をCSVファイルに保存
if random_negative_cases:
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['previous_operator_utterance', 'operator_utterance', 'negative_utterance'])
        writer.writerows(random_negative_cases)

    print(f"Random {len(random_negative_cases)} negative cases saved to {output_file}.")
else:
    print("No negative cases found. File not saved.")
