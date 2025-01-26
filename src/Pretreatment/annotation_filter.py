import json
import csv

# 入力ファイルと出力ファイルを指定
input_file = 'input.json'
output_file = 'collect_utterances.csv'

def extract_utterance_triples(conversations):
    utterance_triples = []
    processed_utterances = set()

    if not isinstance(conversations, list):
        raise ValueError("Conversations data is not a list. Please check the JSON structure.")

    for i, entry in enumerate(conversations):
        if not isinstance(entry, dict):
            print(f"Skipping non-dictionary entry: {entry}")
            continue

        annotations = entry.get("annotation", [])
        for annotation in annotations:
            specific_tag = annotation.get("specific_tag")
            if specific_tag == "SpotRequirement":
                spot_utterance = entry.get('utterance', '')
                if not spot_utterance or spot_utterance in processed_utterances:
                    continue
                processed_utterances.add(spot_utterance)

                # Debug: SpotRequirement発話を確認
                print(f"Found SpotRequirement utterance: {spot_utterance}")

                spot_utterance = f"{'オペレーター:' if entry.get('speaker') == 'operator' else 'カスタマー:'} {spot_utterance}"

                # 遡って最も近いoperator発話を探す
                operator_utterance = ""
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

                # Debug: 文脈を確認
                print(f"Context: {previous_operator_utterance}, Operator: {operator_utterance}")

                utterance_triples.append((previous_operator_utterance, operator_utterance, spot_utterance))

    return utterance_triples

# JSONデータの読み込み
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 発話データを抽出
try:
    utterance_triples = extract_utterance_triples(data)
    print(f"Total utterance triples extracted: {len(utterance_triples)}")
except ValueError as e:
    print(f"Error processing data: {e}")
    utterance_triples = []

# 結果をCSVファイルに保存
if utterance_triples:
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['previous_operator_utterance', 'operator_utterance', 'spot_requirement_utterance'])
        writer.writerows(utterance_triples)

    print(f"SpotRequirement発話とその文脈を {output_file} に保存しました。")
else:
    print("No utterance triples found. File not saved.")
