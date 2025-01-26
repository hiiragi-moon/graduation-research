import json
import pandas as pd
import os
import random

def extract_utterances_to_csv(file_path, negative_sample_size=100):
    # JSONファイルを読み込む
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # JSONがリストのリスト形式かどうかを判定し、展開
    utterances = []
    if isinstance(json_data, list):
        for conversation in json_data:
            if isinstance(conversation, list):
                utterances.extend(conversation)
            elif isinstance(conversation, dict):  # 単一の会話が含まれる場合
                utterances.append(conversation)
    else:
        raise ValueError("Unexpected JSON structure. Must be a list or list of lists.")

    # 特定のタグが含まれるセグメントを抽出
    extracted_data = []
    negative_data = []
    last_operator_utterance = None  # 直前のoperatorの発話を格納
    last_utterance = None           # さらにその前の発話と発話者を格納

    for current_utterance in utterances:
        # 辞書形式でない場合をスキップ
        if not isinstance(current_utterance, dict):
            continue

        # 現在の発話がoperatorの発話であれば、直前の発話として記録
        if current_utterance.get("speaker") == "operator":
            last_utterance = {  # さらにその前の発話に現在の直前発話を移動
                "speaker": "operator",
                "utterance": last_operator_utterance
            }
            last_operator_utterance = current_utterance.get("utterance")

        # 現在の発話が"specific_tag"が"SpotRequirement"のセグメントを持っている場合
        if last_operator_utterance and last_utterance and "annotation" in current_utterance:
            annotations = current_utterance.get("annotation", [])
            for annotation in annotations:
                if isinstance(annotation, dict) and annotation.get("specific_tag") == "SpotRequirement":
                    # さらにその前の発話、直前のオペレーター発話、現在の発話を抽出
                    extracted_data.append({
                        "two_previous_utterance": f"{last_utterance['speaker']}: {last_utterance['utterance']}",
                        "previous_operator_utterance": f"オペレーター: {last_operator_utterance}",
                        "current_utterance": f"カスタマー: {current_utterance.get('utterance')}"
                    })
                    break  # 一度処理したら次の発話に進む
        else:
            # 負例として記録
            if last_operator_utterance and last_utterance:
                negative_data.append({
                    "two_previous_utterance": f"{last_utterance['speaker']}: {last_utterance['utterance']}",
                    "previous_operator_utterance": f"オペレーター: {last_operator_utterance}",
                    "current_utterance": f"カスタマー: {current_utterance.get('utterance')}"
                })

    # 負例をランダムにサンプリング
    if len(negative_data) == 0:
        print("No negative samples found.")
        negative_data_sampled = []
    else:
        negative_data_sampled = random.sample(negative_data, min(len(negative_data), negative_sample_size))

    # 抽出したデータをpandas DataFrameに変換
    positive_df = pd.DataFrame(extracted_data)
    negative_df = pd.DataFrame(negative_data_sampled)

    # 出力ファイル名を決定
    positive_output_file = os.path.join(os.path.dirname(file_path), "annotation_with_context_positive.csv")
    negative_output_file = os.path.join(os.path.dirname(file_path), "annotation_with_context_negative.csv")

    # CSVファイルとして保存
    if not positive_df.empty:
        positive_df.to_csv(positive_output_file, index=False)
        print(f"Positive data saved: {len(positive_df)} records.")
    else:
        print("No positive data to save.")

    if not negative_df.empty:
        negative_df.to_csv(negative_output_file, index=False)
        print(f"Negative data saved: {len(negative_df)} records.")
    else:
        print("No negative data to save.")

    return positive_output_file, negative_output_file

# 使用例:
file_path = "merged.json"  # JSONファイルのパスを指定
try:
    positive_csv, negative_csv = extract_utterances_to_csv(file_path)
    print(f"Positive CSVファイルが作成されました: {positive_csv}")
    print(f"Negative CSVファイルが作成されました: {negative_csv}")
except Exception as e:
    print(f"Error: {e}")