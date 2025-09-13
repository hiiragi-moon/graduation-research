import json
import os

# JSONファイルが格納されているディレクトリと出力する統合ファイル名
input_dir = 'src/pretreatment/json_files'  # JSONファイルが保存されているフォルダ名
output_file = 'src/pretreatment/merged.json'  # 統合されたJSONファイルの保存先

# 統合するためのリスト
merged_data = []

# ディレクトリ内のすべてのJSONファイルを処理
for file_name in os.listdir(input_dir):
    if file_name.endswith('.json'):  # JSONファイルのみを対象
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # データがリスト型であることを確認
                if isinstance(data, list):
                    merged_data.extend(data)  # リストとして統合
                else:
                    print(f"Warning: {file_path} is not a list. Skipping.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_path}: {e}")

# 統合されたデータを1つのJSONファイルに保存
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"すべてのJSONファイルを統合し、{output_file} に保存しました。")
