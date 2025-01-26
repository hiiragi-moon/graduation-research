import pandas as pd

# CSVファイルの読み込み
input_file = 'negative_annotation_3.csv'  # 元のファイル名
output_file = 'negative_annotation_1.csv'  # 抽出後のファイル名

# データを読み込む
data = pd.read_csv(input_file)

# 必要な列を抽出（operator_utterance と customer_utterance）
filtered_data = data[['customer_utterance']]

# 抽出したデータを保存
filtered_data.to_csv(output_file, index=False)

print(f"Filtered data has been saved to {output_file}")
