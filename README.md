# Graduation Reseash

## 概要

Graduation Reseash は、対話データの前処理・アノテーションフィルタリング・機械学習実験を行うためのプロジェクトです。  
主にカスタマーサポート対話の発話分類やアノテーションの整理、BERT を用いた分類モデルの学習・評価を目的としています。

## ディレクトリ構成

```
graduation-research/
├── exam1-renew.py                # BERTによる発話分類実験スクリプト1
├── exam2-renew.py                # BERTによる発話分類実験スクリプト2
├── grad-experiment.py            # 発話分類実験メインスクリプト
├── grad-experiment-useronly.py   # ユーザー発話のみを対象とした実験スクリプト
├── gpu-test.py                   # GPU動作確認用スクリプト
├── test.py                       # テスト用スクリプト
├── artifacts/                    # 実験結果や中間生成物
├── datas/                        # 入力データ（positive/negative サンプル等）
├── pretreatment/                 # 前処理用スクリプト群
│   ├── 1-merge.py                # JSONファイル統合
│   ├── 2-clean.py                # データクリーニング
│   ├── 3-annotation_filter.py    # アノテーションフィルタ
│   └── 4-incorect_annotation_filter.py # 誤アノテーションフィルタ
├── results/                      # 実験結果
├── results_fold_1/ ...           # クロスバリデーション各foldの結果
└── .gitignore                    # Git管理除外ファイル
```

## セットアップ

1. 必要な Python パッケージをインストールしてください（例: transformers, torch, pandas, scikit-learn など）。
    ```
    pip install -r requirements.txt
    ```
    ※ requirements.txt がない場合は、各スクリプトの import を参照してインストールしてください。

2. `datas/` ディレクトリに `positive.csv` と `negative.csv` を配置してください。

3. 前処理を行う場合は `pretreatment/` 内のスクリプトを順に実行してください。

## 主なスクリプト

- `pretreatment/1-merge.py`  
  JSON ファイルを統合し、1つのファイルにまとめます。

- `pretreatment/2-clean.py`  
  統合済みデータのクリーニングを行います。

- `pretreatment/3-annotation_filter.py`  
  特定アノテーション（例: SpotRequirement）を含む発話ペアを抽出します。

- `pretreatment/4-incorect_annotation_filter.py`  
  誤アノテーション例を抽出します。

- `exam1-renew.py`, `exam2-renew.py`  
  BERT を用いた発話分類実験を行います。

- `grad-experiment.py`, `grad-experiment-useronly.py`  
  発話分類のクロスバリデーション実験を行います。

## 実行例

```sh
python pretreatment/1-merge.py
python pretreatment/2-clean.py
python pretreatment/3-annotation_filter.py
python exam1-renew.py
```

## 補足

- `artifacts/` や `results/` 以下には実験結果や中間生成物が保存されます。
- `.gitignore` で大容量データや一時ファイルは Git 管理対象外としています。

---

ご不明点があれば、各スクリプトのコメントやコードをご参照ください。
