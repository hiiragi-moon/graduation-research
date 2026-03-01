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
