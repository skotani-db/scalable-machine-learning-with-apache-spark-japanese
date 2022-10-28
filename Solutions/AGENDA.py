# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="696ca8a5-8c70-44c8-a7a5-f214919599ca"/>
# MAGIC 
# MAGIC # アジェンダ (Agenda)
# MAGIC ## Scalable Machine Learning with Apache Spark&trade;

# COMMAND ----------

# MAGIC %md <i18n value="09e1957b-630d-43a4-9b07-a47edcceadc5"/>
# MAGIC 
# MAGIC ## 1日目午前
# MAGIC | 時間 | レッスン  | 説明 |
# MAGIC |:----:|-------|-------------|
# MAGIC |  20m |  **概要説明** |  自己紹介、教室の設定など |
# MAGIC |  30m  |  **Spark/ML Overview (optional)** | Spark architecture review & ML terminology <br/>（注：このクラスではAirbnbのSF賃貸データを使って賃貸価格などを予測します） | 
# MAGIC | 10m | **休憩** ||                                      
# MAGIC | 35m | **[Data Cleansing]($./ML 01 - Data Cleansing)** | 欠損値、外れ値、データインピュテーションの扱い方 |  
# MAGIC | 35m | **[Data Exploration Lab]($./Labs/ML 01L - Data Exploration Lab)** | データの探索、対数正規分布、基準となる評価指標
# MAGIC |10m | **休憩** ||                                      
# MAGIC |30m | **[Linear Regression I]($./ML 02 - Linear Regression I)** | 簡単な単変量線形回帰モデルの構築<br/> SparkML API: transformer vs estimator |

# COMMAND ----------

# MAGIC %md <i18n value="62d32027-88c0-4709-9da1-ff7129b9ac5f"/>
# MAGIC 
# MAGIC ## 1日目午後
# MAGIC | 時間 | レッスン  | 説明 |
# MAGIC |:----:|-------|-------------|
# MAGIC | 20m | **[Linear Regression I Lab]($./Labs/ML 02L - Linear Regression I Lab)**    |  多変量線形回帰モデルの構築 <br/> RMSEとR2で評価する |
# MAGIC | 30m | **[Linear Regression II]($./ML 03 - Linear Regression II)**      | Spark でカテゴリ変数を扱う方法<br/> Pipeline API <br/> モデルの保存と読込 |
# MAGIC | 10m  | **休憩**  ||                                     
# MAGIC | 40m |**[Linear Regression II Lab]($./Labs/ML 03L - Linear Regression II Lab)** | RFormulaを用いたパイプラインの簡略化 <br/>対数スケールを予測する線形回帰モデルを構築し、予測値を指数化して評価する |
# MAGIC | 30m  | **[MLflow Tracking]($./ML 04 - MLflow Tracking)** | MLflow を使って実験を追跡し、メトリクスを記録して実行を比較する | 
# MAGIC | 10m  |  **休憩** ||                                
# MAGIC | 30m  | **[MLflow Model Registry]($./ML 05 - MLflow Model Registry)** | MLflow を使ってモデルを登録し、モデルのライフサイクルを管理する <br/> モデルのアーカイブと削除 |
# MAGIC | 40m  | **[MLflow Lab]($./Labs/ML 05L - MLflow Lab)** | MLflow を使ってモデルと Delta テーブルを追跡する <br/> モデルの登録 |

# COMMAND ----------

# MAGIC %md <i18n value="d7f9c802-4fe9-40ab-8293-c4c5d7a853a6"/>
# MAGIC 
# MAGIC ## 2日目午前
# MAGIC | 時間 | レッスン | 説明  |
# MAGIC |:----:|-------|-------------|
# MAGIC | 20m  | **Review**    |  *トピックのレビュー* |
# MAGIC | 40m    | **[Decision Trees]($./ML 06 - Decision Trees)**    | 決定木の分散処理とmaxBinsパラメータ（sklearnから異なる結果が得られる原因）<br/> 特徴量の重要度 |
# MAGIC | 10m  | **休憩**  ||                                     
# MAGIC | 40m  | **[Random Forests and Hyperparameter Tuning]($./ML 07 - Random Forests and Hyperparameter Tuning)** | ランダムフォレストは何がランダムか？ <br/> K-Foldクロスバリデーション <br/> パイプライン学習を高速化する並列処理とコツ |                                              
# MAGIC | 30m  | **[Hyperparameter Tuning Lab]($./Labs/ML 07L - Hyperparameter Tuning Lab)**  | ランダムフォレストのグリッドサーチを行う <br/>特徴量の重要度スコアと分類の評価基準の生成 <br/> sklearn と SparkML random forests の違いの確認 | 
# MAGIC | 10m  | **休憩**   ||
# MAGIC | 20m  | **[Hyperopt]($./ML 08 - Hyperopt)**  | ランダムフォレストのハイパーパラメータ探索を行い、最適なモデル構成を保存する |
# MAGIC | 20m    | **[Hyperopt Lab]($./Labs/ML 08L - Hyperopt Lab)**    | SparkTrials による scikit-learn モデルの分散型ハイパーパラメータチューニング |

# COMMAND ----------

# MAGIC %md <i18n value="a5d7a323-b346-4d8c-87aa-f676c937cc0f"/>
# MAGIC 
# MAGIC ## 二日目の午後
# MAGIC | 時間 |レッスン| 説明 |
# MAGIC |:----:|-------|-------------|
# MAGIC | 25m  | **[AutoML]($./ML 09 - AutoML)**  | Databricks AutoMLのAPIを使用して自動的にモデルを学習・チューニングする | 
# MAGIC | 15m    | **[AutoML Lab]($./Labs/ML 09L - AutoML Lab)**    | Databricks AutoML UIを使用して自動的にモデルを学習・チューニングする | 
# MAGIC | 20m    | **[Feature Store]($./ML 10 - Feature Store)**    | Feature Storeで特徴量の作成、結合、変換を行う | 
# MAGIC | 10m  | **休憩**  ||                                     
# MAGIC | 20m    | **[XGBoost]($./ML 11 - XGBoost)**    | Spark で3rdパーティのライブラリを使う <br/>Gradient boosted tree とそのバリエーションについて議論する |                               
# MAGIC | 15m    | **[Inference with Pandas UDFs]($./ML 12 - Inference with Pandas UDFs)**    | シングルノードのMLモデルを構築し、Pandas Scalar Iterator UDF & mapInPandasを使って並列適用する|
# MAGIC | 20m    | **[Pandas UDFs Lab]($./Labs/ML 12L - Pandas UDF Lab)**    |  分散推論ラボ |
# MAGIC | 10m  | **休憩** ||
# MAGIC | 15m    | **[Training with Pandas Function API]($./ML 13 - Training with Pandas Function API)**    | applyInPandasでモデルのグループを並列に構築する </br> MLflowでモデルを追跡する|
# MAGIC | 20m  | **[Pandas API on Spark]($./ML 14 - Pandas API on Spark)** | 中身にSparkを活用したPandasコードを書く |  
# MAGIC 
# MAGIC **ML electivesフォルダーに補足の参考ノートブックを格納している。**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
