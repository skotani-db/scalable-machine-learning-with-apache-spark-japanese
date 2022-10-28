# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="2630af5a-38e6-482e-87f1-1a1633438bb6"/>
# MAGIC 
# MAGIC # AutoML
# MAGIC 
# MAGIC Databricksの <a href="https://docs.databricks.com/applications/machine-learning/automl.html" target="_blank">AutoML</a> を使って、UIもしくはプログラムから機械学習モデルを自動的に構築することができます。モデル学習のためにデータセットを準備し、（HyperOptを使用して）複数のモデルを作成、チューニング、評価する一連の試行を実行して記録します。 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このレッスンでは、以下のことを行います。<br>
# MAGIC  - AutoMLを使用してモデルを自動的にトレーニングおよびチューニングする
# MAGIC  - PythonとUIでAutoMLを実行する
# MAGIC  - AutoMLの実行結果を解釈する

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="7aa84cf3-1b6c-4ba4-9249-00359ee8d70a"/>
# MAGIC 
# MAGIC 現在、AutoMLはXGBoostとsklearn（シングルノードモデルのみ）を使用しており、それぞれでハイパーパラメータを最適化しています。

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="1b5c8a94-3ac2-4977-bfe4-51a97d83ebd9"/>
# MAGIC 
# MAGIC AutoMLを使って最適な <a href="https://docs.databricks.com/applications/machine-learning/automl.html#regression" target="_blank">回帰</a> モデルを探索できます。 
# MAGIC 
# MAGIC 必要なパラメータ
# MAGIC * **`dataset`** \- 学習用の特徴量とターゲット値を格納したSpark DataFrame または Pandas DataFrame を用意します。Spark DataFrameを使う場合、.toPandas()を呼び出すことで内部でPandas DataFrameに変換します。データ量が多い場合にはOOM (out of memory)にならないように注意してください。
# MAGIC * **`target_col`** \- ターゲットラベルのカラム名
# MAGIC 
# MAGIC また、オプションのパラメータとして次を指定できます。
# MAGIC * **`primary_metric`** \- 最適なモデルを選択するために優先的に使用するメトリック。各試行ではいくつかのメトリックを計算しますが、このprimary\_metricによって最適なモデルを決定します。回帰に対していは、次のいずれかを指定します：  **`r2`** (R squared - デフォルト), **`mse`** (mean squared error : 平均二乗誤差), **`rmse`** (root mean squared error : 二乗平均平方根誤差), **`mae`** (mean absolute error : 平均絶対誤差)。
# MAGIC * **`timeout_minutes`** \- AutoML の試行が完了するまでの最大待ち時間。 **`timeout_minutes=None`** と指定するとタイムアウトの制約を受けずに試行を実行します。
# MAGIC * **`max_trials`** \- 実行する試行回数の最大値。 **`max_trials=None`** と指定すると、完了するまで実行します。(Databricks Runtime 10.3 MLからは、max_trialsが廃止されたため、設定しても無効になります。 )

# COMMAND ----------

from databricks import automl

summary = automl.regress(train_df, target_col="price", primary_metric="rmse", timeout_minutes=5, max_trials=10)

# COMMAND ----------

# MAGIC %md <i18n value="57d884c6-2099-4f34-b840-a4e873308ffe"/>
# MAGIC 
# MAGIC 前のセルを実行すると、2つのノートブックと1つのMLflow実験が表示されます。
# MAGIC * **`Data exploration notebook (データ探索ノートブック)`** \- 入力列ごとの値、頻度とその他の情報をまとめたデータ概要レポート。
# MAGIC * **`Best trial notebook (最適試行ノートブック`** \- AutoMLによって構築されたベストモデルを再現するためのソースコード。
# MAGIC * **`MLflow experiment`** \- ArtifactのRootロケーション、experiment(実験)のID、タグなどのハイレベルな情報が含まれています。実行（Run）一覧には、ノートブックやモデルの場所、トレーニングパラメータ、総合的なメトリクスなど、各トライアルに関する詳細なサマリーが記載されています。
# MAGIC 
# MAGIC このノートブックとMLflowの実験を掘り下げると、何がわかるでしょうか？
# MAGIC 
# MAGIC さらに、AutoMLは、最適な試行から得られたメトリックの短いリストを表示します。

# COMMAND ----------

print(summary.best_trial)

# COMMAND ----------

# MAGIC %md <i18n value="3c0cd1ec-8965-4af3-896d-c30938033abf"/>
# MAGIC 
# MAGIC さて、AutoMLから取得したモデルをテストデータに対してテストすることができます。<a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf" target="_blank">mlflow.pyfunc.spark\_udf</a> を使って、モデルをUDFとして登録し、テストデータに対して並列に適用していきます。

# COMMAND ----------

# Load the best trial as an MLflow Model
import mlflow

model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"

predict = mlflow.pyfunc.spark_udf(spark, model_uri)
pred_df = test_df.withColumn("prediction", predict(*test_df.drop("price").columns))
display(pred_df)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE on test dataset: {rmse:.3f}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
