# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="b27f81af-5fb6-4526-b531-e438c0fda55e"/>
# MAGIC 
# MAGIC # MLflow
# MAGIC 
# MAGIC <a href="https://mlflow.org/docs/latest/concepts.html" target="_blank">MLflow</a>は、この3つの核心的な問題を解決しようとしています。
# MAGIC 
# MAGIC * 実験の追跡が難しい
# MAGIC * コードの再現が難しい
# MAGIC * モデルのパッケージングとデプロイの標準的な方法はない
# MAGIC 
# MAGIC 従来は、問題を調査する際に、構築した多数のモデルのそれぞれのパラメータとメトリクスを手作業で記録しておく必要がありました。このようなことはすぐに面倒になり、かなり時間がかかってしまいます。そこで、MLflowの出番となるわけです。
# MAGIC 
# MAGIC MLflowはDatabricks Runtime for MLにプレインストールされています。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは以下を行います。<br>
# MAGIC * MLflowを使った実験の追跡、メトリクスの記録、実行の比較など
# MAGIC 

# COMMAND ----------

# MAGIC %md <i18n value="b7c8a0e0-649e-4814-8310-ae6225a57489"/>
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="c1a29688-f50a-48cf-9163-ebcc381dfe38"/>
# MAGIC 
# MAGIC まずはSF Airbnb Datasetをロードしてみましょう。

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)
print(train_df.cache().count())

# COMMAND ----------

# MAGIC %md <i18n value="9ab8c080-9012-4f38-8b01-3846c1531a80"/>
# MAGIC 
# MAGIC ### MLflow トラッキング (MLflow Tracking)
# MAGIC 
# MAGIC MLflow トラッキングは、機械学習に特化したロギングAPIであり、学習に使うライブラリや環境に依存しないのが特徴です。 データサイエンス・コードの実行である**run**という概念を中心に構成されています。 Run は **experiment** に集約され、一つのexperimentが多数のrunを管理することが可能であり、MLflow サーバが多数のexperimentをホストされることができます。
# MAGIC 
# MAGIC <a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment" target="_blank">mlflow.set_experiment()</a>で実験を指定することができますが、実験を指定しない場合は、自動的にこのノートブックのスコープで実験を設定されることになります。

# COMMAND ----------

# MAGIC %md <i18n value="82786653-4926-4790-b867-c8ccb208b451"/>
# MAGIC 
# MAGIC ### Runの追跡 (Track Runs)
# MAGIC 
# MAGIC 各Runには以下の情報を記録することができます。<br>
# MAGIC 
# MAGIC - **Parameters：** 入力パラメータのキーと値のペア（ランダムフォレストモデルの木の数など）
# MAGIC - **評価指標：** RMSE や ROC 曲線下の面積などの評価指標
# MAGIC - **Artifact：** 任意の形式の出力ファイル 画像、モデルのpickleファイル、データファイルが含まれます。
# MAGIC - **ソース：** 実験を実行したコード
# MAGIC 
# MAGIC **注**:Sparkモデルについては、MLflowはPipelineModelsのみロギングすることができます。

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

with mlflow.start_run(run_name="LR-Single-Feature") as run:
    # Define pipeline
    vec_assembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol="price")
    pipeline = Pipeline(stages=[vec_assembler, lr])
    pipeline_model = pipeline.fit(train_df)

    # Log parameters
    mlflow.log_param("label", "price")
    mlflow.log_param("features", "bedrooms")

    # Log model
    mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas()) 

    # Evaluate predictions
    pred_df = pipeline_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
    rmse = regression_evaluator.evaluate(pred_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)

# COMMAND ----------

# MAGIC %md <i18n value="44bc7cac-de4a-47e7-bfff-6d2eb58172cd"/>
# MAGIC 
# MAGIC これで、すべて完了です。
# MAGIC 
# MAGIC ほかの2つの線形回帰モデルを作成して、実行結果を比較してみましょう。
# MAGIC 
# MAGIC **質問**：他のRunのRMSEを覚えていますか？
# MAGIC 
# MAGIC 次に、全ての特徴量を使用して線形回帰モデルを構築します。

# COMMAND ----------

from pyspark.ml.feature import RFormula

with mlflow.start_run(run_name="LR-All-Features") as run:
    # Create pipeline
    r_formula = RFormula(formula="price ~ .", featuresCol="features", labelCol="price", handleInvalid="skip")
    lr = LinearRegression(labelCol="price", featuresCol="features")
    pipeline = Pipeline(stages=[r_formula, lr])
    pipeline_model = pipeline.fit(train_df)

    # Log pipeline
    mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas())

    # Log parameter
    mlflow.log_param("label", "price")
    mlflow.log_param("features", "all_features")

    # Create predictions and metrics
    pred_df = pipeline_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
    rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)

    # Log both metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

# COMMAND ----------

# MAGIC %md <i18n value="70188282-8d26-427d-b374-954e9a058000"/>
# MAGIC 
# MAGIC 最後に、対数正規分布であるので、価格の対数を予測する線形回帰モデルを構築します。
# MAGIC 
# MAGIC また、ログの正規分布のヒストグラムを作り、アーティファクトとして記録する練習をします。

# COMMAND ----------

from pyspark.sql.functions import col, log, exp
import matplotlib.pyplot as plt

with mlflow.start_run(run_name="LR-Log-Price") as run:
    # Take log of price
    log_train_df = train_df.withColumn("log_price", log(col("price")))
    log_test_df = test_df.withColumn("log_price", log(col("price")))

    # Log parameter
    mlflow.log_param("label", "log_price")
    mlflow.log_param("features", "all_features")

    # Create pipeline
    r_formula = RFormula(formula="log_price ~ . - price", featuresCol="features", labelCol="log_price", handleInvalid="skip")  
    lr = LinearRegression(labelCol="log_price", predictionCol="log_prediction")
    pipeline = Pipeline(stages=[r_formula, lr])
    pipeline_model = pipeline.fit(log_train_df)

    # Log model
    mlflow.spark.log_model(pipeline_model, "log-model", input_example=log_train_df.limit(5).toPandas())

    # Make predictions
    pred_df = pipeline_model.transform(log_test_df)
    exp_df = pred_df.withColumn("prediction", exp(col("log_prediction")))

    # Evaluate predictions
    rmse = regression_evaluator.setMetricName("rmse").evaluate(exp_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(exp_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Log artifact
    plt.clf()

    log_train_df.toPandas().hist(column="log_price", bins=100)
    fig = plt.gcf()
    mlflow.log_figure(fig, f"{DA.username}_log_normal.png")
    plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="66785d5e-e1a7-4896-a8a9-5bfcd18acc5c"/>
# MAGIC 
# MAGIC それです！
# MAGIC 
# MAGIC では、MLflowを使ってモデルのパフォーマンスを比較してみましょう。過去のRunの照会はプログラムとMLflow UIの2つの方法があります。

# COMMAND ----------

# MAGIC %md <i18n value="0b1a68e1-bd5d-4f78-a452-90c7ebcdef39"/>
# MAGIC 
# MAGIC ### 過去のRunを照会する (Querying Past Runs)
# MAGIC 
# MAGIC このデータをPythonで利用するために、プログラム上で過去の実行結果を照会することができます。 これを行うために **`MlflowClient`** オブジェクトを使用します。

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

client.list_experiments()

# COMMAND ----------

# MAGIC %md <i18n value="dcd771b2-d4ed-4e9c-81e5-5a3f8380981f"/>
# MAGIC 
# MAGIC また、<a href="https://mlflow.org/docs/latest/search-syntax.html" target="_blank">search_runs</a> を使えば、指定した実験に対するすべてのランを検索することができます。

# COMMAND ----------

experiment_id = run.info.experiment_id
runs_df = mlflow.search_runs(experiment_id)

display(runs_df)

# COMMAND ----------

# MAGIC %md <i18n value="68990866-b084-40c1-beee-5c747a36b918"/>
# MAGIC 
# MAGIC 最後のランのメトリクスを確認します。

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
runs[0].data.metrics

# COMMAND ----------

runs[0].info.run_id

# COMMAND ----------

# MAGIC %md <i18n value="cfbbd060-6380-444f-ba88-248e10a56559"/>
# MAGIC 
# MAGIC UIで結果を確認します。 以下をご確認ください：
# MAGIC 
# MAGIC 1. **`Experiment ID`**
# MAGIC 1. アーティファクトの保存場所。 アーティファクトがDBFSに保存される場所です。
# MAGIC 1. Runが実行された時間です。**ランの詳細情報を見るには、これをクリックしてください。**
# MAGIC 1. Runを実行したコード。
# MAGIC 
# MAGIC 
# MAGIC 実行時間をクリックした後、下記をご覧ください：
# MAGIC 
# MAGIC 1. Run IDは、上記で表示したものと一致します。
# MAGIC 1. 保存したモデルには、モデルのpickleファイル、Conda環境、 **`MLmodel`** ファイルが含まれています。
# MAGIC 
# MAGIC なお、「Notes」タブでメモを追加することで、モデルに関する重要な情報を記録することができます。
# MAGIC 
# MAGIC 対数正規分布のRunをクリックすると、「アーティファクト」にヒストグラムが保存されることを確認します。

# COMMAND ----------

# MAGIC %md <i18n value="63ca7584-2a86-421b-a57e-13d48db8a75d"/>
# MAGIC 
# MAGIC ### 保存されたモデルのロード (Load Saved Model)
# MAGIC 
# MAGIC <a href="https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html" target="_blank">ロード</a>という練習をしてみましょう。

# COMMAND ----------

model_path = f"runs:/{run.info.run_id}/log-model"
loaded_model = mlflow.spark.load_model(model_path)

display(loaded_model.transform(test_df))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
