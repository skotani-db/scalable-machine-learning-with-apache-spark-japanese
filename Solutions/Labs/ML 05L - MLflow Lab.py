# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="2cf41655-1957-4aa5-a4f0-b9ec55ea213b"/>
# MAGIC 
# MAGIC # MLflow Lab
# MAGIC 
# MAGIC このラボでは、以下のステップに沿って、MLflowでモデルを本番環境に移行するまでの流れを確認します。
# MAGIC 
# MAGIC 1. Airbnbデータセットをロードし、学習データセットとテストデータセットをDeltaテーブルとして保存します。
# MAGIC 1. すべての特徴量を使ってMLlibの線形回帰モデルを学習し、パラメータ、メトリクス、デルタテーブルのバージョンをMLflowでトラッキングします。
# MAGIC 1. 初期モデルを登録し、MLflow Model Registryを使用してステージングに移行します。
# MAGIC 1. 新しいカラム **`log_price`** を学習テーブルとテストテーブルに追加し、Deltaテーブルを更新します。
# MAGIC 1. 2つ目のMLlib線形回帰モデルを学習します。今回は **`log_price`** を目的変数として、すべての特徴量を使って学習し、MLflowにトラッキングします。 
# MAGIC 1. データのバージョンを見て、２つのモデルのパフォーマンスを比較します。
# MAGIC 1. MLflowのモデルレジストリで、より性能の良いモデルを本番に移行します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このラボでは以下を行います。<br>
# MAGIC - デルタテーブルを作成する
# MAGIC - MLflow を使って MLlib モデルと Delta テーブルのバージョンを追跡する
# MAGIC - MLflowモデルレジストリを使用してモデルのバージョンを管理する

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="197ad07c-dead-4444-82de-67353d81dcb0"/>
# MAGIC 
# MAGIC ### Step 1. デルタテーブルの作成

# COMMAND ----------

# MAGIC %md <i18n value="8e9e809d-4142-43d8-b361-830099a02d06"/>
# MAGIC 
# MAGIC データのバージョニングは、Delta Lakeを使用する利点の一つで、データセットの以前のバージョンを保存して、後で復元できるようにします。
# MAGIC 
# MAGIC データセットをtrainとtestに分割し、Delta形式で書き出してみましょう。詳しくは、デルタレイクの<a href="https://docs.delta.io/latest/index.html" target="_blank">ドキュメント</a>でご覧ください。

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

train_delta_path = f"{DA.paths.working_dir}/train.delta"
test_delta_path = f"{DA.paths.working_dir}/test.delta"

# In case paths already exists
dbutils.fs.rm(train_delta_path, True)
dbutils.fs.rm(test_delta_path, True)

train_df.write.mode("overwrite").format("delta").save(train_delta_path)
test_df.write.mode("overwrite").format("delta").save(test_delta_path)

# COMMAND ----------

# MAGIC %md <i18n value="ead09bc6-c6f2-4dfa-bc9c-ddf41accc8f8"/>
# MAGIC 
# MAGIC ここで、trainとtestのDeltaテーブルを読み込み、これらのテーブルの最初のバージョンを指定しましょう。この <a href="https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html" target="_blank">ブログ記事</a> には、指定されたバージョンで Delta テーブルを読み込む例があります。

# COMMAND ----------

# ANSWER
data_version = 0
train_delta = spark.read.format("delta").option("versionAsOf", data_version).load(train_delta_path)  
test_delta = spark.read.format("delta").option("versionAsOf", data_version).load(test_delta_path)

# COMMAND ----------

# MAGIC %md <i18n value="2bf375c9-fb36-47a3-b973-82fa805e8b22"/>
# MAGIC 
# MAGIC ### Deltaテーブルの履歴の確認 (Review Delta Table History)
# MAGIC  Deltaテーブルのすべてのトランザクションは、挿入、更新、削除、マージ、および挿入の初期セットを含めて、このテーブル内に保存されます。

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY delta.`{train_delta_path}`"))

# COMMAND ----------

# MAGIC %md <i18n value="d3d7251e-c070-4d54-844f-ec6880079e5b"/>
# MAGIC 
# MAGIC デフォルトでは、Delta テーブルは <a href="https://docs.databricks.com/delta/delta-batch.html#data-retention" target="_blank">30 日間のコミット履歴を保持します</a>。この保存期間は、 **`delta.logRetentionDuration`** の設定により、どこまで過去に遡ることができるかを調整することができます。なお、設定を変更すると、ストレージのコストが上がる可能性があります。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> この方法によるDeltaでのバージョニングは、長期的なソリューションとしては実現不可能かもしれないことに注意してください。Deltaテーブルの保存期間を長くすることは可能ですが、その分、保存にかかるコストが増えます。モデルのトレーニングやMLflowでのトラッキングを行う際のデータのバージョン管理方法として、データセットのコピーをMLflowのアーティファクトとして保存する（小規模なデータセットの場合）か、または別の分散ロケーションに保存してMLflowのタグとしてそのデータセットのロケーションを記録するかといった方法もあります。

# COMMAND ----------

# MAGIC %md <i18n value="ffe159a7-e5dd-49fd-9059-e399237005a7"/>
# MAGIC 
# MAGIC ### Step 2. MLflowへの初回実行の記録
# MAGIC 
# MAGIC まず、すべての機能を使用したMLflowの実行を記録してみましょう。前回と同様のRFormulaのアプローチを採用しています。しかし今回は、データのバージョンとデータのパスをMLflowに記録することにしましょう。

# COMMAND ----------

# ANSWER
import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import RFormula

with mlflow.start_run(run_name="lr_model") as run:
    # Log parameters
    mlflow.log_param("label", "price-all-features")
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("data_path", train_delta_path)    

    # Create pipeline
    r_formula = RFormula(formula="price ~ .", featuresCol="features", labelCol="price", handleInvalid="skip")
    lr = LinearRegression(labelCol="price", featuresCol="features")
    pipeline = Pipeline(stages = [r_formula, lr])
    model = pipeline.fit(train_delta)

    # Log pipeline
    mlflow.spark.log_model(model, "model")

    # Create predictions and metrics
    pred_df = model.transform(test_delta)
    regression_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
    rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md <i18n value="3bac0fef-149d-4c25-9db1-94fbcd63ba13"/>
# MAGIC 
# MAGIC ### Step 3. MLflow Model Registryを使用してモデルを登録し、ステージングに移行します。
# MAGIC 
# MAGIC 上記モデルの性能に満足しているので、ステージングに移行します。モデルを作成し、MLflowのモデルレジストリに登録しましょう。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> **`model_uri`** へのパスが、上記のサブディレクトリ ( **`mlflow.log_model()`** の第2引数) に一致することを確認します。

# COMMAND ----------

model_name = f"{DA.cleaned_username}_mllib_lr"
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md <i18n value="78b33d27-0815-4d31-80a0-5e110aa96224"/>
# MAGIC 
# MAGIC モデルをステージングへ移行します。

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

# COMMAND ----------

# Define a utility method to wait until the model is ready
def wait_for_model(model_name, version, stage="None", status="READY", timeout=300):
    import time

    last_stage = "unknown"
    last_status = "unknown"

    for i in range(timeout):
        model_version_details = client.get_model_version(name=model_name, version=version)
        last_stage = str(model_version_details.current_stage)
        last_status = str(model_version_details.status)
        if last_status == str(status) and last_stage == str(stage):
            return

        time.sleep(1)

    raise Exception(f"The model {model_name} v{version} was not {status} after {timeout} seconds: {last_status}/{last_stage}")

# COMMAND ----------

# Force our notebook to block until the model is ready
wait_for_model(model_name, 1, stage="Staging")

# COMMAND ----------

# MAGIC %md <i18n value="b5f74e40-1806-46ab-9dd0-97b82d8f297e"/>
# MAGIC 
# MAGIC <a href="https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.update_registered_model" target="_blank">update_registered_model</a> を使用してモデルの説明を追加します。

# COMMAND ----------

# ANSWER
client.update_registered_model(
    name=model_details.name,
    description="This model forecasts Airbnb housing list prices based on various listing inputs."
)

# COMMAND ----------

wait_for_model(model_details.name, 1, stage="Staging")

# COMMAND ----------

# MAGIC %md <i18n value="03dff1c0-5c7b-473f-83ec-4a8283427280"/>
# MAGIC 
# MAGIC ### Step 4. フィーチャー・エンジニアリング：データスキーマを進化させる
# MAGIC 
# MAGIC モデルの性能を向上させるために、特徴量のエンジニアリングを行います。Delta Lakeを使って、データセットの古いバージョンを追跡することができます。
# MAGIC 
# MAGIC 新しいカラムとして **`log_price`** を追加し、Deltaテーブルを更新します。

# COMMAND ----------

from pyspark.sql.functions import col, log, exp

# Create a new log_price column for both train and test datasets
train_new = train_delta.withColumn("log_price", log(col("price")))
test_new = test_delta.withColumn("log_price", log(col("price")))

# COMMAND ----------

# MAGIC %md <i18n value="565313ed-2bca-4cc6-af87-1c0d509c0a69"/>
# MAGIC 
# MAGIC スキーマを安全に進化させるために、 **`mergeSchema`** オプションを渡して、更新されたDataFrameをそれぞれ **`train_delta_path`** と **`test_delta_path`** に保存します。
# MAGIC 
# MAGIC **`mergeSchema`** について詳しくは、Delta Lakeのこの<a href="https://databricks.com/blog/2019/09/24/diving-into-delta-lake-schema-enforcement-evolution.html" target="_blank">ブログ記事</a>をご覧ください。

# COMMAND ----------

# ANSWER
train_new.write.option("mergeSchema", "true").format("delta").mode("overwrite").save(train_delta_path)
test_new.write.option("mergeSchema", "true").format("delta").mode("overwrite").save(test_delta_path)

# COMMAND ----------

# MAGIC %md <i18n value="735a36b6-7510-4e4f-9df8-4ae51f9f87dc"/>
# MAGIC 
# MAGIC 元のスキーマと変更後のスキーマの違いを見てください。

# COMMAND ----------

set(train_new.schema.fields) ^ set(train_delta.schema.fields)

# COMMAND ----------

# MAGIC %md <i18n value="0c7c986b-1346-4ff1-a4e2-ee190891a5bf"/>
# MAGIC 
# MAGIC **`train_delta`** テーブルの Delta 履歴を確認し、最新バージョンの train および test Delta テーブルをロードしてみましょう。

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY delta.`{train_delta_path}`"))

# COMMAND ----------

data_version = 1
train_delta_new = spark.read.format("delta").option("versionAsOf", data_version).load(train_delta_path)  
test_delta_new = spark.read.format("delta").option("versionAsOf", data_version).load(test_delta_path)

# COMMAND ----------

# MAGIC %md <i18n value="f29c99ca-b92c-4f74-8bf5-c74070a8cd50"/>
# MAGIC 
# MAGIC ### Step 5. **`log_price`** を目的変数として、MLflowで実行を追跡します。
# MAGIC 
# MAGIC 更新されたデータでモデルを再トレーニングし、元のモデルの性能を比較します。その結果をMLflowに記録します。

# COMMAND ----------

with mlflow.start_run(run_name="lr_log_model") as run:
    # Log parameters
    mlflow.log_param("label", "log-price")
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("data_path", train_delta_path)    

    # Create pipeline
    r_formula = RFormula(formula="log_price ~ . - price", featuresCol="features", labelCol="log_price", handleInvalid="skip")  
    lr = LinearRegression(labelCol="log_price", predictionCol="log_prediction")
    pipeline = Pipeline(stages = [r_formula, lr])
    pipeline_model = pipeline.fit(train_delta_new)

    # Log model and update the registered model
    mlflow.spark.log_model(
        spark_model=pipeline_model,
        artifact_path="log-model",
        registered_model_name=model_name
    )  

    # Create predictions and metrics
    pred_df = pipeline_model.transform(test_delta)
    exp_df = pred_df.withColumn("prediction", exp(col("log_prediction")))
    rmse = regression_evaluator.setMetricName("rmse").evaluate(exp_df)
    r2 = regression_evaluator.setMetricName("r2").evaluate(exp_df)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)  

    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md <i18n value="e5bd7bfb-f445-44b5-a272-c6ae2849ac9f"/>
# MAGIC 
# MAGIC ### Step 6. Deltaテーブルのバージョンを見て、複数モデルのパフォーマンスを比較します。 
# MAGIC 
# MAGIC MLflowの<a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs" target="_blank"> **`mlflow.search_runs`** </a> APIを使って、学習に使ったデータのバージョンに応じた実行(Run)を特定します。データのバージョンによって、 複数の実行を比較してみましょう。
# MAGIC 
# MAGIC **`params.data_path`** と **`params.data_version`** に基づくフィルタリングを行います。

# COMMAND ----------

# ANSWER
data_version = 0

mlflow.search_runs(filter_string=f"params.data_path='{train_delta_path}' and params.data_version='{data_version}'")

# COMMAND ----------

# ANSWER
data_version = 1

mlflow.search_runs(filter_string=f"params.data_path='{train_delta_path}' and params.data_version='{data_version}'")

# COMMAND ----------

# MAGIC %md <i18n value="fd0fc3ae-7c2e-4d7d-90da-0b6e6b830496"/>
# MAGIC 
# MAGIC どのバージョンのデータで最適なモデルを構築されましたか？

# COMMAND ----------

# MAGIC %md <i18n value="3056bfcc-7623-4410-8b1b-82cba24ae3dd"/>
# MAGIC 
# MAGIC ###  Step 7. MLflowのモデルレジストリを使用して、最もパフォーマンスの高いモデルを本番環境に移行します。
# MAGIC 
# MAGIC 最新モデルのバージョンを取得し、本番に移行します。

# COMMAND ----------

model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=new_model_version,
    description="This model version was built using a MLlib Linear Regression model with all features and log_price as predictor."
)

# COMMAND ----------

model_version_details = client.get_model_version(name=model_name, version=new_model_version)
model_version_details.status

# COMMAND ----------

wait_for_model(model_name, new_model_version)

# COMMAND ----------

# ANSWER
# Move Model into Production
client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Production"
)

# COMMAND ----------

wait_for_model(model_name, new_model_version, "Production")

# COMMAND ----------

# MAGIC %md <i18n value="102094e8-1aa0-4448-9cc4-5e5e36fb5426"/>
# MAGIC 
# MAGIC MLflowのモデルレジストリUIを見て、モデルが登録されていることを確認します。モデルのバージョン1がステージングに、バージョン2がプロダクションになっていることが確認できるはずです。

# COMMAND ----------

# MAGIC %md <i18n value="f74c46fc-b825-4d73-b41f-c45e6cd360fb"/>
# MAGIC 
# MAGIC このラボを終了するために、二つのモデルのバージョンをアーカイブし、レジストリからモデル全体を削除することでクリーンアップしましょう。

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Archived"
)

# COMMAND ----------

wait_for_model(model_name, 1, "Archived")

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Archived"
)

# COMMAND ----------

wait_for_model(model_name, 2, "Archived")

# COMMAND ----------

client.delete_registered_model(model_name)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
