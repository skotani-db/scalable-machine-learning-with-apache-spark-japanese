# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="d6718279-32b1-490e-8a38-f1d6e3578184"/>
# MAGIC 
# MAGIC # Pandas Function APIを使ったトレーニング (Training with Pandas Function API)
# MAGIC 
# MAGIC このノートブックでは、Pandas Function APIを使用して、IoTデバイスの機械学習モデルを管理およびスケーリングする方法を説明します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンで以下を行います:<br>
# MAGIC  - <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.applyInPandas.html" target="_blank"> **`.groupBy().applyInPandas()`** </a> を使用して、IoT デバイスごとに多数のモデルを並行して構築します。

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="35af29dc-0fc5-4e37-963d-3fbe86f4ba59"/>
# MAGIC 
# MAGIC ダミーデータを作成する：
# MAGIC - **`device_id`** :10個のデバイス
# MAGIC - **`record_id`** :1万件のユニークレコード
# MAGIC - **`feature_1`** : モデル学習用の特徴量
# MAGIC - **`feature_2`** : モデル学習用の特徴量
# MAGIC - **`feature_3`** : モデル学習用の特徴量
# MAGIC - **`label`** : 予測しようとする変数

# COMMAND ----------

import pyspark.sql.functions as f

df = (spark
      .range(1000*100)
      .select(f.col("id").alias("record_id"), (f.col("id")%10).alias("device_id"))
      .withColumn("feature_1", f.rand() * 1)
      .withColumn("feature_2", f.rand() * 2)
      .withColumn("feature_3", f.rand() * 3)
      .withColumn("label", (f.col("feature_1") + f.col("feature_2") + f.col("feature_3")) + f.rand())
     )

display(df)

# COMMAND ----------

# MAGIC %md <i18n value="b5f90a62-80fd-4173-adf0-6e73d0e31309"/>
# MAGIC 
# MAGIC Return schemaの定義

# COMMAND ----------

train_return_schema = "device_id integer, n_used integer, model_path string, mse float"

# COMMAND ----------

# MAGIC %md <i18n value="e2ac315f-e950-48c6-9bb8-9ceede8f93dd"/>
# MAGIC 
# MAGIC 一つのデバイスの全データを受け取り、モデルを学習し、ネストされたランとして保存して上記のスキーマでsparkオブジェクトを返すpandas関数を定義します。

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Trains an sklearn model on grouped instances
    """
    # Pull metadata
    device_id = df_pandas["device_id"].iloc[0]
    n_used = df_pandas.shape[0]
    run_id = df_pandas["run_id"].iloc[0] # Pulls run ID to do a nested run

    # Train the model
    X = df_pandas[["feature_1", "feature_2", "feature_3"]]
    y = df_pandas["label"]
    rf = RandomForestRegressor()
    rf.fit(X, y)

    # Evaluate the model
    predictions = rf.predict(X)
    mse = mean_squared_error(y, predictions) # Note we could add a train/test split

    # Resume the top-level training
    with mlflow.start_run(run_id=run_id) as outer_run:
        # Small hack for running as a job
        experiment_id = outer_run.info.experiment_id
        print(f"Current experiment_id = {experiment_id}")

        # Create a nested run for the specific device
        with mlflow.start_run(run_name=str(device_id), nested=True, experiment_id=experiment_id) as run:
            mlflow.sklearn.log_model(rf, str(device_id))
            mlflow.log_metric("mse", mse)
            mlflow.set_tag("device", str(device_id))

            artifact_uri = f"runs:/{run.info.run_id}/{device_id}"
            # Create a return pandas DataFrame that matches the schema above
            return_df = pd.DataFrame([[device_id, n_used, artifact_uri, mse]], 
                                    columns=["device_id", "n_used", "model_path", "mse"])

    return return_df

# COMMAND ----------

# MAGIC %md <i18n value="2b6bf899-de7c-4ab9-b343-a11a832ddd77"/>
# MAGIC 
# MAGIC グループ化されたデータに対してpandas関数を適用します。
# MAGIC 
# MAGIC なお、実際にどのように適用するかは、推論のためのデータがどこにあるかによって大きく異なります。この例では、デバイスとランのIDを含むトレーニングデータを再利用します。

# COMMAND ----------

with mlflow.start_run(run_name="Training session for all devices") as run:
    run_id = run.info.run_id

    model_directories_df = (df
        .withColumn("run_id", f.lit(run_id)) # Add run_id
        .groupby("device_id")
        .applyInPandas(train_model, schema=train_return_schema)
        .cache()
    )

combined_df = df.join(model_directories_df, on="device_id", how="left")
display(combined_df)

# COMMAND ----------

# MAGIC %md <i18n value="3f660cc6-4979-48dd-beea-9dab9b536230"/>
# MAGIC 
# MAGIC モデルを適用するためのpandas関数とreturn schemaを定義します。*デバイスごとに1回だけDBFSからモデルを読み込みます*。

# COMMAND ----------

apply_return_schema = "record_id integer, prediction float"

def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
    """
    Applies model to data for a particular device, represented as a pandas DataFrame
    """
    model_path = df_pandas["model_path"].iloc[0]

    input_columns = ["feature_1", "feature_2", "feature_3"]
    X = df_pandas[input_columns]

    model = mlflow.sklearn.load_model(model_path)
    prediction = model.predict(X)

    return_df = pd.DataFrame({
        "record_id": df_pandas["record_id"],
        "prediction": prediction
    })
    return return_df

prediction_df = combined_df.groupby("device_id").applyInPandas(apply_model, schema=apply_return_schema)
display(prediction_df)

# COMMAND ----------

# MAGIC %md <i18n value="d760694c-8be7-4cbb-8825-8b8aa0d740db"/>
# MAGIC 
# MAGIC ### 登録されたモデルから複数のモデルのサービング (Serving Multiple Models from a Registered Model)
# MAGIC 
# MAGIC MLflowでは、モデルをリアルタイムのREST APIとしてデプロイすることができます。現時点では、1つのMLflowモデルは1つのインスタンス（通常は1つのVM）から提供されます。しかし、1つのエンドポイントから複数のモデルを提供する必要がある場合もあります。異なる入力の1000の類似モデルをサービングすることを想像してください。特に、特定のモデルが十分に稼働されていない場合、1000個のエンドポイントを実行することは、リソースを浪費することになりかねません。
# MAGIC 
# MAGIC これを回避する一つの方法は、多くのモデルを一つのカスタムモデルにパッケージ化し、内部で入力に基づいて一つのモデルにルーティングし、そのモデルの「束」を一つの「モデル」としてデプロイすることです。
# MAGIC 
# MAGIC 以下では、各デバイスで学習させたすべてのモデルを束ねたカスタムモデルを作成する方法を紹介します。このモデルに提供されるデータの各行からモデルはデバイスIDを特定し、そのデバイスIDで学習した適切なモデルを適用して、与えられた行の予測を行います。
# MAGIC 
# MAGIC まず、各デバイスIDのモデルにアクセスする必要があります。

# COMMAND ----------

experiment_id = run.info.experiment_id

model_df = (spark.read.format("mlflow-experiment")
            .load(experiment_id)
            .filter("tags.device IS NOT NULL")
            .orderBy("end_time", ascending=False)
            .select("tags.device", "run_id")
            .limit(10))

display(model_df)

# COMMAND ----------

# MAGIC %md <i18n value="b9b38048-397b-4eb3-a7c7-541aef502d4a"/>
# MAGIC 
# MAGIC デバイスIDとそのデバイスIDで学習させたモデルをマッピングする辞書を作成します。

# COMMAND ----------

device_to_model = {row["device"]: mlflow.sklearn.load_model(f"runs:/{row['run_id']}/{row['device']}") for row in model_df.collect()}
                                                          
device_to_model

# COMMAND ----------

# MAGIC %md <i18n value="f1081d85-677f-4a55-a3f5-a7e3a6710d3a"/>
# MAGIC 
# MAGIC デバイスIDとモデルのマッピングを属性として取り、デバイスIDに基づいた適切なモデルに入力するカスタムモデルを作成します。

# COMMAND ----------

from mlflow.pyfunc import PythonModel

class OriginDelegatingModel(PythonModel):
    
    def __init__(self, device_to_model_map):
        self.device_to_model_map = device_to_model_map
        
    def predict_for_device(self, row):
        '''
        This method applies to a single row of data by
        fetching the appropriate model and generating predictions
        '''
        model = self.device_to_model_map.get(str(row["device_id"]))
        data = row[["feature_1", "feature_2", "feature_3"]].to_frame().T
        return model.predict(data)[0]
    
    def predict(self, model_input):
        return model_input.apply(self.predict_for_device, axis=1)

# COMMAND ----------

# MAGIC %md <i18n value="da424f95-113f-4feb-a20c-6d0178d03bdb"/>
# MAGIC 
# MAGIC ここでは、このモデルの使い方を紹介します。

# COMMAND ----------

example_model = OriginDelegatingModel(device_to_model)
example_model.predict(combined_df.toPandas().head(20))

# COMMAND ----------

# MAGIC %md <i18n value="624309e5-7ba8-4968-92d4-3fe71e36375b"/>
# MAGIC 
# MAGIC ここから、1つのインスタンスからすべてのデバイスIDのモデルをサービングするために使用するモデルをログに記録し、登録することができます。

# COMMAND ----------

with mlflow.start_run():
    model = OriginDelegatingModel(device_to_model)
    mlflow.pyfunc.log_model("model", python_model=model)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
