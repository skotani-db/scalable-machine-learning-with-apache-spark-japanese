# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="c7e3123c-9ace-4d9a-89a1-10307b238964"/>
# MAGIC 
# MAGIC # Feature Storeラボ (Feature Store Lab)
# MAGIC 
# MAGIC <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html" target="_blank">Databricks Feature Store</a> に慣れてきたので、以下の新しいデータセットに学習した概念を適用してみてください。
# MAGIC 
# MAGIC Feature Store Python API ドキュメントは <a href="https://docs.databricks.com/dev-tools/api/python/latest/index.html#feature-store-python-api-reference" target="_blank"> こちら</a>から参照できます。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このラボでは、次のことを行います。<br>
# MAGIC  - Feature Storeの作成 
# MAGIC  - 既存のFeature Storeのアップデート
# MAGIC  - MLflowモデルをFeature tableと一緒に登録
# MAGIC  - バッチ推論の実行

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="2861b87d-095f-44c7-848e-9e4ea539ed2d"/>
# MAGIC 
# MAGIC ### データの読込 (Load the data)
# MAGIC この例では、新しいCOVID-19データセットを使用します。以下のセルを実行し、データフレーム **`covid_df`** を作成します。

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

file_path = f"{DA.paths.datasets}/COVID/coronavirusdataset/Time.csv"
covid_df = (spark.read
            .format("csv")
            .option("header",True)
            .option("inferSchema", True)
            .load(file_path)
            .withColumn("index", monotonically_increasing_id()))

display(covid_df)

# COMMAND ----------

# MAGIC %md <i18n value="e0cab81f-4d2b-486b-9af7-01f6fb212168"/>
# MAGIC 
# MAGIC 以下のセルを実行して、ラボ用のデータベースと一意のテーブル名 **`table_name`** を設定します。

# COMMAND ----------

import uuid

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DA.cleaned_username}")
table_name = f"{DA.cleaned_username}.airbnb_{str(uuid.uuid4())[:6]}"

print(table_name)

# COMMAND ----------

# MAGIC %md <i18n value="9db69933-359c-46df-88df-523c20aec03e"/>
# MAGIC 
# MAGIC FeatureStoreClient **`fs`** をセットアップしてみましょう。
# MAGIC 
# MAGIC Feature store clientを作成するには、 **`feature_store`** モジュールから **`FeatureStoreClient`** オブジェクトを初期化します。

# COMMAND ----------

# TODO
from databricks import feature_store
 
fs = # FILL_IN

# COMMAND ----------

# MAGIC %md <i18n value="961b8810-5fce-4b34-b066-75f33a12e4f8"/>
# MAGIC 
# MAGIC ### 特徴量抽出 (Extract Features)
# MAGIC 
# MAGIC この簡単な例では、当日の他の情報を使って、毎日の死亡者数を予測します。
# MAGIC 
# MAGIC Feature tableに書き込む前に、特徴量とラベルを分けるための特徴量計算関数を書きます。
# MAGIC 
# MAGIC 以下の特徴量計算関数を、 **`deceased`** 以外の特徴量の列のみを選択するように記入してください。

# COMMAND ----------

# TODO
@feature_store.feature_table
def select_features(dataframe):
    return # FILL_IN

covid_features_df = select_features(covid_df)
display(covid_features_df)

# COMMAND ----------

# MAGIC %md <i18n value="d3892ba8-77ca-403f-869f-a6e0b0706555"/>
# MAGIC 
# MAGIC ### Feature Tableの作成 (Create Feature Table)
# MAGIC 
# MAGIC さて、特徴量の準備ができたので、以下のセルを埋めて、feature tableを作成します。
# MAGIC 
# MAGIC 名前は必ず、上記定義した **`table_name`** にしてください。
# MAGIC 
# MAGIC **NOTE:** 主キーは以下のようにリストに定義する必要があります：["主キー名"]。

# COMMAND ----------

# TODO
fs.create_table(
    name=#FILL_IN,
    primary_keys=#FILL_IN,
    df=#FILL_IN,
    schema=#FILL_IN,
    description=#FILL_IN
)

# COMMAND ----------

# MAGIC %md <i18n value="a7a6fb53-a77d-465b-b9bd-1671e3afbbd6"/>
# MAGIC 
# MAGIC ### Feature Tableの更新 (Update Feature Table)
# MAGIC 
# MAGIC 各行の日付の月と日の列を別々に追加することを想像してください。
# MAGIC 
# MAGIC これらの値を計算してテーブルを再作成するのではなく、既存のテーブルに新しいカラムを追加するだけで済みます。
# MAGIC 
# MAGIC まず、月と日のカラムを作成しましょう。

# COMMAND ----------

from pyspark.sql.functions import month, dayofmonth

add_df = (covid_features_df
  .select("date", "index")
  .withColumn("month", month("date"))
  .withColumn("day", dayofmonth("date"))
)

display(add_df)

# COMMAND ----------

# MAGIC %md <i18n value="c040ac8a-f24d-4951-82bc-af1f509a90c2"/>
# MAGIC 
# MAGIC ここで、 **`write_table`** を使用して、新しいカラムをfeature tableに追加しましょう。
# MAGIC 
# MAGIC **NOTE:** **`"overwrite"`** または **`"merge"`** のどちらのモードも使うことができます。ここではどちらを使うべきでしょうか？

# COMMAND ----------

# TODO
fs.write_table(
    name=#FILL_IN,
    df=#FILL_IN,
    mode=#FILL_IN
)

# COMMAND ----------

# MAGIC %md <i18n value="67d3eb45-0083-4a4c-a391-92ce31628b42"/>
# MAGIC 
# MAGIC ここで、 **`fs.read_table`** で **`table_name`** を指定して、更新されたfeature tableを確認してみましょう。

# COMMAND ----------

# TODO
updated_df = #FILL_IN

display(updated_df)

# COMMAND ----------

# MAGIC %md <i18n value="529e3acd-adc0-45e3-b2a7-eed04a3911b2"/>
# MAGIC 
# MAGIC ### トレーニング (Training) 
# MAGIC 
# MAGIC Feature tableができたので、それを使ってモデルの学習を行います。特徴量以外に目的変数 **`deceased`** が必要なので、まずそれを取得しましょう。

# COMMAND ----------

target_df = covid_df.select(["index", "deceased"])

display(target_df)

# COMMAND ----------

# MAGIC %md <i18n value="386a9f38-3c89-41ba-9ce2-3645aa727411"/>
# MAGIC 
# MAGIC それでは、トレーニングデータセットとテストデータセットを作成しましょう。

# COMMAND ----------

from sklearn.model_selection import train_test_split

def load_data(table_name, lookup_key):
    model_feature_lookups = [feature_store.FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fs.create_training_set(target_df, model_feature_lookups, label="deceased", exclude_columns=["index","date"])
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("deceased", axis=1)
    y = training_pd["deceased"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "index")
X_train.head()

# COMMAND ----------

# MAGIC %md <i18n value="3d1f0e1f-5ead-4f3d-81d5-d400231d0e43"/>
# MAGIC 
# MAGIC これでモデルを学習し、feature storeに登録しましょう。

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
try:
    client.delete_registered_model(f"feature_store_covid_{DA.cleaned_username}") # Deleting model if already created
except:
    None

# COMMAND ----------

import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

def train_model(table_name):
    X_train, X_test, y_train, y_test, training_set = load_data(table_name, "index")

    ## fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("r2", r2_score(y_test, y_pred))

        fs.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=f"feature_store_covid_{DA.cleaned_username}",
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )
    
train_model(table_name)

# COMMAND ----------

# MAGIC %md <i18n value="1836dc20-0e77-467e-af8f-33dcfdac7209"/>
# MAGIC 
# MAGIC これで、学習済みモデルが完成しました。Feature StoreのUIから、モデルが登録されていることを確認します。Feature tableからこのモデルがどの特徴量を使い、どれを除外したのかがわかるでしょうか。
# MAGIC 
# MAGIC 最後に、モデルを適用してみましょう。

# COMMAND ----------

## For sake of simplicity, we will just predict on the same inference_data_df
batch_input_df = target_df.drop("deceased") # Exclude true label
predictions_df = fs.score_batch(f"models:/feature_store_covid_{DA.cleaned_username}/1", 
                                  batch_input_df, result_type="double")
display(predictions_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
