# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="b69335d5-86c7-40c5-b430-509a7444dae7"/>
# MAGIC 
# MAGIC # Feature Store
# MAGIC 
# MAGIC Databricksの <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html" target="_blank">Feature Store</a> は、特徴量を一元的に管理するリポジトリです。組織全体で特徴量を共有と発見を可能にし、またモデルの学習と推論に同じ特徴量計算のコードを使用することを保証します。
# MAGIC 
# MAGIC Feature Store Python API ドキュメントは <a href="https://docs.databricks.com/dev-tools/api/python/latest/index.html#feature-store-python-api-reference" target="_blank">こちら</a> をご確認ください。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このレッスンでは、以下を行います:<br>
# MAGIC  - Databricks Feature StoreでFeature Storeを構築する
# MAGIC  - 特徴量テーブルを更新する
# MAGIC  - バッチスコアリングを実行する

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, lit, expr, rand
import uuid
from databricks import feature_store
from pyspark.sql.types import StringType, DoubleType
from databricks.feature_store import feature_table, FeatureLookup
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# COMMAND ----------

# MAGIC %md <i18n value="5dcd3e8e-2553-429f-bbe1-aef0bc1ef0ab"/>
# MAGIC 
# MAGIC データをロードして、各レコードに一意のIDを生成してみましょう。 **`index`** カラムは、特徴テーブルの「キー」として機能し、特徴量をlookupするために使用されます。

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path).coalesce(1).withColumn("index", monotonically_increasing_id())
display(airbnb_df)

# COMMAND ----------

# MAGIC %md <i18n value="a04b29f6-e7a6-4e6a-875f-945edf938e9e"/>
# MAGIC 
# MAGIC 新しいデータベースと一意のテーブル名を作成します（ノートブックを何度も再実行する場合に備えて、実行の度にUUIDで異なる値を設定します）。

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DA.cleaned_username}")
table_name = f"{DA.cleaned_username}.airbnb_" + str(uuid.uuid4())[:6]
print(table_name)

# COMMAND ----------

# MAGIC %md <i18n value="a0712a39-b413-490f-a59e-dbd7f533e9a9"/>
# MAGIC 
# MAGIC それでは、 <a href="https://docs.databricks.com/applications/machine-learning/feature-store.html#create-a-feature-table-in-databricks-feature-store" target="_blank">Feature Store Client</a> を作成して、Feature Storeに情報を入力してみましょう。

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
# help(fs.create_table)

# COMMAND ----------

# MAGIC %md <i18n value="90998fdb-87ed-4cdd-8844-fbd59ac5631f"/>
# MAGIC 
# MAGIC #### 特徴量テーブルの作成 (Create Feature Table)
# MAGIC 
# MAGIC 次に、 **`fs.create_table`** メソッドを使って特徴量テーブルを作成します。
# MAGIC 
# MAGIC このメソッドはいくつかのパラメータを入力として受け取ります: 
# MAGIC * **`name`** - 次の形式の特徴量テーブル名 **`<データベース名>.<テーブル名>.`**
# MAGIC * **`primary_keys`** - プライマリーキーとなるカラム名(複数)。複数のカラムが必要な場合は、カラム名のリストを指定する。
# MAGIC * **`df`** - この特徴量テーブルに挿入するデータ。指定した **`df`** のスキーマが特徴量テーブルのスキーマとして使用される。
# MAGIC * **`schema`** - 特徴量テーブルのスキーマ。スキーマを指定するために **`schema`** または **`df`** のどちらかを指定する必要があります。
# MAGIC * **`description`** - 特徴量テーブルの説明 
# MAGIC * **`partition_columns`**- 特徴テーブルをパーティション分割する際に使用する列。

# COMMAND ----------

## select numeric features and exclude target column "price"
numeric_cols = [x.name for x in airbnb_df.schema.fields if (x.dataType == DoubleType()) and (x.name != "price")]
numeric_features_df = airbnb_df.select(["index"] + numeric_cols)
display(numeric_features_df)

# COMMAND ----------

fs.create_table(
    name=table_name,
    primary_keys=["index"],
    df=numeric_features_df,
    schema=numeric_features_df.schema,
    description="Numeric features of airbnb data"
)

# COMMAND ----------

# MAGIC %md <i18n value="4a7cbb2e-87a2-4ea8-85e6-207ec5e42147"/>
# MAGIC 
# MAGIC テーブルの作成とデータ登録を別メソッドで実行することも出来ます。 **`fs.create_table`** ではschemaを指定し（ **`df`** は与えない)、 **`fs.write_table`** でdfを指定してデータを登録します。 **`fs.write_table`** は **`overwrite`** と **`merge`** の2つのモードをサポートしています。
# MAGIC 
# MAGIC 例
# MAGIC 
# MAGIC ```
# MAGIC fs.create_table(
# MAGIC     name=table_name,
# MAGIC     primary_keys=["index"],
# MAGIC     schema=numeric_features_df.schema,
# MAGIC     description="Original Airbnb data"
# MAGIC )
# MAGIC 
# MAGIC fs.write_table(
# MAGIC     name=table_name,
# MAGIC     df=numeric_features_df,
# MAGIC     mode="overwrite"
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md <i18n value="44586907-302a-4916-93f6-e92210619c6f"/>
# MAGIC 
# MAGIC それではUIを使って、FeatureStoreが特徴量テーブルをどのように追跡するか見てみましょう。UIに移動するには、まず機械学習ワークスペースにいることを確認します。次に、ナビゲーションバーの左下にあるFeature Storeのアイコンをクリックします。
# MAGIC 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/FS_Nav.png" alt="step12" width="150"/>

# COMMAND ----------

# MAGIC %md <i18n value="cf0ad0d0-8456-471b-935c-8a34a836fca7"/>
# MAGIC 
# MAGIC このスクリーンショットでは、作成した特徴量テーブルを見ることができます。
# MAGIC <br>
# MAGIC <br>
# MAGIC 下の方の **`Producers`** の部分に注目してください。どのノートブックで特徴量テーブルが作成されたかを示しています。
# MAGIC <br>
# MAGIC <br>
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/mlflow/fs_details+(1).png" alt="step12" width="1000"/>

# COMMAND ----------

# MAGIC %md <i18n value="b07da702-485e-44b8-bd00-f0330c8b7657"/>
# MAGIC 
# MAGIC プログラムから FeatureStore Client を使ってFeatureStoreのメタデータを見るには **`get_table()`** を使います。

# COMMAND ----------

fs.get_table(table_name).path_data_sources

# COMMAND ----------

fs.get_table(table_name).description

# COMMAND ----------

# MAGIC %md <i18n value="1df7795c-1a07-47ae-92a8-1c5f7aec75ae"/>
# MAGIC 
# MAGIC ### Feature Storeを用いたモデルの学習 (Train a model with feature store)

# COMMAND ----------

# MAGIC %md <i18n value="bcbf72b7-a013-40fd-bf55-a2b179a7728e"/>
# MAGIC 
# MAGIC 予測対象である **`price`** は、特徴量テーブルの特徴量として登録されているべきではありません(should NOT)。
# MAGIC 
# MAGIC さらに、推論時に使用する特徴量は、特徴量テーブルに既に登録されるものに限らなりません。 
# MAGIC 
# MAGIC この（架空の）例では、ある特徴量を作りました : **`score_diff_from_last_month`** 。推論時に生成され、学習時にも使用される特徴量です。

# COMMAND ----------

## inference data -- index (key), price (target) and a online feature (make up a fictional column - diff of review score in a month) 
inference_data_df = airbnb_df.select("index", "price", (rand() * 0.5-0.25).alias("score_diff_from_last_month"))
display(inference_data_df)

# COMMAND ----------

# MAGIC %md <i18n value="b8301fa9-27bd-4d3b-bf13-9ab784205d81"/>
# MAGIC 
# MAGIC 学習用データセットを作成します。このデータセットには指定された"key (特徴量テーブルのカラム名)"を使って、特徴量テーブルからlookupして得た特徴量とオンライン特徴量( **`score_diff_from_last_month`** )を使用します。特徴量を検索するために <a href="https://docs.databricks.com/dev-tools/api/python/latest/index.html" target="_blank">FeatureLookup</a> を使用しますが、特徴量を指定しない場合は、主キー以外のすべての特徴量を返します。

# COMMAND ----------

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="price", exclude_columns="index")
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("price", axis=1)
    y = training_pd["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "index")
X_train.head()

# COMMAND ----------

# MAGIC %md <i18n value="eae1aa4a-f770-4173-9502-cb946e6949d2"/>
# MAGIC 
# MAGIC **RandomForestRegressor** モデルを学習し、Feature StoreでモデルをMLflowに記録します。MLflowのrunにより、MLflowで自動記録されたコンポーネントと共に、Feature Storeで記録されたモデルを追跡します。以下では、Feature Storeで明示的にモデルを記録するため、MLflowオートロギングのモデルの記録を無効化します。
# MAGIC 
# MAGIC 注：以下はデモのために、過度に単純化した例です。

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()

try:
    client.delete_registered_model(f"feature_store_airbnb_{DA.cleaned_username}") # Deleting model if already created
except:
    None

# COMMAND ----------

# Disable model autologging and instead log explicitly via the FeatureStore
mlflow.sklearn.autolog(log_models=False)

def train_model(X_train, X_test, y_train, y_test, training_set, fs):
    ## fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fs.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=f"feature_store_airbnb_{DA.cleaned_username}",
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )

train_model(X_train, X_test, y_train, y_test, training_set, fs)

# COMMAND ----------

# MAGIC %md <i18n value="40b7718f-101c-4ac4-8639-545b8ef6d932"/>
# MAGIC 
# MAGIC ここで、MLflow UIからrunを確認します。MLflow autologでログに記録されたモデルのパラメータを確認することができます。
# MAGIC <br>
# MAGIC <br>
# MAGIC <img src="https://files.training.databricks.com/images/mlflow/fs_log_model_mlflow_params.png" alt="step12" width="1000"/>

# COMMAND ----------

# MAGIC %md <i18n value="f03314dc-1ade-4bd8-958f-ddf04ac1bb13"/>
# MAGIC 
# MAGIC 保存されたartifactの中のモデル**feature\_store\_model**は、 **`fs.log_model`** でパッケージ化されて記録されたFeature storeモデルであり、バッチ推論に直接使用できます。
# MAGIC <br>
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/301/updated_feature_store_9_1.png" alt="step12" width="1000"/>

# COMMAND ----------

# MAGIC %md <i18n value="acd4d5a4-c4ed-4695-a911-5fd88dcfa513"/>
# MAGIC 
# MAGIC この **`feature_store_model`** は、MLflowのモデルレジストリにも登録されています。 **`Models`** のページをご覧ください。また、Feature Storeのページでも記録されており、特徴量テーブルのどの特徴量がモデルに使用されているかが記録されます。後でUIを使って特徴量とモデルの対応関係を確認してみましょう。

# COMMAND ----------

# MAGIC %md <i18n value="921dc6c9-b9ed-43c7-86ff-608791a11367"/>
# MAGIC 
# MAGIC ### Feature storeでバッチ推論 (Feature Store Batch Scoring)
# MAGIC 
# MAGIC **`score_batch`** でFeature Storeに登録したMLflowモデルをデータに適用しましょう。入力データには、キーカラムである **`index`** とオンライン特徴量である **`score_diff_from_last_month`** のみを使います。それ以外の特徴量はすべてlookupで自動的に取得してくれます。

# COMMAND ----------

## For sake of simplicity, we will just predict on the same inference_data_df
batch_input_df = inference_data_df.drop("price") # Exclude true label
predictions_df = fs.score_batch(f"models:/feature_store_airbnb_{DA.cleaned_username}/1", 
                                  batch_input_df, result_type="double")
display(predictions_df)

# COMMAND ----------

# MAGIC %md <i18n value="fa42d4d3-a6a6-4205-b799-032154d1d8a3"/>
# MAGIC 
# MAGIC ### 特徴量テーブルを上書きする (Overwrite feature table)
# MAGIC 最後に、いくつかのレビューカラムの要約情報を追加して特徴量テーブルを更新します。新しい特徴量カラムは物件の平均レビュースコアを計算して作成します。

# COMMAND ----------

## select numeric features and aggregate the review scores
review_columns = ["review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", 
                 "review_scores_communication", "review_scores_location", "review_scores_value"]

condensed_review_df = (airbnb_df
                       .select(["index"] + numeric_cols)
                       .withColumn("average_review_score", expr("+".join(review_columns)) / lit(len(review_columns)))
                       .drop(*review_columns)
                      )
             
display(condensed_review_df)

# COMMAND ----------

# MAGIC %md <i18n value="da3ee1df-391c-4f26-99d0-82937e91a40a"/>
# MAGIC 
# MAGIC では、 **`overwrite(上書き)`** 新規特徴量の追加と元々の特徴量を削除します。

# COMMAND ----------

fs.write_table(
    name=table_name,
    df=condensed_review_df,
    mode="overwrite"
)

# COMMAND ----------

# MAGIC %md <i18n value="ae45b580-e79e-4f54-85a0-1274cb5f5c5f"/>
# MAGIC 
# MAGIC ### Feature Store UIから特徴量のpermission(アクセス許可)、lineage(特徴量とモデルの対応関係)、freshness(データの鮮度)を探る

# COMMAND ----------

# MAGIC %md <i18n value="5d4d8425-b9b7-4e47-8856-91e1142e9c47"/>
# MAGIC 
# MAGIC UI上では、以下のことが確認できます。
# MAGIC * 特徴量リストに新しいカラムが追加されたこと。
# MAGIC * 削除したカラムはまだ存在していること。ただし、削除された特徴量は、テーブルを読み込む際に **`null`** になります。
# MAGIC * "Models"カラムが作られ、該当特徴量を使用したモデルがリストアップされています。
# MAGIC * **`Notebooks`** カラムが作られ、該当特徴量を使用したノートブックが表示されています。このカラムは、どのノートブックが該当特徴量を消費しているかを示しています。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/feature_store_consumers.png" alt="step12" width="800"/>

# COMMAND ----------

# MAGIC %md <i18n value="884ff3ff-f965-4c37-8cff-f6a1600ee0b6"/>
# MAGIC 
# MAGIC では、Feature Storeから特徴量データを読み込んでみましょう。デフォルトでは **`fs.read_table()`** は最新版の特徴量テーブルを読み込みます。特徴量テーブルの特定バージョンを読み込むために、オプションで引数 **`as_of_delta_timestamp`** で日時を指定します。形式は、タイムスタンプか文字列です。
# MAGIC 
# MAGIC 
# MAGIC 削除されたカラムの値は **`null`** に置き換えられていることに注意してください。

# COMMAND ----------

# Displays most recent table
display(fs.read_table(name=table_name))

# COMMAND ----------

# MAGIC %md <i18n value="4148328d-4046-4251-b4db-f9e427b2e0f9"/>
# MAGIC 
# MAGIC 特徴量をリアルタイム・サービングすることが必要な場合は、特徴量を<a href="https://docs.databricks.com/applications/machine-learning/feature-store.html#publish-features-to-an-online-feature-store" target="_blank">Online store</a>にPublishします。
# MAGIC 
# MAGIC UI上で特徴テーブルを扱う権限を設定することができます。
# MAGIC 
# MAGIC テーブルを削除するには **`delete`** ボタンをUIでクリックします。**データベースからdeltaテーブルも削除する必要があります。** <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/mlflow/fs_permissions+(1).png" alt="ステップ12" width="700"/>

# COMMAND ----------

# MAGIC %md <i18n value="81e53dea-dc51-418c-b366-eed3a9c4ce2f"/>
# MAGIC 
# MAGIC ### 追加した特徴量(average_review_score)を用いたモデルの再学習 (Retrain a new model with the new average_review_score feature)

# COMMAND ----------

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="price", exclude_columns="index")
    training_pd = training_set.load_df().drop(*review_columns).toPandas()  #remove all those null columns, should now have the new average_review_score in it

    # Create train and test datasets
    X = training_pd.drop("price", axis=1)
    y = training_pd["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, training_set

X_train, X_test, y_train, y_test, training_set = load_data(table_name, "index")
X_train.head()

# COMMAND ----------

# MAGIC %md <i18n value="94873d7f-3bb9-4d5f-a414-c24480a84f3b"/>
# MAGIC 
# MAGIC 指定された`key`で特徴量を検索して学習用データセットを作成します。

# COMMAND ----------

def train_model(X_train, X_test, y_train, y_test, training_set, fs):
    ## fit and log model
    with mlflow.start_run() as run:

        rf = RandomForestRegressor(max_depth=3, n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))

        fs.log_model(
            model=rf,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn,
            training_set=training_set,
            registered_model_name=f"feature_store_airbnb_{DA.cleaned_username}",
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )

train_model(X_train, X_test, y_train, y_test, training_set, fs)

# COMMAND ----------

# MAGIC %md <i18n value="b0ffd91d-c73f-4f86-a02a-43ffdc73460c"/>
# MAGIC 
# MAGIC ### Feature Storeでバッチスコアリング (Feature Score Batch Scoring)
# MAGIC 
# MAGIC FeatueStoreに登録されたMLflowモデルversion2に、特徴量を与えて **`score_batch`** を行います。

# COMMAND ----------

## For sake of simplicity, we will just predict on the same inference_data_df
batch_input_df = inference_data_df.drop("price") # Exclude true label
predictions_df = fs.score_batch(f"models:/feature_store_airbnb_{DA.cleaned_username}/2", #notice we are using version2
                                  batch_input_df, result_type="double")
display(predictions_df)

# COMMAND ----------

# MAGIC %md <i18n value="67471f1c-0dc0-445f-ae6a-beafb3508a16"/>
# MAGIC 
# MAGIC UI上では、以下のことが確認できます。
# MAGIC * モデルバージョン2は、新しく作成されたaverage_review_score(平均レビュースコア)という特徴量を使用しています。
# MAGIC * 削除したカラムもまだ存在しています。ただし、削除された特徴量は、テーブルで読み込むと **`null`** の値になります。
# MAGIC * "Models"カラムには、当該特徴量を用いたモデルのバージョンがリストアップされます。
# MAGIC * 最後に **`Notebooks`** カラムが表示されます。このカラムは、どのノートブックが該当特徴量を消費するかを示しています。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/feature_store_consumers_2.png" alt="step12" width="1000"/>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
