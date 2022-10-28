# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="2b5dc285-0d50-4ea7-a71b-8a7aa355ad7c"/>
# MAGIC 
# MAGIC # Pandas UDFを使った推論 (Inference with Pandas UDFs)
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは次を行います:<br>
# MAGIC - scikit-learnのモデルを構築し、MLflowで追跡そしてPandas Scalar Iterator UDFsと **`mapInPandas()`** を使って大規模に適用します。
# MAGIC 
# MAGIC Pandas UDFについて詳しく知りたい方は、こちらの<a href="https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html" target="_blank">ブログ記事</a>でSpark 3.0の新機能を参照してください。

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="8b52bca0-45f0-4ada-be31-c2c473fb8e77"/>
# MAGIC 
# MAGIC sklearnのモデルを学習し、MLflowで記録します。

# COMMAND ----------

import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

with mlflow.start_run(run_name="sklearn-random-forest") as run:
    # Enable autologging 
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, log_models=True)
    # Import the data
    df = pd.read_csv(f"{DA.paths.datasets}/airbnb/sf-listings/airbnb-cleaned-mlflow.csv".replace("dbfs:/", "/dbfs/")).drop(["zipcode"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

    # Create model
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md <i18n value="7ebcaaf9-c6f5-4c92-865a-c7f2c7afb555"/>
# MAGIC 
# MAGIC Spark DataFrameの作成

# COMMAND ----------

spark_df = spark.createDataFrame(X_test)

# COMMAND ----------

# MAGIC %md <i18n value="1cdc4475-f55f-4126-9d38-dedb19577f4e"/>
# MAGIC 
# MAGIC ### Pandas/ベクトル化されたUDF (Pandas/Vectorized UDFs)
# MAGIC 
# MAGIC Spark 2.3からは、Pythonで利用できるPandas UDFがあり、UDFの効率を向上させることができます。PandasのUDFは、Apache Arrowを利用して計算を高速化します。それが処理時間の改善にどう役立つかを見てみましょう。
# MAGIC 
# MAGIC * <a href="https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html" target="_blank">ブログ記事</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow" target="_blank">ドキュメンテーション</a>
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2017/10/image1-4.png" alt="Benchmark" width ="500" height="1500">
# MAGIC 
# MAGIC ユーザー定義関数が実行されます。
# MAGIC * <a href="https://arrow.apache.org/" target="_blank">Apache Arrow</a>は、Sparkで使用されてJVM と Python プロセス間のデータをほぼゼロの（デ）シリアライズコストで効率的に転送するためのインメモリ列型データ形式です。詳しくは<a href="https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html" target="_blank">こちら</a>をご覧ください。
# MAGIC * pandasのインスタンスおよびAPIと連携するため、関数内部でpandasを使用します。
# MAGIC 
# MAGIC **注**:Spark 3.0では、Pythonのタイプヒントを使用してPandas UDFを定義する必要があります。

# COMMAND ----------

from pyspark.sql.functions import pandas_udf

@pandas_udf("double")
def predict(*args: pd.Series) -> pd.Series:
    model_path = f"runs:/{run.info.run_id}/model" 
    model = mlflow.sklearn.load_model(model_path) # Load model
    pdf = pd.concat(args, axis=1)
    return pd.Series(model.predict(pdf))

prediction_df = spark_df.withColumn("prediction", predict(*spark_df.columns))
display(prediction_df)

# COMMAND ----------

# MAGIC %md <i18n value="e97526c6-ef40-4d55-9763-ee3ebe846096"/>
# MAGIC 
# MAGIC ### Pandas Scalar Iterator UDF
# MAGIC 
# MAGIC モデルが非常に大きい場合、同じPythonワーカープロセスでバッチごとに同じモデルを繰り返しロードすることは、Pandas UDFにとって高いオーバーヘッドとなります。Spark 3.0では、Pandas UDFはpandas.Seriesまたはpandas.DataFrameのiteratorを受け取ることができるので、iterator内のシリーズごとにモデルを読み込むのではなく、一度だけモデルを読み込むことで済みます。
# MAGIC 
# MAGIC そうすれば、必要なセットアップのコストが発生する回数も少なくなります。扱うレコード数が **`spark.conf.get('spark.sql.execution.arrow.maxRecordsPerBatch')`** (デフォルトは 10,000) より多い場合、pandas scalar UDFはpd.Seriesのバッチを反復処理するので、スピードアップが見られるはずです。
# MAGIC 
# MAGIC 一般的な構文：
# MAGIC 
# MAGIC ```
# MAGIC @pandas_udf(...)
# MAGIC def predict(iterator):
# MAGIC     model = ... # load model
# MAGIC     for features in iterator:
# MAGIC         yield model.predict(features)
# MAGIC ```

# COMMAND ----------

from typing import Iterator, Tuple

@pandas_udf("double")
def predict(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
    model_path = f"runs:/{run.info.run_id}/model" 
    model = mlflow.sklearn.load_model(model_path) # Load model
    for features in iterator:
        pdf = pd.concat(features, axis=1)
        yield pd.Series(model.predict(pdf))

prediction_df = spark_df.withColumn("prediction", predict(*spark_df.columns))
display(prediction_df)

# COMMAND ----------

# MAGIC %md <i18n value="23b8296e-e0bc-481e-bd35-4048d532c71d"/>
# MAGIC 
# MAGIC ### Pandas Function API
# MAGIC 
# MAGIC Pandas UDFを使う代わりに、Pandas Function APIを使うことができます。Apache Spark 3.0のこの新しい機能では、PySpark DataFrameに対してPandasインスタンスを取得・出力するPythonネイティブ関数を直接適用することができるようになりました。Apache Spark 3.0でサポートされるPandas Functions APIは、grouped map、mapとco-grouped mapです。
# MAGIC 
# MAGIC **`mapInPandas()`** は pandas.DataFrame のiteratorを入力とし、別の pandas.DataFrame のiteratorを出力する。モデルが入力として全てのカラムを必要とする場合、柔軟で使いやすいですが、DataFrame全体のシリアライズ/デシリアライズが必要です（入力として渡されるため）。iteratorが出力する各pandas.DataFrameのバッチサイズは、 **`spark.sql.execution.arrow.maxRecordsPerBatch`** の設定により制御できます。

# COMMAND ----------

def predict(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    model_path = f"runs:/{run.info.run_id}/model" 
    model = mlflow.sklearn.load_model(model_path) # Load model
    for features in iterator:
        yield pd.concat([features, pd.Series(model.predict(features), name="prediction")], axis=1)
    
display(spark_df.mapInPandas(predict, """`host_total_listings_count` DOUBLE,`neighbourhood_cleansed` BIGINT,`latitude` DOUBLE,`longitude` DOUBLE,`property_type` BIGINT,`room_type` BIGINT,`accommodates` DOUBLE,`bathrooms` DOUBLE,`bedrooms` DOUBLE,`beds` DOUBLE,`bed_type` BIGINT,`minimum_nights` DOUBLE,`number_of_reviews` DOUBLE,`review_scores_rating` DOUBLE,`review_scores_accuracy` DOUBLE,`review_scores_cleanliness` DOUBLE,`review_scores_checkin` DOUBLE,`review_scores_communication` DOUBLE,`review_scores_location` DOUBLE,`review_scores_value` DOUBLE, `prediction` DOUBLE"""))

# COMMAND ----------

# MAGIC %md <i18n value="d13b87a7-0625-4acc-88dc-438cf06e18bd"/>
# MAGIC 
# MAGIC あるいは、以下のようなスキーマを定義することもできます。

# COMMAND ----------

from pyspark.sql.functions import lit
from pyspark.sql.types import DoubleType

schema = spark_df.withColumn("prediction", lit(None).cast(DoubleType())).schema
display(spark_df.mapInPandas(predict, schema))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
