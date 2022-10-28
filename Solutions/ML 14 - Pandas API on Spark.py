# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="c311be95-77f9-477b-93a5-c9289b3dedb6"/>
# MAGIC 
# MAGIC # SparkにおけるPandas API (Pandas API on Spark)
# MAGIC 
# MAGIC pandas API on Sparkプロジェクトは、Apache Sparkにおけるpandas DataFrame APIを実装することにより、データサイエンティストがビッグデータを扱う際の生産性を高めることを目的としています。2つのエコシステムを使い慣れたAPIに統一することで、pandas API on Sparkは小規模データと大規模データの間のシームレスな切り替えを提供します。
# MAGIC 
# MAGIC PySpark 3.2にマージされた<a href="https://github.com/databricks/koalas" target="_blank">Koalas</a>プロジェクトについてご存知の方もいるかもしれません。Apache Spark 3.2以降では、スタンドアローンのKoalasプロジェクトがメンテナンスモードに入ったため、PySparkを直接ご利用ください。こちらの<a href="https://databricks.com/blog/2021/10/04/pandas-api-on-upcoming-apache-spark-3-2.html" target="_blank">ブログ記事</a>をご覧ください。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは次を行います:<br>
# MAGIC - Sparkのpandas APIと普通のpandas APIとの類似性を確認します。
# MAGIC - SparkとPySparkのpandas APIで同じDataFrameの操作を行う場合の構文の違いを理解します。

# COMMAND ----------

# MAGIC %md <i18n value="d711990a-af32-4357-b710-d2db434e4f15"/>
# MAGIC 
# MAGIC <div style="img align: center; line-height:0; padding-top:9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/301/31gb.png" width="900"/>
# MAGIC </div>
# MAGIC 
# MAGIC <div style="img align: center; line-height:0; padding-top:9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/301/95gb.png" width="900"/>
# MAGIC </div>
# MAGIC 
# MAGIC **Pandas** DataFrames は ミュータブル(変更可能) で、強制的に評価され、行の順序を維持します。これらは1台のマシンに限定され、a)に示すようにデータセットが小さい場合に非常に高い性能を出します。
# MAGIC 
# MAGIC **Spark** DataFrameは、分散的、遅延評価され、不変であり、行の順序を維持しません。b)とc)に示したように、大規模データの場合の性能は非常に高いです。
# MAGIC 
# MAGIC **pandas API on Spark**は、pandas APIとSparkの性能上の利点という、両者の利点を提供します。しかし、Sparkでネイティブにソリューションを実装するのに比べれば、速度は劣ります。以下にその理由を説明します。

# COMMAND ----------

# MAGIC %md <i18n value="c3080510-c8d9-4020-9910-37199f0ad5de"/>
# MAGIC 
# MAGIC ## InternalFrame
# MAGIC 
# MAGIC InternalFrameは、現在のSpark DataFrameと内部の不変のメタデータを保持します。
# MAGIC 
# MAGIC Pandas API on Sparkのカラム名からSparkのカラム名へのマッピング、およびpandas API on Sparkのインデックス名からSparkのカラム名へのマッピングを管理します。
# MAGIC 
# MAGIC ユーザが何らかのAPIを呼び出すと、Sparkのpandas API DataFrameがInternalFrameのSpark DataFrameとメタデータを更新します。現在のInternalFrameを新しい状態で作成またはコピーし、新しいpandas API on Spark DataFrameを返します。
# MAGIC 
# MAGIC <div style="img align: center; line-height:0; padding-top:9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/301/InternalFramePs.png" width="900"/>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="785ed714-6726-40d5-b7fb-c63c094e568e"/>
# MAGIC 
# MAGIC ## InternalFrameメタデータのみ更新 (InternalFrame Metadata Updates Only)
# MAGIC 
# MAGIC Spark DataFrameではなく、メタデータのみを更新する場合は、以下のような新しい構造になります。
# MAGIC 
# MAGIC <div style="img align: center; line-height:0; padding-top:9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/301/InternalFrameMetadataPs.png" width="900"/>.
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="e6d7a47f-a4c8-4178-bc70-62c2ac6764d5"/>
# MAGIC 
# MAGIC ## InternalFrame inplace更新 (InternalFrame Inplace Updates)
# MAGIC 
# MAGIC 一方、pandas API on Spark DataFrameは、新しいDataFrameを返すのではなく内部の状態を更新することがあります。例えば、引数inplace=Trueを与えると、新しい構造は以下のようになります。
# MAGIC 
# MAGIC <div style="img align: center; line-height:0; padding-top:9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/301/InternalFrameUpdate.png" width="900"/>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="23a2fc6d-1360-4e41-beab-b1fe8e23aac3"/>
# MAGIC 
# MAGIC ### データセットの読み込み (Read in the dataset)
# MAGIC 
# MAGIC * PySpark
# MAGIC * Pandas
# MAGIC * Pandas API on Spark

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="1be64dea-9d63-476d-a7d6-9f6fa4ccd784"/>
# MAGIC 
# MAGIC PySparkでParquetを読み込みます。

# COMMAND ----------

spark_df = spark.read.parquet(f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")
display(spark_df)

# COMMAND ----------

# MAGIC %md <i18n value="00b99bdc-e4d1-44d2-b117-ae2cd97d0490"/>
# MAGIC 
# MAGIC PandasでをParquetを読み込みます。

# COMMAND ----------

import pandas as pd

pandas_df = pd.read_parquet(f"{DA.paths.datasets.replace('dbfs:/', '/dbfs/')}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")
pandas_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="e75a3ba6-98f6-4b39-aecb-345109cb2ce9"/>
# MAGIC 
# MAGIC Pandas API on SparkでParquetを読み込みます。Pandas API on Sparkが、pandasのようにインデックスカラムを作成します。
# MAGIC 
# MAGIC Pandas API on Spark はDeltaからの読み込み(**`read_delta`**)もサポートしていますが、pandasはまだサポートしていません。

# COMMAND ----------

import pyspark.pandas as ps

df = ps.read_parquet(f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")
df.head()

# COMMAND ----------

# MAGIC %md <i18n value="f099c73b-0bd8-4ff1-a12e-578ffb0cb152"/>
# MAGIC 
# MAGIC ### <a href="https://koalas.readthedocs.io/en/latest/user_guide/options.html#default-index-type" target="_blank">インデックスの種類</a>
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/301/koalas_index.png)

# COMMAND ----------

ps.set_option("compute.default_index_type", "distributed-sequence")
df_dist_sequence = ps.read_parquet(f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/")
df_dist_sequence.head()

# COMMAND ----------

# MAGIC %md <i18n value="07b3f029-f81b-442f-8cdd-cb2d29033a35"/>
# MAGIC 
# MAGIC ### Spark DataFrameとpandas API on Spark DataFrameの変換 (Converting to pandas API on Spark DataFrame to/from Spark DataFrame)

# COMMAND ----------

# MAGIC %md <i18n value="ed25204e-2822-4694-b3b3-968ea8ef7343"/>
# MAGIC 
# MAGIC PySpark DataFrameからpandas API on Spark DataFrame を作成します。

# COMMAND ----------

df = ps.DataFrame(spark_df)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="a41480c7-1787-4bd6-a4c3-c85552a5f762"/>
# MAGIC 
# MAGIC PySpark DataFrameからpandas API on Spark DataFrameを作成する代替手段

# COMMAND ----------

df = spark_df.to_pandas_on_spark()
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="5abf965b-2f69-469e-a0cf-ba8ffd714764"/>
# MAGIC 
# MAGIC Pandas API on Spark DataFrameからSpark DataFrameへ変換します。

# COMMAND ----------

display(df.to_spark())

# COMMAND ----------

# MAGIC %md <i18n value="480e9e60-9286-4f4c-9db3-b650b32cb7ce"/>
# MAGIC 
# MAGIC ### 値のカウント (Value Counts)

# COMMAND ----------

# MAGIC %md <i18n value="99f93d32-d09d-4fea-9ac9-57099eb2c819"/>
# MAGIC 
# MAGIC PySparkで異なるプロパティタイプをカウントします。

# COMMAND ----------

display(spark_df.groupby("property_type").count().orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md <i18n value="150b6a18-123d-431a-84b1-ad2d2b7beae2"/>
# MAGIC 
# MAGIC Pandas API on Sparkで異なるプロパティタイプをカウントします。

# COMMAND ----------

df["property_type"].value_counts()

# COMMAND ----------

# MAGIC %md <i18n value="767f19b5-137f-4b33-9ef4-e5bb48603299"/>
# MAGIC 
# MAGIC ### 可視化
# MAGIC 
# MAGIC Pandas API on Sparkでは、可視化の種類に応じて、プロットの実行方法が最適化されています。
# MAGIC <br><br>
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/301/ps_plotting.png)

# COMMAND ----------

df.plot(kind="hist", x="bedrooms", y="price", bins=200)

# COMMAND ----------

# MAGIC %md <i18n value="6b70f1df-dfe1-43de-aeec-5541b036927c"/>
# MAGIC 
# MAGIC ### pandas API on Spark DataFramesをSQLで操作 (SQL on pandas API on Spark DataFrames)

# COMMAND ----------

ps.sql("SELECT distinct(property_type) FROM {df}")

# COMMAND ----------

# MAGIC %md <i18n value="7345361b-e6c4-4ce3-9ba4-8f132c8c8df2"/>
# MAGIC 
# MAGIC ### 興味深い事実
# MAGIC 
# MAGIC * Pandas API on Sparkを使えば、Delta Tablesから読み込んだり、ファイルのディレクトリから読み込んだりすることができます。
# MAGIC * Pandas API on Spark の DF が <1000 (デフォルト) の場合、pandas API on Spark は pandas をショートカットとして使用します - 閾値は **`compute.shortcut_limit`** で調整することが可能です。
# MAGIC * 棒グラフを作成した場合、上位n行のみが使用されます - これは **`plotting.max_rows`** を使用して調整することができます。
# MAGIC * **`.apply`** <a href="https://koalas.readthedocs.io/en/latest/reference/api/databricks.koalas.DataFrame.apply.html#databricks.koalas.DataFrame.apply" target="_blank">文書</a> と 戻り値のヒントを利用する方法はpandas UDF に似ています。
# MAGIC * 実行計画の確認方法、pandas API on Spark DFのキャッシュ方法（すぐに直感的に理解できるものではありません。）
# MAGIC * コアラは有袋類で、最高時速は30km/h(20 mph)です。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
