# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="8c6d3ef3-e44b-4292-a0d3-1aaba0198525"/>
# MAGIC 
# MAGIC # データクレンジング (Data Cleansing)
# MAGIC 
# MAGIC 今回はSparkを使って、<a href="http://insideairbnb.com/get-the-data.html" target="_blank">Inside Airbnb</a>のSF Airbnb賃貸データセットの探索的データ解析とクレンジングを行います。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/301/sf.jpg" style="height: 200px; margin: 10px; border: 1px solid #ddd; padding: 10px"/>
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このレッスンで次を行います: <br>
# MAGIC  - 欠損値の補完
# MAGIC  - 外れ値の特定と除外

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="969507ea-bffc-4255-9a99-2306a594625f"/>
# MAGIC 
# MAGIC Airbnbのデータセットをロードしてみましょう。

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06.csv"

raw_df = spark.read.csv(file_path, header="true", inferSchema="true", multiLine="true", escape='"')

display(raw_df)

# COMMAND ----------

raw_df.columns

# COMMAND ----------

# MAGIC %md <i18n value="94856418-c319-4915-a73e-5728fcd44101"/>
# MAGIC 
# MAGIC シンプルにするため、特定のカラムだけを残すようにします。特徴量の選択については後述します。

# COMMAND ----------

columns_to_keep = [
    "host_is_superhost",
    "cancellation_policy",
    "instant_bookable",
    "host_total_listings_count",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "bed_type",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "price"
]

base_df = raw_df.select(columns_to_keep)
base_df.cache().count()
display(base_df)

# COMMAND ----------

# MAGIC %md <i18n value="a12c5a59-ad1c-4542-8695-d822ec10c4ca"/>
# MAGIC 
# MAGIC ### データ型の修正 (Fixing Data Types)
# MAGIC 
# MAGIC 上のスキーマを見てください。 **`賃貸価格(price)`** フィールドが文字列として扱われたことに気がつくでしょう。このタスクでは 数値（double 型）フィールドである必要があります。 
# MAGIC 
# MAGIC データ型を修正しましょう。

# COMMAND ----------

from pyspark.sql.functions import col, translate

fixed_price_df = base_df.withColumn("price", translate(col("price"), "$,", "").cast("double"))

display(fixed_price_df)

# COMMAND ----------

# MAGIC %md <i18n value="4ad08138-4563-4a93-b038-801832c9bc73"/>
# MAGIC 
# MAGIC ### 要約統計 (Summary statistics)
# MAGIC 
# MAGIC 要約統計を出力する2つのオプションがあります。
# MAGIC * **`describe`** : count, mean, stddev, min, max 
# MAGIC * **`summary`** : 上記のdescribeの項目 + 四分位範囲(IQR : interquartile range)
# MAGIC 
# MAGIC **質問** : 平均値よりも IQR/中央値を使うべき時は？その逆は？

# COMMAND ----------

display(fixed_price_df.describe())

# COMMAND ----------

display(fixed_price_df.summary())

# COMMAND ----------

# MAGIC %md <i18n value="bd55efda-86d0-4584-a6fc-ef4f221b2872"/>
# MAGIC 
# MAGIC ### Dbutils によるデータ要約 (Dnutils Data Summary)
# MAGIC 
# MAGIC  **`dbutils.data.summarize`** を使用して、より詳細な要約統計とデータプロットを見ることができます。

# COMMAND ----------

dbutils.data.summarize(fixed_price_df)

# COMMAND ----------

# MAGIC %md <i18n value="e9860f92-2fbe-4d23-b728-678a7bb4734e"/>
# MAGIC 
# MAGIC ### 極端な値の除外 (Geeting rid of exterme values)
# MAGIC 
# MAGIC  **`price`** 列について *最小値* と *最大値* を見てみましょう。

# COMMAND ----------

display(fixed_price_df.select("price").describe())

# COMMAND ----------

# MAGIC %md <i18n value="4a8fe21b-1dac-4edf-a0a3-204f170b05c9"/>
# MAGIC 
# MAGIC 超高額な物件もありますが、そのデータをどうするかはSME（Subject Matter Experts）次第です。しかし「無料」のAirbnbはフィルタリングすることは可能でしょう。
# MAGIC 
# MAGIC まず、*price* がゼロの物件がいくつあるか見てみましょう。

# COMMAND ----------

fixed_price_df.filter(col("price") == 0).count()

# COMMAND ----------

# MAGIC %md <i18n value="bf195d9b-ea4d-4a3e-8b61-372be8eec327"/>
# MAGIC 
# MAGIC *price* が0より大きい行だけを残すようにします。

# COMMAND ----------

pos_prices_df = fixed_price_df.filter(col("price") > 0)

# COMMAND ----------

# MAGIC %md <i18n value="dc8600db-ebd1-4110-bfb1-ce555bc95245"/>
# MAGIC 
# MAGIC *minimum\_nights* カラムの *最小値* と *最大値* を見てみましょう。

# COMMAND ----------

display(pos_prices_df.select("minimum_nights").describe())

# COMMAND ----------

display(pos_prices_df
        .groupBy("minimum_nights").count()
        .orderBy(col("count").desc(), col("minimum_nights"))
       )

# COMMAND ----------

# MAGIC %md <i18n value="5aa4dfa8-d9a1-42e2-9060-a5dcc3513a0d"/>
# MAGIC 
# MAGIC minimum\_nights の上限は1年が妥当と思われます。*minimum\_nights* が365より大きいレコードをフィルタリングしてみましょう。

# COMMAND ----------

min_nights_df = pos_prices_df.filter(col("minimum_nights") <= 365)

display(min_nights_df)

# COMMAND ----------

# MAGIC %md <i18n value="25a35390-d716-43ad-8f51-7e7690e1c913"/>
# MAGIC 
# MAGIC ### 欠損値の取り扱い (Handling Null Values)
# MAGIC 
# MAGIC 欠損値を扱うには、さまざまな方法があります。時には、欠損であることが実際に予測するための重要な指標となることもあります（例えば、フォームの特定の部分を記入しなければ、それが承認される確率は低下します）。
# MAGIC 
# MAGIC 欠損を処理するいくつかの方法: 
# MAGIC * 欠損を含むレコードはすべて削除する。 
# MAGIC * 数値型の場合:
# MAGIC     * 平均値/中央値/ゼロ/その他で置き換える。
# MAGIC * カテゴリカル変数の場合: 
# MAGIC     * 最頻値に置き換える。
# MAGIC     * 欠損を表す表現を用意する
# MAGIC * 欠損値補完のために設計された ALS (Alternating Least Squares) のようなテクニックを使用する。
# MAGIC   
# MAGIC **カテゴリ／数値特徴量に対して補完(Imputation)を行う場合、このフィールドが補完されたことがわかるように新フィールドを追加しなければならない（MUST）。**
# MAGIC 
# MAGIC SparkMLのImputer（後述）は、カテゴリ特徴量はサポートしていません。

# COMMAND ----------

# MAGIC %md <i18n value="83e56fca-ce6d-4e3c-8042-0c1c7b9eaa5a"/>
# MAGIC 
# MAGIC ### 補完：Double型へのキャスト (Impute: Cast to Double)
# MAGIC 
# MAGIC SparkMLの <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html?highlight=imputer#pyspark.ml.feature.Imputer" target="_blank">Imputer</a> は、すべてのフィールドがdouble型であることを要求しています。すべての整数フィールドをdoubleにキャストしてみましょう

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

integer_columns = [x.name for x in min_nights_df.schema.fields if x.dataType == IntegerType()]
doubles_df = min_nights_df

for c in integer_columns:
    doubles_df = doubles_df.withColumn(c, col(c).cast("double"))

columns = "\n - ".join(integer_columns)
print(f"Columns converted from Integer to Double:\n - {columns}")

# COMMAND ----------

# MAGIC %md <i18n value="69b58107-82ad-4cec-8984-028a5df1b69e"/>
# MAGIC 
# MAGIC 代入の前に、NULL値の存在を示すダミー列を追加する。

# COMMAND ----------

from pyspark.sql.functions import when

impute_cols = [
    "bedrooms",
    "bathrooms",
    "beds", 
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

for c in impute_cols:
    doubles_df = doubles_df.withColumn(c + "_na", when(col(c).isNull(), 1.0).otherwise(0.0))

# COMMAND ----------

display(doubles_df.describe())

# COMMAND ----------

# MAGIC %md <i18n value="c88f432d-1252-4acc-8c91-4834c00da789"/>
# MAGIC 
# MAGIC ### 変換器と推定器 (Transformers and Estimators)
# MAGIC 
# MAGIC Spark MLは、機械学習アルゴリズムのAPIを標準化し、複数のアルゴリズムを1つのパイプライン（ワークフロー）にまとめることを容易にしている。Spark ML APIで導入された2つの重要な概念について説明します: **`変換器(transformer)`** と **`推定器(estimator)`** .
# MAGIC 
# MAGIC **変換器(transformer)**: DataFrameを別のDataFrameに変換します。DataFrameを入力として受け取り、1つまたは複数の列が追加された新しいDataFrameを返します。Transformerはデータからパラメータを学習せず、単純にルールベースの変換を適用します。Transformerは **`.transform()`** メソッドを持ちます。
# MAGIC 
# MAGIC **推定器(estimator)**: DataFrameが持つデータにフィットして、Transformerを生成することができるアルゴリズムです。例えば、学習アルゴリズムはDataFrameから学習し、モデルを生成するestimatorです。Estimatorは **`.fit()`** メソッドを持っており、DataFrameからパラメータを学習（または「フィット」）します。

# COMMAND ----------

from pyspark.ml.feature import Imputer

imputer = Imputer(strategy="median", inputCols=impute_cols, outputCols=impute_cols)

imputer_model = imputer.fit(doubles_df)
imputed_df = imputer_model.transform(doubles_df)

# COMMAND ----------

# MAGIC %md <i18n value="4df06e83-27e6-4cc6-b66d-883317b2a7eb"/>
# MAGIC 
# MAGIC これでデータはきれいになりました。このDataFrameをDeltaに保存して、モデル作りを始めましょう。

# COMMAND ----------

imputed_df.write.format("delta").mode("overwrite").save(f"{DA.paths.working_dir}/imputed_results")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
