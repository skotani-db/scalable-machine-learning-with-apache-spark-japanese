# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="62811f6d-e550-4c60-8903-f38d7ed56ca7"/>
# MAGIC 
# MAGIC # 回帰: レンタル料金の予測 (Regression: Predicting Rental Price)
# MAGIC 
# MAGIC このノートブックでは、前回のラボでクレンジングしたデータセットを使って、サンフランシスコのAirbnbの料金を予測します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このレッスンでは次を行います:<br>
# MAGIC  - SparkML APIを使用して線形回帰モデルを構築する
# MAGIC  - (推定器)estimatorと変換器(transformer)の違いを明確にする

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)

# COMMAND ----------

# MAGIC %md <i18n value="ee10d185-fc70-48b8-8efe-ea2feee28e01"/>
# MAGIC 
# MAGIC ## トレーニング用データ、テスト用データの分割 (Train/Test Split)
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/301/TrainTestSplit.png)
# MAGIC 
# MAGIC **質問**: なぜ、seed を設定する必要がありますか？クラスタの構成(後述するパーティション数)を変更した場合はどうなりますか？

# COMMAND ----------

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)
print(train_df.cache().count())

# COMMAND ----------

# MAGIC %md <i18n value="b70f996a-31a2-4b62-a699-dc6026105465"/>
# MAGIC 
# MAGIC パーティションの数を変えてトレーニング用データの数が同じになるか見てみましょう。

# COMMAND ----------

train_repartition_df, test_repartition_df = (airbnb_df
                                             .repartition(24)
                                             .randomSplit([.8, .2], seed=42))

print(train_repartition_df.count())

# COMMAND ----------

# MAGIC %md <i18n value="5b96c695-717e-4269-84c7-8292ceff9d83"/>
# MAGIC 
# MAGIC ## 線形回帰 (Linear Regression)
# MAGIC 
# MAGIC シンプルなモデルとして、 **`寝室(bedroom)`** の数だけが与えられて **`価格(price)`** を予測するモデルを作ります。
# MAGIC 
# MAGIC **質問**: 線形回帰モデルの仮説にはどのようなものがありますか？

# COMMAND ----------

display(train_df.select("price", "bedrooms"))

# COMMAND ----------

display(train_df.select("price", "bedrooms").summary())

# COMMAND ----------

display(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="4171a9ae-e928-41e3-9689-c6fcc2b3d57c"/>
# MAGIC 
# MAGIC 価格についてデータセットに外れ値(outlier)があるようです（1泊10,000ドル？）。このことを念頭に置きながら、モデルを作っていきましょう。
# MAGIC 
# MAGIC <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegression.html?highlight=linearregression#pyspark.ml.regression.LinearRegression" target="_blank">線形回帰</a>を使って、最初のモデルを構築します。
# MAGIC 
# MAGIC Linear Regression による推定では、入力としてベクトルを想定しているため、以下のセルを実行すると失敗します。以下、VectorAssemblerを使って修正します。

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="bedrooms", labelCol="price")

# Uncomment when running
# lr_model = lr.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="f1353d2b-d9b8-4c8c-af18-2abb8f0d0b84"/>
# MAGIC 
# MAGIC ## Vector Assembler
# MAGIC 
# MAGIC 上のセルのコードでは、何がいけなかったでしょうか？線形回帰の**推定器(estimator)** ( **`.fit()関数`** ) は、Vector型の入力を受付けます。
# MAGIC 
# MAGIC  **`寝室(bedroom)`** 列から1つのベクトルを作ることは、 <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html?highlight=vectorassembler#pyspark.ml.feature.VectorAssembler" target="_blank">VectorAssembler</a> を使って簡単にできます。VectorAssemblerは、 **変換器(Transformer)** の１つです。TransformersはDataFrameを受け取り、1つまたは複数のカラムが追加された新しいDataFrameを返します。Transformerはデータから学習するのではなく、ルールに基づいた変換を行います。
# MAGIC 
# MAGIC VectorAssemblerの使用例は、 <a href="https://spark.apache.org/docs/latest/ml-features.html#vectorassembler" target="_blank">MLプログラミングガイド</a> に掲載されています。

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")

vec_train_df = vec_assembler.transform(train_df)

# COMMAND ----------

lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(vec_train_df)

# COMMAND ----------

# MAGIC %md <i18n value="ab8f4965-71db-487d-bbb3-329216580be5"/>
# MAGIC 
# MAGIC ## モデルの点検(Inspect the model)

# COMMAND ----------

m = lr_model.coefficients[0]
b = lr_model.intercept

print(f"The formula for the linear regression line is y = {m:.2f}x + {b:.2f}")

# COMMAND ----------

# MAGIC %md <i18n value="ae6dfaf9-9164-4dcc-a699-31184c4a962e"/>
# MAGIC 
# MAGIC ## テストデータへのモデルの適用 (Apply model to test set)

# COMMAND ----------

vec_test_df = vec_assembler.transform(test_df)

pred_df = lr_model.transform(vec_test_df)

pred_df.select("bedrooms", "features", "price", "prediction").show()

# COMMAND ----------

# MAGIC %md <i18n value="8d73c3ee-34bc-4f8b-b2ba-03597548680c"/>
# MAGIC 
# MAGIC ## モデルの評価 (Evaluate Model)
# MAGIC 
# MAGIC 変数が1つだけ(bedroom)の線形回帰モデルの予測結果を見てみましょう。ベースラインモデルに勝てるか？

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE is {rmse}")

# COMMAND ----------

# MAGIC %md <i18n value="703fbf0b-a2e1-4086-b002-8f63e06afdd8"/>
# MAGIC 
# MAGIC Wahoo!RMSEはベースラインモデルよりも良い結果になりました。とはいえ、やはりそんなにすごいものではありません。これからのノートブックで、さらに減らしていけるか見ていきましょう。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
