# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="b9944704-a562-44e0-8ef6-8639f11312ca"/>
# MAGIC 
# MAGIC # XGBoost
# MAGIC 
# MAGIC ここまでは、SparkMLのみを使用してきました。3rdパーティライブラリの勾配ブースティング決定木(Gradient Boosted Trees)を見てみましょう。 
# MAGIC  
# MAGIC <a href="https://docs.microsoft.com/en-us/azure/databricks/runtime/mlruntime" target="_blank">Databricks Runtime for ML</a> には分散XGBoostがインストールされているので、それを使用していることを確認してください。 
# MAGIC 
# MAGIC **質問**：gradient boosted treeとrandom forestの違いは？どの部分を並列化できるのか？
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは次を行います:<br>
# MAGIC  - サードパーティライブラリ（XGBoost）を使ってモデルをさらに改善します

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="3e08ca45-9a00-4c6a-ac38-169c7e87d9e4"/>
# MAGIC 
# MAGIC ## データ準備 (Data Preparation)
# MAGIC 
# MAGIC まず、すべてのカテゴリ型特徴量のインデックスを作成し、ラベルを **`log(price)`** に設定します.

# COMMAND ----------

from pyspark.sql.functions import log, col
from pyspark.ml.feature import StringIndexer, VectorAssembler

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.withColumn("label", log(col("price"))).randomSplit([.8, .2], seed=42)

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price") & (field != "label"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

# MAGIC %md <i18n value="733cd880-143d-42c2-9f29-602e48f60efe"/>
# MAGIC 
# MAGIC ### Pyspark Distributed XGBoost
# MAGIC 
# MAGIC それでは、分散型XGBoostのモデルを作ってみましょう。技術的にはMLlibの一部ではありませんが、<a href="https://databricks.github.io/spark-deep-learning/_modules/sparkdl/xgboost/xgboost.html" target="_blank">XGBoost</a> をMLのパイプラインに統合することができます。 
# MAGIC 
# MAGIC Pyspark XGBoost の分散処理バージョンを使用する際に、2つの追加パラメータを指定することができます。
# MAGIC 
# MAGIC * **`num_workers`** : 分散処理するworkerの数。MLR 9.0以上が必要です。
# MAGIC * **`use_gpu`** : GPUを使ったトレーニングを可能にし、より高速なパフォーマンスを実現します（オプション）。
# MAGIC 
# MAGIC **注意:** **`use_gpu`** は、ML GPUランタイムを必要とします。現在、分散学習を行う際に使用するGPUは、workerあたり最大1つです。

# COMMAND ----------

from sparkdl.xgboost import XgboostRegressor
from pyspark.ml import Pipeline

params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42, "missing": 0}

xgboost = XgboostRegressor(**params)

pipeline = Pipeline(stages=[string_indexer, vec_assembler, xgboost])
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="8d5f8c24-ee0b-476e-a250-95ce2d73dd28"/>
# MAGIC 
# MAGIC ## 評価 (Evaluate)
# MAGIC 
# MAGIC XGBoostのモデルの性能を評価します。指数にするのを忘れずに。

# COMMAND ----------

from pyspark.sql.functions import exp, col

log_pred_df = pipeline_model.transform(test_df)

exp_xgboost_df = log_pred_df.withColumn("prediction", exp(col("prediction")))

display(exp_xgboost_df.select("price", "prediction"))

# COMMAND ----------

# MAGIC %md <i18n value="364402e1-8073-4b24-8e03-c7e2566f94d2"/>
# MAGIC 
# MAGIC メトリクスを計算します。

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(exp_xgboost_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(exp_xgboost_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md <i18n value="21cf0d1b-c7a8-43c0-8eea-7677bb0d7847"/>
# MAGIC 
# MAGIC ## 他の Gradient Boosted アプローチ
# MAGIC 
# MAGIC XGBoostの他にも <a href="https://catboost.ai/" target="_blank">CatBoost</a> 、 <a href="https://github.com/microsoft/LightGBM" target="_blank">LightGBM</a> 、 <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html?highlight=gbt#pyspark.ml.classification.GBTClassifier" target="_blank">SparkML</a> / <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html" target="_blank">scikit-learn</a> における基本的な(バニラの)勾配ブースティング決定木など、Gradient Boosted アプローチはたくさんあります。それぞれ <a href="https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db" target="_blank">長所と短所</a> があるので、詳しくはそちらをご覧ください。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
