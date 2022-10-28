# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="4e1b9835-762c-42f2-9ff8-75164cb1a702"/>
# MAGIC 
# MAGIC # 線形回帰 II ラボ (Linear Regression II Lab)
# MAGIC 
# MAGIC よしっ!モデルを改良していきます。RMSEやR2はまだそれほどよくないですが、ベースラインや単一の特徴量を使用するよりは優れています。
# MAGIC 
# MAGIC このラボでは、モデルのパフォーマンスをさらに向上させる方法を紹介します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンで以下を行います。<br>
# MAGIC  - RFormulaを使用して、StringIndexer、OneHotEncoder、VectorAssemblerの処理を簡略化する。
# MAGIC  - 「price」を「log(price)」に変換して予測し、その結果を指数化することでRMSEを低下させる。

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="a427d25c-591f-4899-866a-14064eff40e3"/>
# MAGIC 
# MAGIC ## RFormula
# MAGIC 
# MAGIC StringIndexer と OneHotEncoder に対して、どの列がカテゴリ型であるかを手動で指定する代わりに、<a href="(https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.RFormula.html?highlight=rformula#pyspark.ml.feature.RFormula" target="_blank">RFormula</a>が自動的にそれを行ってくれます。
# MAGIC 
# MAGIC RFormulaでは、String型のカラムをカテゴリ型として扱い、自動的にStringIndexerとOneHotEncoderを実行してくれます。String型でなければ、そのままにしておきます。そして、one-hot encodeされた特徴量と数値型特徴量をまとめて、 **`features`** という1つのベクトルにします。
# MAGIC 
# MAGIC RFormulaの詳しい使用例は<a href="https://spark.apache.org/docs/latest/ml-features.html#rformula" target="_blank">こちら</a>をご覧ください。

# COMMAND ----------

# ANSWER
from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

r_formula = RFormula(formula="price ~ .", featuresCol="features", labelCol="price", handleInvalid="skip") # Look at handleInvalid

lr = LinearRegression(labelCol="price", featuresCol="features")
pipeline = Pipeline(stages=[r_formula, lr])
pipeline_model = pipeline.fit(train_df)
pred_df = pipeline_model.transform(test_df)

regression_evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")
rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md <i18n value="c9898a31-90e4-4a6d-87e6-731b95c764bd"/>
# MAGIC 
# MAGIC ## 対数スケール (Log Scale)
# MAGIC 
# MAGIC さて、RFormulaを使っても同じ結果が得られることが確認できたので、このモデルを改良していきます。思い起こせば、「price」変数は対数正規分布しているように見えるので、対数スケールで予測してみます。
# MAGIC 
# MAGIC 「price」を対数変換し、線形回帰モデルで「log(price)」を予測します。

# COMMAND ----------

from pyspark.sql.functions import log

display(train_df.select(log("price")))

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import col, log

log_train_df = train_df.withColumn("log_price", log(col("price")))
log_test_df = test_df.withColumn("log_price", log(col("price")))

r_formula = RFormula(formula="log_price ~ . - price", featuresCol="features", labelCol="log_price", handleInvalid="skip") 

lr.setLabelCol("log_price").setPredictionCol("log_pred")
pipeline = Pipeline(stages=[r_formula, lr])
pipeline_model = pipeline.fit(log_train_df)
pred_df = pipeline_model.transform(log_test_df)

# COMMAND ----------

# MAGIC %md <i18n value="51b5e35f-e527-438a-ab56-2d4d0d389d29"/>
# MAGIC 
# MAGIC ## 指数化 (Exponentiate)
# MAGIC 
# MAGIC RMSEを解釈するためには、予測値を対数スケールから逆変換する必要があります。

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import col, exp

exp_df = pred_df.withColumn("prediction", exp(col("log_pred")))

rmse = regression_evaluator.setMetricName("rmse").evaluate(exp_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(exp_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md <i18n value="05d3baa6-bb71-4c31-984b-a2daabc35f97"/>
# MAGIC 
# MAGIC よくやった！以前のモデルと比較して、R2が増加し、RMSEが大幅に減少しています。
# MAGIC 
# MAGIC 次のノートブックでは、RMSEをさらに低減する方法を紹介します。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
