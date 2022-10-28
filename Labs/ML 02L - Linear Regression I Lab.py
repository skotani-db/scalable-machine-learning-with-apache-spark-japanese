# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="45bb1181-9fe0-4255-b0f0-b42637fc9591"/>
# MAGIC 
# MAGIC #線形回帰ラボ (Linear Regression Lab)
# MAGIC 
# MAGIC 前回のレッスンでは、「bedrooms」という1つの変数だけで「price」を予測しました。次に、他のいくつかの特徴量を使って「price」を予測してみます。
# MAGIC 
# MAGIC ステップ：
# MAGIC 1. **`bedrooms`** 、 **`bathrooms`** 、 **`bathrooms_na`** 、 **`minimum_nights`** 、 と **`number_of_reviews`** をVectorAssemblerの入力とします。
# MAGIC 1.線形回帰モデルを構築します。
# MAGIC 1. **`RMSE`** と **`R2`** を評価します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは以下を行います。<br>
# MAGIC  - 複数の特徴量を使用して線形回帰モデルを構築する。
# MAGIC  - 様々な指標で適合度を評価する。

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# TODO

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

vec_assembler = # FILL_IN

lr_model = # FILL_IN

pred_df = # FILL_IN

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
rmse = # FILL_IN
r2 = # FILL_IN
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md <i18n value="25a260af-8d6e-4897-8228-80074c4f1d64"/>
# MAGIC 
# MAGIC 各変数の係数を調べます。

# COMMAND ----------

for col, coef in zip(vec_assembler.getInputCols(), lr_model.coefficients):
    print(col, coef)
  
print(f"intercept: {lr_model.intercept}")

# COMMAND ----------

# MAGIC %md <i18n value="218d51b8-7453-4f6a-8965-5a60e8c80eaf"/>
# MAGIC 
# MAGIC ## 分散処理設定 (Distributed Setting)
# MAGIC 
# MAGIC データが小さいうちはパラメータを素早く解くことができますが、閉形式のソリューションは大規模なデータセットにうまく対応できません。
# MAGIC 
# MAGIC Sparkは、次のアプローチで線形回帰の問題を解きます。
# MAGIC 
# MAGIC * まず、Sparkは行列分解を使って線形回帰の問題を解こうとします。
# MAGIC * もし失敗した場合は、<a href="https://spark.apache.org/docs/latest/ml-advanced.html#limited-memory-bfgs-l-bfgs" target="_blank">L-BFGS</a>を使用してパラメータを解きます。L-BFGSはBFGSのメモリ制限版であり、特に非常に多くの変数を持つ問題に適しています。<a href="https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm" target="_blank">BFGS</a>は<a href="https://en.wikipedia.org/wiki/Quasi-Newton_method" target="_blank">準ニュートン法(quasi-newton methods)</a> に属し、関数のゼロまたは局所最大・最小を繰り返し求めるために使われる。
# MAGIC 
# MAGIC 線形回帰が分散環境でどのように実装されているか、ボトルネックに興味がある方は、これらのスライドをご覧ください。
# MAGIC * <a href="https://files.training.databricks.com/static/docs/distributed-linear-regression-1.pdf" target="_blank">distributed-linear-regression-1</a>
# MAGIC * <a href="https://files.training.databricks.com/static/docs/distributed-linear-regression-2.pdf" target="_blank">distributed-linear-regression-2</a>

# COMMAND ----------

# MAGIC %md <i18n value="f3e00d9e-3b02-44cf-87b7-20b54ba350c9"/>
# MAGIC 
# MAGIC ### 次のステップ
# MAGIC 
# MAGIC え？！かなりひどいモデルを作ってしまいました。次のノートブックでは、このモデルをさらに改良していきます。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
