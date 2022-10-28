# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="263caa08-bb08-4022-8d8f-bd2f51d77752"/>
# MAGIC 
# MAGIC # 分類：(Classification:)ロジスティック回帰
# MAGIC 
# MAGIC ここまでは、回帰のユースケースのみを検証してきました。では、分類の扱い方を見てみましょう。
# MAGIC 
# MAGIC このラボでは、同じAirbnbのデータセットを使用しますが、価格を予測する代わりに、ホストが<a href="https://www.airbnb.com/superhost" target="_blank">スーパーホスト</a>であるかどうかを予測します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは以下を行います。<br>
# MAGIC  - ロジスティック回帰モデルを構築
# MAGIC  - モデルの性能を評価するための様々なメトリックスを使用

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)

# COMMAND ----------

# MAGIC %md <i18n value="3f07e772-c15d-46e4-8acd-866b661fbb9b"/>
# MAGIC 
# MAGIC ## ベースラインモデル (Baseline Model)
# MAGIC 
# MAGIC 機械学習モデルを構築する前に、比較するためのベースラインモデルを構築します。まず、ホストが <a href="https://www.airbnb.com/superhost" target="_blank">superhost</a> であるかどうかを予測することから始めます。
# MAGIC 
# MAGIC ベースラインモデルでは、誰もスーパーホストではないことを予測し、その精度を評価することにしています。他の指標については、後ほどより複雑なモデルを構築する際に検討する予定です。
# MAGIC 
# MAGIC 0. **`host_is_superhost`** カラム (t/f) を 1/0 に変換し、結果のカラムを **`label`** と呼びます。その後、 **`host_is_superhost`** を DROPします。
# MAGIC 0. 結果のDataFrameに、全ての値を **`0.0`** にした **`prediction`** カラムを追加します。誰もスーパーホストではないことを常に予測するようにします。
# MAGIC 
# MAGIC この2つのステップを終えたら、次に「モデル」の精度を評価します。
# MAGIC 
# MAGIC 便利な機能がいくつかあります。
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.when.html#pyspark.sql.functions.when" target="_blank">when()</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.withColumn.html?highlight=withcolumn#pyspark.sql.DataFrame.withColumn" target="_blank">withColumn()</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.lit.html?highlight=lit#pyspark.sql.functions.lit" target="_blank">lit()</a>

# COMMAND ----------

# ANSWER

from pyspark.sql.functions import when, col, lit

label_df = airbnb_df.select(when(col("host_is_superhost") == "t", 1.0).otherwise(0.0).alias("label"), "*").drop("host_is_superhost")

pred_df = label_df.withColumn("prediction", lit(0.0))

# COMMAND ----------

# MAGIC %md <i18n value="d04eb817-2010-4021-a898-42ca8abaa00d"/>
# MAGIC 
# MAGIC ## モデル評価
# MAGIC 
# MAGIC とりあえず、「accuracy」を指標にしましょう。<a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.MulticlassClassificationEvaluator.html?highlight=multiclassclassificationevaluator#pyspark.ml.evaluation.MulticlassClassificationEvaluator" target="_blank">MulticlassClassificationEvaluator</a> を使います。

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

mc_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"The accuracy is {100*mc_evaluator.evaluate(pred_df):.2f}%")

# COMMAND ----------

# MAGIC %md <i18n value="5fe00f31-d186-4ab8-b6bb-437f7ddc4a00"/>
# MAGIC 
# MAGIC ## Train-Test分割 (Train-Test Split)
# MAGIC 
# MAGIC よしっ!これでベースラインモデルができました。次に、データを学習用とテスト用に分割します。

# COMMAND ----------

train_df, test_df = label_df.randomSplit([.8, .2], seed=42)
print(train_df.cache().count())

# COMMAND ----------

# MAGIC %md <i18n value="a7998d44-af91-4dfa-b80c-8b96ebfe5311"/>
# MAGIC 
# MAGIC ## 可視化 (Visualize)
# MAGIC 
# MAGIC トレーニングデータセットにおける **`review_scores_rating`** と **`label`** の関係性を見てみましょう。

# COMMAND ----------

display(train_df.select("review_scores_rating", "label"))

# COMMAND ----------

# MAGIC %md <i18n value="1ce4ba05-f558-484d-a8e8-53bde1e119fc"/>
# MAGIC 
# MAGIC ## ロジスティック回帰 (Logistic Regression)
# MAGIC 
# MAGIC ここで、すべての特徴量を使用して、<a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html?highlight=logisticregression#pyspark.ml.classification.LogisticRegression" target="_blank">ロジスティック回帰モデル</a>を構築します（ヒント：RFormulaを使用します）。前処理ステップとロジスティック回帰モデル構築ステップをPipelineに入れます。

# COMMAND ----------

# ANSWER
from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.classification import LogisticRegression

r_formula = RFormula(formula="label ~ .", 
                    featuresCol="features", 
                    labelCol="label", 
                    handleInvalid="skip") # Look at handleInvalid

lr = LogisticRegression(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[r_formula, lr])
pipeline_model = pipeline.fit(train_df)
pred_df = pipeline_model.transform(test_df)

# COMMAND ----------

# MAGIC %md <i18n value="3a06d71c-8551-44c8-b33e-8ae40a443713"/>
# MAGIC 
# MAGIC ## 評価 (Evaluate)
# MAGIC 
# MAGIC AUROCは何に役に立つでしょうか？PR曲線下の面積のような評価指標を追加します。

# COMMAND ----------

# ANSWER
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

mc_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"The accuracy is {100*mc_evaluator.evaluate(pred_df):.2f}%")

bc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print(f"The area under the ROC curve: {bc_evaluator.evaluate(pred_df):.2f}")

bc_evaluator.setMetricName("areaUnderPR")
print(f"The area under the PR curve: {bc_evaluator.evaluate(pred_df):.2f}")

# COMMAND ----------

# MAGIC %md <i18n value="0ef0e2b9-6ce9-4377-8587-83b5260fd05a"/>
# MAGIC 
# MAGIC ## ハイパーパラメータチューニングの追加 (Add Hyperparameter Tuning)
# MAGIC 
# MAGIC クロスバリデーターを用いて、ロジスティック回帰モデルのハイパーパラメーターを変更します。どの程度、指標を改善できますか？

# COMMAND ----------

# ANSWER
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

param_grid = (ParamGridBuilder()
            .addGrid(lr.regParam, [0.1, 0.2])
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
            .build())

cv = CrossValidator(estimator=lr, evaluator=mc_evaluator, estimatorParamMaps=param_grid,
                    numFolds=3, parallelism=4, seed=42)

pipeline = Pipeline(stages=[r_formula, cv])

pipeline_model = pipeline.fit(train_df)

pred_df = pipeline_model.transform(test_df)

# COMMAND ----------

# MAGIC %md <i18n value="111f2dc7-5535-45b7-82f6-ad2e5f2cbf16"/>
# MAGIC 
# MAGIC ## 再評価 (Evaluate again)

# COMMAND ----------

mc_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(f"The accuracy is {100*mc_evaluator.evaluate(pred_df):.2f}%")

bc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print(f"The area under the ROC curve: {bc_evaluator.evaluate(pred_df):.2f}")

# COMMAND ----------

# MAGIC %md <i18n value="7e88e044-0a34-4815-8eab-1dc37532a082"/>
# MAGIC 
# MAGIC ## スーパーボーナス (Super Bonus)
# MAGIC 
# MAGIC MLflowを使って実験を記録してみましょう。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
