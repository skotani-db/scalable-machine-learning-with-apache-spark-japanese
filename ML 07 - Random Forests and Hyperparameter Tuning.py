# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="2ab084da-06ed-457d-834a-1d19353e5c59"/>
# MAGIC 
# MAGIC # ランダムフォレストとハイパーパラメータチューニング(Random Forests and Hyperparameter Tuning)
# MAGIC 
# MAGIC では、グリッドサーチとクロスバリデーションを使って最適なハイパーパラメータを見つけるためのランダムフォレストをチューニングする方法を見ていきましょう。Databricks Runtime for MLを使用すると、MLflowは自動的にSparkML cross-validatorで実験(Experiment)をログに記録します!
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このレッスンで次を行います:<br>
# MAGIC  - グリッドサーチを用いたハイパーパラメータのチューニング
# MAGIC  - SparkMLパイプラインの最適化

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

rf = RandomForestRegressor(labelCol="price", maxBins=40)
stages = [string_indexer, vec_assembler, rf]
pipeline = Pipeline(stages=stages)

# COMMAND ----------

# MAGIC %md <i18n value="4561938e-90b5-413c-9e25-ef15ba40e99c"/>
# MAGIC 
# MAGIC ## ParamGrid
# MAGIC 
# MAGIC まず、ランダムフォレストで調整できる様々なハイパーパラメータを見てみましょう。
# MAGIC 
# MAGIC **ポップクイズ：**パラメータとハイパーパラメータの違いは？

# COMMAND ----------

print(rf.explainParams())

# COMMAND ----------

# MAGIC %md <i18n value="819de6f9-75d2-45df-beb1-6b59ecd2cfd2"/>
# MAGIC 
# MAGIC チューニングできるハイパーパラメーターはたくさんあり、手作業で設定するには時間がかかるでしょう。
# MAGIC 
# MAGIC 手作業（アドホック）ではなく、Sparkの <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html?highlight=paramgridbuilder#pyspark.ml.tuning.ParamGridBuilder" target="_blank">ParamGridBuilder</a> を使って、よりシステマチックに最適なハイパーパラメータを求めましょう。
# MAGIC 
# MAGIC テストするハイパーパラメータのグリッドを定義してみましょう。
# MAGIC * **`maxDepth`** : 各決定木の最大の深さ (次の値を使用 :  **`2, 5`** )
# MAGIC * **`木の本数`** : 学習する決定木の本数 (次の値を使用 :  **`5, 10`**)
# MAGIC 
# MAGIC **`addGrid()`** は、パラメータの名前 (例. **`rf.maxDepth`**) と、取り得る値のリスト (例えば **`[2, 5]`** )を受け付けます.

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

param_grid = (ParamGridBuilder()
              .addGrid(rf.maxDepth, [2, 5])
              .addGrid(rf.numTrees, [5, 10])
              .build())

# COMMAND ----------

# MAGIC %md <i18n value="9f043287-11b8-482d-8501-2f7d8b1458ea"/>
# MAGIC 
# MAGIC ## クロスバリデーション (Cross Validation)
# MAGIC 
# MAGIC 3-fold クロスバリデーションを用いて最適なハイパーパラメータを特定することにします。
# MAGIC 
# MAGIC ![crossValidation](https://files.training.databricks.com/images/301/CrossValidation.png)
# MAGIC 
# MAGIC 3-fold クロスバリデーションでは、2/3のデータで学習し、残りの1/3のデータ（ホールドアウトセット）で評価します。このプロセスを3回繰り返し、各foldが検証セットとして利用されます。そして、3回の結果の平均を取ります。

# COMMAND ----------

# MAGIC %md <i18n value="ec0440ab-071d-4201-be86-5eeedaf80a4f"/>
# MAGIC 
# MAGIC <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator" target="_blank">CrossValidator</a> に、 **`estimator`** (pipeline)、 **`evaluator`** 、 **`estimatorParamMaps`** を渡すことで、以下を設定します： 
# MAGIC * どのモデルを使うか
# MAGIC * どのようにモデルを評価するか
# MAGIC * モデルに対してどのハイパーパラメータを設定するか
# MAGIC 
# MAGIC また、データをいくつのfoldに分割するか（3つ）、また、再現性を確保するためseed（42）も設定もできます。

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")

cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=param_grid, 
                    numFolds=3, seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="673c9261-a861-4ace-b008-c04565230a8e"/>
# MAGIC 
# MAGIC **質問** :今、いくつのモデルをトレーニングしているでしょうか？

# COMMAND ----------

cv_model = cv.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="c9bc1596-7b0f-4595-942c-109cfca51698"/>
# MAGIC 
# MAGIC ## 並列化パラメータ (Parallelism Parameter)
# MAGIC 
# MAGIC うーん、実行に時間がかかりましたね。それは、モデルが並列ではなく、逐次的に学習されていたためです。
# MAGIC 
# MAGIC Spark 2.3では、 <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator.parallelism" target="_blank">Parallelism（並列化）</a>というパラメータが導入されました。（ドキュメント引用: **`並列アルゴリズムを実行する際に使用するスレッド数 (>= 1)`** 。）
# MAGIC 
# MAGIC この値を4にして、より速くトレーニングできるかどうか見てみましょう。Sparkの <a href="https://spark.apache.org/docs/latest/ml-tuning.html" target="_blank">ドキュメント</a> では、2～10の間の値を推奨しています。

# COMMAND ----------

cv_model = cv.setParallelism(4).fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="2d00b40f-c5e7-4089-890b-a50ccced34c6"/>
# MAGIC 
# MAGIC **質問**: うーん、まだ実行に時間がかかりましたね。cross validatorの中にパイプラインを入れるべきか、パイプラインの中にcross validatorを入れるべきか、どちらでしょうか？
# MAGIC 
# MAGIC パイプラインにestimatorやtransformerがあるかによります。
# MAGIC 
# MAGIC パイプラインの中にcross validatorを入れるほう：パイプラインにStringIndexer（estimator）などがある場合、パイプライン全体をクロスバリデータにすると、毎回リフィットする必要があります（上が例）。ホールドアウトセットからトレーニングセットへのデータ漏洩の懸念がある場合、この方法が最も安全です。cross validatorはまずデータを分割し、次にパイプラインをフィットします。
# MAGIC 
# MAGIC cross validatorの中にパイプラインを入れるほう：下の例のようにパイプラインの末尾にcross validatorを配置すると、(cross validatorがデータを分割する前にestimatorがfitをすると)ホールドアウトセットからトレーニングセットに情報が漏れる可能性があります。

# COMMAND ----------

cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=param_grid, 
                    numFolds=3, parallelism=4, seed=42)

stages_with_cv = [string_indexer, vec_assembler, cv]
pipeline = Pipeline(stages=stages_with_cv)

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="dede990c-2551-4c07-8aad-d697ae827e71"/>
# MAGIC 
# MAGIC 最適なハイパーパラメータ構成におけるモデルを見てみましょう。

# COMMAND ----------

list(zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics))

# COMMAND ----------

pred_df = pipeline_model.transform(test_df)

rmse = evaluator.evaluate(pred_df)
r2 = evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md <i18n value="8f80daf2-8f0b-4cab-a8e6-4060c78d94b0"/>
# MAGIC 
# MAGIC 改良されてますね! 決定木の性能を上回っているようですね。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
