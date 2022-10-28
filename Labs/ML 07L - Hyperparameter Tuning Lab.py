# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="b778c8d0-84e6-4192-a921-b9b60fd20d9b"/>
# MAGIC 
# MAGIC # ランダムフォレストのハイパーパラメータチューニング (Hyperparameter Tuning with Random Forests)
# MAGIC 
# MAGIC このラボでは、Airbnb問題を分類問題に変換し、ランダムフォレスト分類器を構築し、ランダムフォレストのいくつかのハイパーパラメータを調整します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンで以下を行います。<br>
# MAGIC  - ランダムフォレストに対するグリッド探索の実行
# MAGIC  - 特徴の重要度スコアと分類メトリクスの生成
# MAGIC  - scikit-learnのRandom ForestとSparkMLの違いの理解
# MAGIC  
# MAGIC 分散ランダムフォレストの実装については、Sparkの<a href="https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/tree/impl/RandomForest.scala#L42" target="_blank">ソースコード</a>に詳しく書かれています。

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="02dc0920-88e1-4f5b-886c-62b8cc02d1bb"/>
# MAGIC 
# MAGIC #回帰から分類へ (From Regression to Classification)
# MAGIC 
# MAGIC 今回は、Airbnbのデータセットを分類問題に変えて、**価格の高い部件と低い部件を分類します**。 **`class`** カラムは次のように設定します。
# MAGIC 
# MAGIC - 150ドル以下の安価な部件を **`0`** にする。
# MAGIC - 150ドル以上の高額物件を **`1`** にする。

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"

airbnb_df = (spark
            .read
            .format("delta")
            .load(file_path)
            .withColumn("priceClass", (col("price") >= 150).cast("int"))
            .drop("price")
           )

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "priceClass"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

# MAGIC %md <i18n value="e3bb8033-43ea-439c-a134-36bedbeff408"/>
# MAGIC 
# MAGIC ## なぜOHEを使わないか？(Why can't we OHE?)
# MAGIC 
# MAGIC **Question:** ランダムフォレストに渡す前に、変数をOne Hot Encodedをすると、何がうまくいかないのでしょうか？
# MAGIC 
# MAGIC **HINT:** 特徴量抽出の「ランダム性」がどうなるか考えてみてください。

# COMMAND ----------

# MAGIC %md <i18n value="0e9bdc2f-0d8d-41cb-9509-47833d66bc5e"/>
# MAGIC 
# MAGIC ## ランダムフォレスト (Random Forest)
# MAGIC 
# MAGIC **`labelCol=priceClass`** 、 **`maxBins=40`** 、 **`seed=42`** （再現性のため）で **`rf`** というランダムフォレスト分類器を作成します。
# MAGIC 
# MAGIC Pythonの **`pyspark.ml.classification.RandomForestClassifier`** の下にあります。

# COMMAND ----------

# TODO

rf = <FILL_IN>

# COMMAND ----------

# MAGIC %md <i18n value="7f3962e7-51b8-4477-9599-2465ab94a049"/>
# MAGIC 
# MAGIC ## グリッド検索 (Grid Search)
# MAGIC 
# MAGIC チューニングできるハイパーパラメーターはたくさんあり、手作業で設定するには時間がかかります。
# MAGIC 
# MAGIC Sparkの<a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html?highlight=paramgrid#pyspark.ml.tuning.ParamGridBuilder" target="_blank">ParamGridBuilder</a>を使って、より体系的に最適なハイパーパラメータを見つけてみましょう。この変数を **`param_grid`** にします。
# MAGIC 
# MAGIC テストするためハイパーパラメータのグリッドを定義します。
# MAGIC   - maxDepth: 決定木の深さの最大値 ( **`2, 5, 10`** を使用)
# MAGIC   - numTrees: 決定木の数 ( **`10、20、100`** を使用)
# MAGIC 
# MAGIC **`addGrid()`** に、パラメータ名（例： **`rf.maxDepth`** ）と、取り得る値のリスト（例： **`[2, 5, 10]`** ）を渡します。

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md <i18n value="e1862bae-e31e-4f5a-ab0e-926261c4e27b"/>
# MAGIC 
# MAGIC ## 評価器 (Evaluator)
# MAGIC 
# MAGIC 前のレッスンには、 **`RegressionEvaluator`** を使用していました。 分類には、種類が2つなら<a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.BinaryClassificationEvaluator.html?highlight=binaryclass#pyspark.ml.evaluation.BinaryClassificationEvaluator" target="_blank">BinaryClassificationEvaluator</a>、2つ以上なら<a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.MulticlassClassificationEvaluator.html?highlight=multiclass#pyspark.ml.evaluation.MulticlassClassificationEvaluator" target="_blank">MulticlassClassificationEvaluator</a>を使います。
# MAGIC 
# MAGIC **`areaUnderROC`** をメトリクスとする **`BinaryClassificationEvaluator`** を作成します。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic" target="_blank">ROC曲線について詳しくはこちら。</a>要するに、真陽性と偽陽性を比較するものです。

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md <i18n value="ea1c0e11-125d-4067-bd70-0bd6c7ca3cdb"/>
# MAGIC 
# MAGIC ## 交差検証 (Cross Validation)
# MAGIC 
# MAGIC 3-分割交差検証を行い、 **`parallelism`** を4とし、再現性のために **`seed`** を42に設定します。
# MAGIC 
# MAGIC ランダムフォレストを交差検証(CV)に入れ、 <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html?highlight=crossvalidator#pyspark.ml.tuning.CrossValidator" target="_blank">交差検証</a>を高速化します（CVのパイプラインと比べて）。

# COMMAND ----------

# TODO

from pyspark.ml.tuning import CrossValidator

cv = <FILL_IN>

# COMMAND ----------

# MAGIC %md <i18n value="1f8cebd5-673c-4513-b73b-b64b0a56297c"/>
# MAGIC 
# MAGIC #パイプライン (Pipeline)
# MAGIC 
# MAGIC 交差検証を使ったパイプラインをトレーニングデータにフィットさせます（数分かかるかもしれません）。

# COMMAND ----------

stages = [string_indexer, vec_assembler, cv]

pipeline = Pipeline(stages=stages)

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="70cdbfa3-0dd7-4f23-b755-afc0dadd7eb2"/>
# MAGIC 
# MAGIC ## ハイパーパラメーター (Hyperparameter)
# MAGIC 
# MAGIC どのハイパーパラメータの組み合わせで一番性能が良かったでしょうか？

# COMMAND ----------

cv_model = pipeline_model.stages[-1]
rf_model = cv_model.bestModel

# list(zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics))

print(rf_model.explainParams())

# COMMAND ----------

# MAGIC %md <i18n value="11e6c47a-ddb1-416d-92a5-2f61340f9a5e"/>
# MAGIC 
# MAGIC ## 特徴量の重要性 (Feature Importance)

# COMMAND ----------

import pandas as pd

pandas_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), rf_model.featureImportances)), columns=["feature", "importance"])
top_features = pandas_df.sort_values(["importance"], ascending=False)
top_features

# COMMAND ----------

# MAGIC %md <i18n value="ae7e312e-d32b-4b02-97ff-ad4d2c737892"/>
# MAGIC 
# MAGIC これらの特徴量は理にかなっていますか？Airbnbの部件を選ぶ際に、これらの特徴量を使いますか？

# COMMAND ----------

# MAGIC %md <i18n value="950eb40f-b1d2-4e7f-8b07-76faff6b8186"/>
# MAGIC 
# MAGIC ## テストデータにモデル適用 (Apply Model to test set)

# COMMAND ----------

# TODO

pred_df = <FILL_IN>
area_under_roc = <FILL_IN>
print(f"Area under ROC is {area_under_roc:.2f}")

# COMMAND ----------

# MAGIC %md <i18n value="01974668-f242-4b8a-ac80-adda3b98392d"/>
# MAGIC 
# MAGIC ## モデル保存 (Save Model)
# MAGIC 
# MAGIC モデルを **`DA.paths.working_dir`** (Classroom-Setupで定義された変数) に保存します。

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md <i18n value="f5fdf1a9-2a65-4252-aa76-18807dbb3a9d"/>
# MAGIC 
# MAGIC ## Sklearn vs SparkML
# MAGIC 
# MAGIC <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html" target="_blank">Sklearn RandomForestRegressor</a> と <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.RandomForestRegressor.html?highlight=randomfore#pyspark.ml.regression.RandomForestRegressor" target="_blank">SparkML RandomForestRegressor</a>の比較です。
# MAGIC 
# MAGIC 特にこれらのパラメータを見てください。
# MAGIC * **n_estimators** (sklearn) vs **numTrees** (SparkML)
# MAGIC * **max_depth** (sklearn) vs **maxDepth** (SparkML)
# MAGIC * **max_features** (sklearn) vs **featureSubsetStrategy** (SparkML)
# MAGIC * **maxBins** (SparkMLのみ)
# MAGIC 
# MAGIC 違うところに気づきましたか？

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
