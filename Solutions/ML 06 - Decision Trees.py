# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="3bdc2b9e-9f58-4cb7-8c55-22bade9f79df"/>
# MAGIC 
# MAGIC # 決定木 (Decision Trees)
# MAGIC 
# MAGIC 前回のノートでは、パラメトリックモデルであるLinear Regression（線形回帰）を使っていました。線形回帰モデルでもっとハイパーパラメータを調整することもできますが、今回は木構造の手法を試して、パフォーマンスが向上するかどうかを確認します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このレッスンでは次を行います: <br>
# MAGIC  - シングルノードの決定木と分散型の決定木の実装の違いを確認する
# MAGIC  - 特徴量の重要度(importance)を取得する
# MAGIC  - 決定木のよくある落とし穴を検証する

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="9af16c65-168c-4078-985d-c5f8991f171f"/>
# MAGIC 
# MAGIC ## カテゴリ型特徴量をどう扱うか？ (How to Handle Categorical Features?)
# MAGIC 
# MAGIC 前回のノートブックで、StringIndexer/OneHotEncoder/VectorAssemblerやRFormulaが使えることを確認しました。
# MAGIC 
# MAGIC **しかし決定木、特にランダムフォレストでは、変数のOne Hot Encodingをすべきではありません。**
# MAGIC 
# MAGIC これについては、<a href="https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769#:~:text=One%2Dhot%20encoding%20categorical%20variables,importance%20resulting%20in%20poorer%20performance" target="_blank">ブログ</a> に詳しく説明があります。そのエッセンスは次です : 
# MAGIC >>> "基数の多い(high cardinality)カテゴリ変数にOHEを適用すると、ツリーベースの手法では非効率になることがあります。アルゴリズムにより、連続変数がダミー変数よりも重要視されるようになるため、特徴量の重要度の順序が不明瞭になり、パフォーマンスが低下する可能性があります。"

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

# COMMAND ----------

# MAGIC %md <i18n value="35e2f231-2ebb-4889-bc55-089200dd1605"/>
# MAGIC 
# MAGIC ## VectorAssembler
# MAGIC 
# MAGIC <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html?highlight=vectorassembler#pyspark.ml.feature.VectorAssembler" target="_blank">VectorAssembler</a> を使って、すべてのカテゴリ型および数値型の入力を結合してみましょう。

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Filter for just numeric columns (and exclude price, our label)
numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
# Combine output of StringIndexer defined above and numeric columns
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

# MAGIC %md <i18n value="2096f7aa-7fab-4807-b45f-fcbd0424a3e8"/>
# MAGIC 
# MAGIC ## 決定木 (Decision Tree)
# MAGIC 
# MAGIC では、デフォルトのハイパーパラメータで <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html?highlight=decisiontreeregressor#pyspark.ml.regression.DecisionTreeRegressor" target="_blank">DecisionTreeRegressor</a> を構築してみましょう。

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(labelCol="price")

# COMMAND ----------

# MAGIC %md <i18n value="506ab7fa-0952-4c55-ad9b-afefb6469380"/>
# MAGIC 
# MAGIC ## PipelineをFitする (Fit Pipeline)
# MAGIC 
# MAGIC 以下のセルはエラーになるはずです。後で修正します。

# COMMAND ----------

from pyspark.ml import Pipeline

# Combine stages into pipeline
stages = [string_indexer, vec_assembler, dt]
pipeline = Pipeline(stages=stages)

# Uncomment to perform fit
# pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="d0791ff8-8d79-4d32-937d-9fcfbac4e9bd"/>
# MAGIC 
# MAGIC ## maxBins
# MAGIC 
# MAGIC <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html?highlight=decisiontreeregressor#pyspark.ml.regression.DecisionTreeRegressor.maxBins" target="_blank">maxBins</a> は、どのようなパラメータでしょうか？ **`maxBins`** パラメータを説明するために、分散決定木の実装の1つであるPLANETの実装を見てみましょう。

# COMMAND ----------

# MAGIC %md <i18n value="1f9c229e-6f8c-4174-9927-c284e64e5753"/>
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/DistDecisionTrees.png" height=500px>

# COMMAND ----------

# MAGIC %md <i18n value="3b7e60c3-22de-4794-9cd4-6713255b79a4"/>
# MAGIC 
# MAGIC Sparkでは、データは行で分割されます。そのため、分割を行う必要がある場合、各Workerは分割点ごとに各特徴の要約統計量を計算する必要があります。そして、分割するためにこれらの統計情報を（tree reduceによって）集約する必要があります。 
# MAGIC 
# MAGIC 考えてみてください。Worker1が値 **`32`** を持っているが、他のどのWorkerもその値を持っていなかったとしたらどうなるでしょうか。どれだけ良い分割になるのかどうやって分かりますか。そこで、Sparkには連続変数を離散化してバケットにするためのmaxBinsパラメータを使います。しかし、バケット数は最も基数の多いカテゴリ型変数と同じ大きさでなければなりません。

# COMMAND ----------

# MAGIC %md <i18n value="0552ed6a-120f-4e49-ae3a-5f92bd9f863d"/>
# MAGIC 
# MAGIC では、maxBinsを **`40`** に増やしてみましょう。

# COMMAND ----------

dt.setMaxBins(40)

# COMMAND ----------

# MAGIC %md <i18n value="92252524-e388-439b-a92b-958cc332a861"/>
# MAGIC 
# MAGIC Pipelineをfitしましょう。

# COMMAND ----------

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="2426e78b-9bd2-4b7d-a65b-52054906e438"/>
# MAGIC 
# MAGIC ## 特徴量の重要度 (Feature Importance)
# MAGIC 
# MAGIC フィットした決定木モデルを取得し、特徴量の重要度を見ましょう。

# COMMAND ----------

dt_model = pipeline_model.stages[-1]
display(dt_model)

# COMMAND ----------

dt_model.featureImportances

# COMMAND ----------

# MAGIC %md <i18n value="823c20ff-f20b-4853-beb0-4b324debb2e6"/>
# MAGIC 
# MAGIC ## 特徴量の重要度の解釈 (Interpreting Feature Importance)
# MAGIC 
# MAGIC うーん。feature 4, feature 11のような表記は分かり難いです。特徴量の重要度スコアは小さい値なので、Pandasを使って元の列名を復元できるようにしましょう。

# COMMAND ----------

import pandas as pd

features_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), dt_model.featureImportances)), columns=["feature", "importance"])
features_df

# COMMAND ----------

# MAGIC %md <i18n value="1fe0f603-add5-4904-964b-7288ae98b2e8"/>
# MAGIC 
# MAGIC ## なぜわずかな特徴量だけがnon-zeroなのか？ (Why so few features are non-zero?)
# MAGIC 
# MAGIC SparkMLの場合、デフォルトの **`maxDepth`** は5なので、検討できる特徴量の数が限られています（同じ特徴量を異なる分割点で何度も分割することもあります）。
# MAGIC 
# MAGIC Databricksのwidgetを使って、top-K個の特徴量を取得してみましょう。

# COMMAND ----------

dbutils.widgets.text("top_k", "5")
top_k = int(dbutils.widgets.get("top_k"))

top_features = features_df.sort_values(["importance"], ascending=False)[:top_k]["feature"].values
print(top_features)

# COMMAND ----------

# MAGIC %md <i18n value="d9525bf7-b871-45c8-b0f9-dca5fd7ae825"/>
# MAGIC 
# MAGIC ## スケール不変性(Scale Invariant)
# MAGIC 
# MAGIC 決定木の場合、特徴量のスケールの大きさは問題にならない。例えば、分割点が100であっても、0.33に正規化されていても、データの1/3を分割する。重要なのは、その分割点から左右にいくつのデータが落ちるかだけで、分割点の絶対値ではありません。
# MAGIC 
# MAGIC これは線形回帰には当てはまらず、Sparkのデフォルトでは最初に正規化することになっています。考えてみてください。靴のサイズをアメリカ式とヨーロッパ式で測ると、同じ足のサイズに合わせた靴であっても、その値は大きく異なります。

# COMMAND ----------

# MAGIC %md <i18n value="bad0dd6d-05ba-484b-90d6-cfe16a1bc11e"/>
# MAGIC 
# MAGIC ## テストセットへのモデルの適用 (Apply model to test set)

# COMMAND ----------

pred_df = pipeline_model.transform(test_df)

display(pred_df.select("features", "price", "prediction").orderBy("price", ascending=False))

# COMMAND ----------

# MAGIC %md <i18n value="094553a3-10c0-4e08-9a58-f94430b4a512"/>
# MAGIC 
# MAGIC ## 落とし穴 (Pitfall)
# MAGIC 
# MAGIC Airbnbで巨大な物件を借りたらどうなるでしょう？20ベッドルームと20バスルームでした。このとき決定木は何を予測するのか？
# MAGIC 
# MAGIC 決定木は、学習に用いた値より大きな値を予測することができないです。トレーニングセットの最大値は1万ドルだったので、それ以上の値を予測することはできません。

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md <i18n value="033a9c19-0f9d-4c33-aa5e-f58665637448"/>
# MAGIC 
# MAGIC ## Uh oh!
# MAGIC 
# MAGIC このモデルは線形回帰モデルよりもずっと悪いし、平均値を予測するよりも悪いです。
# MAGIC 
# MAGIC 次のノートブックでは、ハイパーパラメータのチューニングとアンサンブルモデルを使って、単一の決定木の性能を向上させる方法を説明します。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
