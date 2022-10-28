# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="60a5d18a-6438-4ee3-9097-5145dc31d938"/>
# MAGIC 
# MAGIC # 線形回帰モデルの改善 (Linear Regression: Improving our model)
# MAGIC 
# MAGIC このノートブックでは、モデルに特徴量を追加します。また、カテゴリ型特徴量の扱い方について説明します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このレッスンで次を行います: <br>
# MAGIC  - カテゴリカル変数のOne Hot Encode
# MAGIC  - Pipeline APIの使用
# MAGIC  - モデルの保存と読込

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)

# COMMAND ----------

# MAGIC %md <i18n value="f8b3c675-f8ce-4339-865e-9c64f05291a6"/>
# MAGIC 
# MAGIC ## トレーニングデータ  / テストデータの分割 (Train/Test Split)
# MAGIC 
# MAGIC 前のノートブックと同様に80/20分割をします。フェアな比較ができるように同じシード値を使用して行います（※クラスタの設定を変更しない限り！）。

# COMMAND ----------

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="09003d63-70c1-4fb7-a4b7-306101a88ae3"/>
# MAGIC 
# MAGIC ## カテゴリカル変数 (Categorical Variables)
# MAGIC 
# MAGIC カテゴリ型特徴量を扱うにはいくつかの方法があります。
# MAGIC * 数値の値の割当 
# MAGIC * ダミー変数の作成（One Hot Encodingとも呼ばれる） 
# MAGIC * Embeddingの作成（主にテキストデータで利用される）
# MAGIC 
# MAGIC ### One Hot Encoder
# MAGIC カテゴリ型変数にOne Hot Encode（OHE）を適用します。Sparkには **`dummies`** 関数はありません。OHEを2ステップで行います。まず <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html?highlight=stringindexer#pyspark.ml.feature.StringIndexer" target="_blank">StringIndexer</a> を使って、文字列型のカラムのラベル値を、MLで使うカラムのラベルインデックス値に対応付ける必要があります。
# MAGIC 
# MAGIC 次に、StringIndexerの出力に <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html?highlight=onehotencoder#pyspark.ml.feature.OneHotEncoder" target="_blank">OneHotEncoder</a> を適用します。

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
ohe_output_cols = [x + "OHE" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")
ohe_encoder = OneHotEncoder(inputCols=index_output_cols, outputCols=ohe_output_cols)

# COMMAND ----------

# MAGIC %md <i18n value="dedd7980-1c27-4f35-9d94-b0f1a1f92839"/>
# MAGIC 
# MAGIC ## Vector Assembler
# MAGIC 
# MAGIC OHEされたカテゴリ型特徴量と数値の特徴量を組み合わせることができます。

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = ohe_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

# MAGIC %md <i18n value="fb06fb9b-5dac-46df-aff3-ddee6dc88125"/>
# MAGIC 
# MAGIC ## 線形回帰 (Linear Regression)
# MAGIC 
# MAGIC 全ての特徴量が揃ったので、線形回帰モデルを構築してみましょう。

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="price", featuresCol="features")

# COMMAND ----------

# MAGIC %md <i18n value="a7aabdd1-b384-45fc-bff2-f385cc7fe4ac"/>
# MAGIC 
# MAGIC ## パイプライン (Pipeline)
# MAGIC 
# MAGIC これら全てのステージをパイプラインにまとめてみましょう。 <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html?highlight=pipeline#pyspark.ml.Pipeline" target="_blank">パイプライン</a> は、変換器(tarnsformer)と推定器(estimator)をまとめて実行する方法です。
# MAGIC 
# MAGIC 処理内容をパイプラインにまとめておくことで、テストデータの処理にも同じパイプラインを再利用できます。

# COMMAND ----------

from pyspark.ml import Pipeline

stages = [string_indexer, ohe_encoder, vec_assembler, lr]
pipeline = Pipeline(stages=stages)

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="c7420125-24be-464f-b609-1bb4e765d4ff"/>
# MAGIC 
# MAGIC ## モデルの保存 (Saving Models)
# MAGIC 
# MAGIC クラスタがダウンした場合に備えて、モデルを永続的なストレージ（DBFSなど）に保存しておけば、結果を再計算する必要がありません。

# COMMAND ----------

pipeline_model.write().overwrite().save(DA.paths.working_dir)

# COMMAND ----------

# MAGIC %md <i18n value="15f4623d-d99a-42d6-bee8-d7c4f79fdecb"/>
# MAGIC 
# MAGIC ## モデルのロード (Loading models)
# MAGIC 
# MAGIC モデルをロードする際、モデルの種類（線形回帰モデルだったのか、ロジスティック回帰モデルだったのか）を知る必要があります。
# MAGIC 
# MAGIC このため、変換器(transformer)や推定器(estimator)を常にパイプラインに配置し、汎用的なPipelineModelをロードすることをお勧めします。

# COMMAND ----------

from pyspark.ml import PipelineModel

saved_pipeline_model = PipelineModel.load(DA.paths.working_dir)

# COMMAND ----------

# MAGIC %md <i18n value="1303ef7d-1a57-4573-8afe-561f7730eb33"/>
# MAGIC 
# MAGIC ## テストデータへのモデルの適用 (Apply model to test set)

# COMMAND ----------

pred_df = saved_pipeline_model.transform(test_df)

display(pred_df.select("features", "price", "prediction"))

# COMMAND ----------

# MAGIC %md <i18n value="9497f680-1c61-4bf1-8ab4-e36af502268d"/>
# MAGIC 
# MAGIC ## モデルの評価 (Evaluate model)
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/r2d2.jpg) R2の結果はどうですか？

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md <i18n value="cc0618e0-59d9-4a6d-bb90-a7945da1457e"/>
# MAGIC 
# MAGIC 見た通り、OHEを行わないモデルと比較して、RMSEが低くなったし、R2が高くなりました。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
