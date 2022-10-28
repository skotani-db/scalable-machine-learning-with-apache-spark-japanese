# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="1fa7a9c8-3dad-454e-b7ac-555020a4bda8"/>
# MAGIC 
# MAGIC # Hyperopt
# MAGIC 
# MAGIC Hyperoptは、「実数値、離散、条件付き次元を含む、厄介な探索空間上でのシリアルおよびパラレル最適化」のためのPythonライブラリです。
# MAGIC 
# MAGIC 機械学習ワークフローにおいて、hyperoptは、他のライブラリで利用可能なものより高度な最適化戦略を用いてハイパーパラメータ最適化プロセスを分散/並列化するために使用することができます。
# MAGIC 
# MAGIC Apache Sparkでhyperoptをスケールさせるには、2つの方法があります。
# MAGIC * シングルマシンのhyperoptで、分散学習アルゴリズム（MLlibなど）を使う 
# MAGIC * 分散hyperoptで、SparkTrialsクラスと一緒にシングルマシンの学習アルゴリズム（scikit-learnなど）を使う。 
# MAGIC 
# MAGIC このレッスンでは、シングルマシンのhyperoptでMLlibを使用しますが、ラボでは、分散hyperoptでシングルノードモデルのハイパーパラメータチューニングを使用する方法を紹介します。 
# MAGIC 
# MAGIC 残念ながら現時点では、hyperoptを使用して分散型の学習アルゴリズムとともにハイパーパラメータ最適化を分散させることはできません。しかし、Spark MLを使ってより高度なハイパーパラメータ探索アルゴリズム（ランダム探索、TPEなど）を使用する利点があります。
# MAGIC 
# MAGIC 
# MAGIC リソース
# MAGIC 
# MAGIC 0. <a href="http://hyperopt.github.io/hyperopt/scaleout/spark/" target="_blank">Documentation</a>
# MAGIC 0. <a href="https://docs.databricks.com/applications/machine-learning/automl/hyperopt/index.html" target="_blank">Hyperopt on Databricks</a>
# MAGIC 0. <a href="https://databricks.com/blog/2019/06/07/hyperparameter-tuning-with-mlflow-apache-spark-mllib-and-hyperopt.html" target="_blank">Hyperparameter Tuning with MLflow, Apache Spark MLlib and Hyperopt</a>
# MAGIC 0. <a href="https://databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html" target="_blank">How (Not) to Tune Your Model With Hyperopt</a>
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このレッスンで次を行います : <br>
# MAGIC  - TPEを使用してMLlibモデルの最適なパラメータを見つけるためにhyperoptを使用します。

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="2340cdf4-9753-41b4-a613-043b90f0f472"/>
# MAGIC 
# MAGIC まずはSF Airbnb Datasetをロードしてみましょう。

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, val_df, test_df = airbnb_df.randomSplit([.6, .2, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="37bbd5bd-f330-4d02-8af6-1b185612cdf8"/>
# MAGIC 
# MAGIC その後、ランダムフォレストパイプラインと回帰のevaluatorを作成します。

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

rf = RandomForestRegressor(labelCol="price", maxBins=40, seed=42)
pipeline = Pipeline(stages=[string_indexer, vec_assembler, rf])
regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price")

# COMMAND ----------

# MAGIC %md <i18n value="e4627900-f2a5-4f65-881e-1374187dd4f9"/>
# MAGIC 
# MAGIC 次に、ワークフローのhyperopt部分を作ります。
# MAGIC 
# MAGIC まず、**目的関数**を定義します。目的関数は、主に2つの要件を持っています: 
# MAGIC 
# MAGIC 1. **入力** **`params`** は、モデルの学習に使用するハイパーパラメータの値を含みます。
# MAGIC 2. **出力** は、最適化するための損失(loss)メトリックを含みます。
# MAGIC 
# MAGIC ここでは **`max_depth`** と **`num_trees`** を指定し、損失指標としてRMSEを返すようにしています。
# MAGIC 
# MAGIC 指定したハイパーパラメータ値を使用するように、 **`RandomForestRegressor`** のパイプラインを再構築します。

# COMMAND ----------

def objective_function(params):    
    # set the hyperparameters that we want to tune
    max_depth = params["max_depth"]
    num_trees = params["num_trees"]

    with mlflow.start_run():
        estimator = pipeline.copy({rf.maxDepth: max_depth, rf.numTrees: num_trees})
        model = estimator.fit(train_df)

        preds = model.transform(val_df)
        rmse = regression_evaluator.evaluate(preds)
        mlflow.log_metric("rmse", rmse)

    return rmse

# COMMAND ----------

# MAGIC %md <i18n value="d4f9dd2b-060b-4eef-8164-442b2be242f4"/>
# MAGIC 
# MAGIC 次に、ハイパーパラメータの探索空間を定義します。 
# MAGIC 
# MAGIC これはグリッドサーチ処理におけるパラメータグリッドと同様です。ただし、テストする個々の具体的な値ではなく、値の範囲を指定します。実際の値を選択するのは、hyperoptの最適化アルゴリズムに任されています。
# MAGIC 
# MAGIC 検索空間を定義するのに役立つヒントについては <a href="https://github.com/hyperopt/hyperopt/wiki/FMin" target="_blank">ドキュメント</a> を参照してください。

# COMMAND ----------

from hyperopt import hp

search_space = {
    "max_depth": hp.quniform("max_depth", 2, 5, 1),
    "num_trees": hp.quniform("num_trees", 10, 100, 1)
}

# COMMAND ----------

# MAGIC %md <i18n value="27891521-e481-4734-b21c-b2c5fe1f01fe"/>
# MAGIC 
# MAGIC **`fmin()`** が、 **`目的関数`** に使用する新しいハイパーパラメータの構成を生成します。 以下では最大4つのモデルを評価します(変数で指定)。その際に、前のモデルから得た情報を使って、次に試すべきハイパーパラメータを効果的に決定することができます。 
# MAGIC 
# MAGIC Hyperoptでは、ランダムサーチまたはTree of Parzen Estimators（TPE）を用いて、ハイパーパラメータのチューニングを並行して行うことができます。以下のセルで、 **`tpe`** をインポートしていることに注意してください。.<a href="http://hyperopt.github.io/hyperopt/scaleout/spark/" target="_blank">ドキュメント</a> によると、TPEは以下のような適応的なアルゴリズムです。 
# MAGIC 
# MAGIC > ハイパーパラメータ空間を繰り返し探索します。テストされる新しいハイパーパラメータ設定は、過去の結果に基づいて選択される。 
# MAGIC 
# MAGIC このため **`tpe.suggest`** はベイジアンの探索方法です。
# MAGIC 
# MAGIC MLflowはHyperoptと統合されているため、ハイパーパラメータのチューニングの一環として、学習させたすべてのモデルの結果とその結果を追跡することができます。このノートブックのMLflowの実験を追跡することができますが、このノートブック以外の実験を指定することもできます。

# COMMAND ----------

from hyperopt import fmin, tpe, Trials
import numpy as np
import mlflow
import mlflow.spark
mlflow.pyspark.ml.autolog(log_models=False)

num_evals = 4
trials = Trials()
best_hyperparam = fmin(fn=objective_function, 
                       space=search_space,
                       algo=tpe.suggest, 
                       max_evals=num_evals,
                       trials=trials,
                       rstate=np.random.default_rng(42))

# Retrain model on train & validation dataset and evaluate on test dataset
with mlflow.start_run():
    best_max_depth = best_hyperparam["max_depth"]
    best_num_trees = best_hyperparam["num_trees"]
    estimator = pipeline.copy({rf.maxDepth: best_max_depth, rf.numTrees: best_num_trees})
    combined_df = train_df.union(val_df) # Combine train & validation together

    pipeline_model = estimator.fit(combined_df)
    pred_df = pipeline_model.transform(test_df)
    rmse = regression_evaluator.evaluate(pred_df)

    # Log param and metrics for the final model
    mlflow.log_param("maxDepth", best_max_depth)
    mlflow.log_param("numTrees", best_num_trees)
    mlflow.log_metric("rmse", rmse)
    mlflow.spark.log_model(pipeline_model, "model")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
