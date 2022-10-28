# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="0f0f211a-70ba-4432-ab87-19bf7c8fc6cc"/>
# MAGIC 
# MAGIC # AutoML ラボ (AutoML Lab)
# MAGIC 
# MAGIC <a href="https://docs.databricks.com/applications/machine-learning/automl.html" target="_blank">Databricks AutoML</a>は、UIとプログラムのいずれかの方法で機械学習モデルを自動的に構築することができます。モデルのトレーニングのためにデータセットを準備し、（HyperOptを使用して）複数のモデルを作成、チューニング、評価する一連の試行を実行し記録します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、次のことを行います。<br>
# MAGIC  - AutoMLを使用してモデルを自動的にトレーニングおよびチューニングする
# MAGIC  - PythonとUIでAutoMLを実行する
# MAGIC  - AutoMLの実行結果を解釈する

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="9178a13f-aeaf-49cb-ac27-600e7cea0077"/>
# MAGIC 
# MAGIC 現在、AutoMLはXGBoostとsklearn（シングルノードモデルのみ）を組み合わせて使用していますが、それぞれの中で最適なハイパーパラメータをチューニングします。

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="af913436-4be4-4a26-8381-d40d4e1af9d2"/>
# MAGIC 
# MAGIC ### UI使用 (Use the UI)
# MAGIC 
# MAGIC プログラムでモデルを構築する代わりに、UIを利用することも可能です。しかし、その前にデータセットをテーブルとして登録する必要があります。

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DA.cleaned_username}")
train_df.write.mode("overwrite").saveAsTable(f"{DA.cleaned_username}.autoMLTable")

# COMMAND ----------

# MAGIC %md <i18n value="2f854d06-800c-428c-8add-aece6c9a91b6"/>
# MAGIC 
# MAGIC まず、左側のナビゲーターから機械学習(Machine Learning)ペルソナを選択していることを確認してから、ワークスペースのトップページでAutoMLを開始してください。
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/301/AutoML_1_2.png" alt="step12" width="750"/>

# COMMAND ----------

# MAGIC %md <i18n value="98f64ede-5b15-442b-8346-874e0fdea6b5"/>
# MAGIC 
# MAGIC 機械学習の問題のタイプに **`regression`** を選択し、前のセルで作成したデータセットのテーブルも選択します。次に、予測ターゲットに **`price`** を選択します。
# MAGIC 
# MAGIC <img src="http://files.training.databricks.com/images/301/AutoML_UI.png" alt="ui" width="750"/>

# COMMAND ----------

# MAGIC %md <i18n value="4e561687-2509-4084-bd33-4221cb047eba"/>
# MAGIC 
# MAGIC 高度な設定(Advanced Configuration)を展開して、評価メトリクスを「rmse」に、タイムアウトを5分に、試行回数を20回に設定します。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/301/AutoML_Advanced.png" alt="advanced" width="500"/>.

# COMMAND ----------

# MAGIC %md <i18n value="b15305f8-04cd-422f-a1da-ad7640b3846b"/>
# MAGIC 
# MAGIC これで実行を開始することができます。完了したら、ベストモデルをクリックするとチューニングされたモデルを確認することができます。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/301/AutoMLResultsUpdated.png" alt="results" width="1000"/>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
