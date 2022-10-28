# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="1108b110-983d-4034-9156-6b95c04dc62c"/>
# MAGIC 
# MAGIC # Spark Review
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)このレッスンでは次を行います:<br>
# MAGIC  - Spark DataFrameを作成する
# MAGIC  - Spark UIで分析する
# MAGIC  - データをキャッシュする
# MAGIC  - Pandas DataFrame と Spark DataFrame を行き来する

# COMMAND ----------

# MAGIC %md <i18n value="890d085b-9058-49a7-aa15-bff3649b9e05"/>
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/sparkcluster.png)

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="df081f79-6894-4174-a554-fa0943599408"/>
# MAGIC 
# MAGIC ## Spark DataFrame

# COMMAND ----------

from pyspark.sql.functions import col, rand

df = (spark.range(1, 1000000)
      .withColumn("id", (col("id") / 1000).cast("integer"))
      .withColumn("v", rand(seed=1)))

# COMMAND ----------

# MAGIC %md <i18n value="a0c6912d-a8d6-449b-a3ab-5ca91c7f9805"/>
# MAGIC 
# MAGIC なぜ、上のコマンドではSparkジョブが実行されなかったのでしょうか？データに「触れなかった」ので、Sparkはクラスタ全体で何も実行する必要がありませんでした。

# COMMAND ----------

display(df.sample(.001))

# COMMAND ----------

# MAGIC %md <i18n value="6eadef21-d75c-45ba-8d77-419d1ce0c06c"/>
# MAGIC 
# MAGIC ## Views
# MAGIC 
# MAGIC この作成したDataFrameにSQLでアクセスするにはどうすればよいでしょうか。

# COMMAND ----------

df.createOrReplaceTempView("df_temp")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM df_temp LIMIT 10

# COMMAND ----------

# MAGIC %md <i18n value="2593e6b0-d34b-4086-9fed-c4956575a623"/>
# MAGIC 
# MAGIC ## Count
# MAGIC 
# MAGIC レコードの数を見てみましょう。

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md <i18n value="5d00511e-15da-48e7-bd26-e89fbe56632c"/>
# MAGIC 
# MAGIC ## Spark UI
# MAGIC 
# MAGIC Spark UIを使ってみましょう - shuffle readとshuffle writeのフィールドはどうなっていますか？次のコマンドがヒントになるはずです。

# COMMAND ----------

df.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md <i18n value="50330454-0168-4f50-8355-0204632b20ec"/>
# MAGIC 
# MAGIC ## キャッシュ (Cache)
# MAGIC 
# MAGIC データに繰り返しアクセスする場合は、データをキャッシュすることでより高速になります。

# COMMAND ----------

df.cache().count()

# COMMAND ----------

# MAGIC %md <i18n value="7dd81880-1575-410c-a168-8ac081a97e9d"/>
# MAGIC 
# MAGIC ## Count を再実行 (Re-run Count)
# MAGIC 
# MAGIC すごい！こんなに速くなりました。

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md <i18n value="ce238b9e-fee4-4644-9469-b7d9910f6243"/>
# MAGIC 
# MAGIC ## データの収集 (Collect Data)
# MAGIC 
# MAGIC データをドライバに引き戻すとき（例えば、 **`.collect()`** , **` .toPandas()`** などを呼び出す場合）、ドライバに戻すデータの量に注意する必要があります。量が多すぎる場合、out of memory (OOM) 例外が発生する可能性があります。
# MAGIC 
# MAGIC ベストプラクティスは、データセットが小さいことが分かっている場合を除き、レコードの数を明示的に制限してから **`.collect()`** や **`.toPandas()`** を呼び出すことです.

# COMMAND ----------

df.limit(10).toPandas()

# COMMAND ----------

# MAGIC %md <i18n value="279e3325-b121-402b-a2d0-486e1cc26fc0"/>
# MAGIC 
# MAGIC ## <a href="https://www.youtube.com/watch?v=l6SuXvhorDY&feature=emb_logo" target="_blank">Spark 3.0</a> の新機能 (What's new in Spark 3.0)
# MAGIC 
# MAGIC * <a href="https://www.youtube.com/watch?v=jzrEc4r90N8&feature=emb_logo" target="_blank">Adaptive Query Execution (AQE)</a>
# MAGIC   * クエリーの実行中に統計情報を収集し、動的にクエリーの実行方法を最適化します
# MAGIC     * サイズの小さなシャッフルパーティションを動的に結合(coalesce)します
# MAGIC     * ジョイン戦略を動的に切り替えます
# MAGIC     * Skew ジョイン(一部のパーティションに偏って多くのデータがある場合(Skew)のジョイン)を動的に最適化します
# MAGIC   * 次のプロパティ設定を行います: **`spark.sql.adaptive.enabled=true`**
# MAGIC * Dynamic Partition Pruning (DPP)
# MAGIC   * 他のクエリの実行結果に基づいて、クエリに関係するデータを持たないパーティションのスキャンを避けることができます
# MAGIC * Joinのヒント
# MAGIC * <a href="https://www.youtube.com/watch?v=UZl0pHG-2HA&feature=emb_logo" target="_blank">Pandas UDFの改良</a>
# MAGIC   * 型のヒント
# MAGIC   * イテレータ
# MAGIC   * Pandas Function API (mapInPandas, applyInPandas など)
# MAGIC * その他多数上記リンク先の <a href="https://spark.apache.org/docs/latest/api/python/migration_guide/pyspark_2.4_to_3.0.html" target="_blank">移行ガイド</a> とリソースをご覧ください。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
