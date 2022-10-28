# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="7051c998-fa70-4ff4-8c4b-439030503fb8"/>
# MAGIC 
# MAGIC # データ探索 (Data Exploration)
# MAGIC 
# MAGIC このノートブックでは、前回のラボでクレンジングしたデータセットを使って、探索的データ解析（EDA）を行います。
# MAGIC 
# MAGIC これにより、より良いモデルを作るために、データの理解を深めることができます。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは以下を行います。<br>
# MAGIC  - 対数正規分布の判別
# MAGIC  - ベースラインモデルの構築と評価

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="af8bcd70-9430-4470-95b0-2fcff94ed149"/>
# MAGIC 
# MAGIC データセットの80%をトレーニングセット、20%をテストセットにします。ここでは、<a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html?highlight=randomsplit#pyspark.sql.DataFrame.randomSplit" target="_blank">randomSplit</a> メソッドを使用します。
# MAGIC 
# MAGIC トレーニングとテストの分割については後で詳しく説明しますが、このノートブックでは、 **`train_df`** でデータ探索を行ってください。

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_df = spark.read.format("delta").load(file_path)
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="d4fed64b-d7ad-4426-805e-192854e1471c"/>
# MAGIC 
# MAGIC price列のヒストグラムを作って探索しましょう（ビン数を300に変更）。

# COMMAND ----------

display(train_df.select("price"))

# COMMAND ----------

# MAGIC %md <i18n value="f9d67fce-097f-40fd-9261-0ec1a5acd12a"/>
# MAGIC 
# MAGIC これは<a href="https://en.wikipedia.org/wiki/Log-normal_distribution" target="_blank">対数正規</a>分布でしょうか？価格の **`log`** を取り、ヒストグラムを確認します。これは後々のために覚えておいてください。 :)

# COMMAND ----------

# TODO

display(<FILL_IN>)

# COMMAND ----------

# MAGIC %md <i18n value="9a834ac8-878c-4d6e-b685-d0d37f148830"/>
# MAGIC 
# MAGIC 次に、 **`price`** がいくつかの変数にどのように依存するかを見てみましょう。
# MAGIC * **`価格`** vs **`ベッドルーム数`** 
# MAGIC * **`価格`** vs **`宿泊施設`** 
# MAGIC 
# MAGIC 集計対象を必ず **`AVG`** に変更してください。

# COMMAND ----------

display(train_df)

# COMMAND ----------

# MAGIC %md <i18n value="6694a7c9-9258-4b67-a403-4d72898994fa"/>
# MAGIC 
# MAGIC カテゴリー別の特徴量の分布を見てみましょう。

# COMMAND ----------

display(train_df.groupBy("room_type").count())

# COMMAND ----------

# MAGIC %md <i18n value="cbb15ce5-15f0-488c-a0a9-f282d7460b40"/>
# MAGIC 
# MAGIC レンタルが最も多いneighbourhood(地域)はどこですか？neighbourhoodとその関連カウントを降順で表示します。

# COMMAND ----------

# TODO
display(<FILL_IN>)

# COMMAND ----------

# MAGIC %md <i18n value="f72931f3-0d1d-4721-b5af-1da14da2d60d"/>
# MAGIC 
# MAGIC #### 場所によって値段はどのくらい違うのですか？
# MAGIC 
# MAGIC displayHTML を使って、任意の HTML、CSS、JavaScript コードをレンダリング(rendering)することができます。

# COMMAND ----------

from pyspark.sql.functions import col

lat_long_price_values = train_df.select(col("latitude"), col("longitude"), col("price")/600).collect()

lat_long_price_strings = [f"[{lat}, {long}, {price}]" for lat, long, price in lat_long_price_values]

v = ",\n".join(lat_long_price_strings)

# DO NOT worry about what this HTML code is doing! We took it from Stack Overflow :-)
displayHTML("""
<html>
<head>
 <link rel="stylesheet" href="https://unpkg.com/leaflet@1.3.1/dist/leaflet.css"
   integrity="sha512-Rksm5RenBEKSKFjgI3a41vrjkw4EVPlJ3+OiI65vTjIdo9brlAacEuKOiQ5OFh7cOI1bkDwLqdLw3Zg0cRJAAQ=="
   crossorigin=""/>
 <script src="https://unpkg.com/leaflet@1.3.1/dist/leaflet.js"
   integrity="sha512-/Nsx9X4HebavoBvEBuyp3I7od5tA0UzAxs+j83KgC8PU0kgB4XiK4Lfe4y4cgBtaRJQEIFCW+oC506aPT2L1zw=="
   crossorigin=""></script>
 <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.heat/0.2.0/leaflet-heat.js"></script>
</head>
<body>
    <div id="mapid" style="width:700px; height:500px"></div>
  <script>
  var mymap = L.map('mapid').setView([37.7587,-122.4486], 12);
  var tiles = L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors',
}).addTo(mymap);
  var heat = L.heatLayer([""" + v + """], {radius: 25}).addTo(mymap);
  </script>
  </body>
  </html>
""")

# COMMAND ----------

# MAGIC %md <i18n value="f8bcc454-6e0a-4b5d-a9f7-4917f3a66553"/>
# MAGIC 
# MAGIC ## ベースラインモデル (Baseline Model)
# MAGIC 
# MAGIC 機械学習モデルを構築する前に、比較するためのベースラインモデルを構築します。また、モデルを評価するための指標も決めます。ここでは、RMSEを使用してみましょう。
# MAGIC 
# MAGIC このデータセットを使って、常に平均価格を予測するベースラインモデルと、常に<a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.approxQuantile.html?highlight=approxquantile#pyspark.sql.DataFrame.approxQuantile" target="_blank">中央値</a>を予測するモデルを構築し、その結果を確認しましょう。これを2段階に分けて行う。
# MAGIC 
# MAGIC 0. **`train_df`** :**`train_df`** から平均値と中央値を抽出し、それぞれを変数 **`avg_price`** と **`median_price`** に格納します。
# MAGIC 0. **`test_df`** :**`avgPrediction`** と **`medianPrediction`** という2つの列を追加し、それぞれ **`train_df`** からの平均値と中央値を格納します。結果DataFrameの名前を **`pred_df`** にします。
# MAGIC 
# MAGIC 便利な機能をいくつかご紹介します。
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.avg.html?highlight=avg#pyspark.sql.functions.avg" target="_blank">avg()</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.col.html?highlight=col#pyspark.sql.functions.col" target="_blank">col()</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.lit.html?highlight=lit#pyspark.sql.functions.lit" target="_blank">lit()</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.approxQuantile.html?highlight=approxquantile#pyspark.sql.DataFrame.approxQuantile" target="_blank">approxQuantile()</a> **ヒント**:中央値関数がないので、approxQuantileを使用する必要があります。
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.withColumn.html?highlight=withcolumn#pyspark.sql.DataFrame.withColumn" target="_blank">withColumn()</a>

# COMMAND ----------

# TODO

# COMMAND ----------

# MAGIC %md <i18n value="37f17c9d-94b7-48f4-b579-352edab84703"/>
# MAGIC 
# MAGIC ## モデル評価 (Evaluate model)
# MAGIC 
# MAGIC SparkMLの<a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RegressionEvaluator.html?highlight=regressionevaluator#pyspark.ml.evaluation.RegressionEvaluator" target="_blank">RegressionEvaluator</a> を使って、<a href="https://en.wikipedia.org/wiki/Root-mean-square_deviation" target="_blank">root mean square error（RMSE）</a>を計算します。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png"> 評価指標RMSEの詳細については、SparkMLの評価器と合わせて次のレッスンで説明します。とりあえず、RMSEは平均的に予測値がどれだけ真値から外れているかを数値化したものだと理解しておけばよいです。

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_mean_evaluator = RegressionEvaluator(predictionCol="avgPrediction", labelCol="price", metricName="rmse")
print(f"The RMSE for predicting the average price is: {regression_mean_evaluator.evaluate(pred_df)}")

regressionMedianEvaluator = RegressionEvaluator(predictionCol="medianPrediction", labelCol="price", metricName="rmse")
print(f"The RMSE for predicting the median price is: {regressionMedianEvaluator.evaluate(pred_df)}")

# COMMAND ----------

# MAGIC %md <i18n value="a2158c71-c343-4ee6-a1e4-4c6f8dd1792c"/>
# MAGIC 
# MAGIC おお！今回のデータセットでは、中央値や平均値を常に予測することはあまりうまくいかないことがわかります。機械学習モデルで改善できるかどうか、見てみましょう。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
