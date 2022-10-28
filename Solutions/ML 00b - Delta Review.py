# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="fd2d84ac-6a17-44c2-bb92-18b0c7fef797"/>
# MAGIC 
# MAGIC # デルタ・レビュー (Delta Review)
# MAGIC 
# MAGIC <a href="https://docs.delta.io/latest/quick-start.html#create-a-table" target="_blank">Delta Lake</a>を理解し活用するために、いくつかの重要な操作を確認します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは次を行います。<br>
# MAGIC - Delta Tableを作成する。
# MAGIC - Delta Tableからデータを読込む。
# MAGIC - Delta Tableのデータを更新する。
# MAGIC - <a href="https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html" target="_blank">タイムトラベル</a>を使用して、Delta Tableの以前のバージョンにアクセスする。
# MAGIC - <a href="https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html" target="_blank">トランザクションログを理解する。</a>
# MAGIC 
# MAGIC このノートブックでは、<a href="http://insideairbnb.com/get-the-data.html" target="_blank">Inside Airbnb</a>のSF Airbnbレンタルデータセットを使用します。

# COMMAND ----------

# MAGIC %md <i18n value="68fcecd4-2280-411c-94c1-3e111683c6a3"/>
# MAGIC 
# MAGIC ###なぜDelta Lakeなのか？ (Why Delta Lake?)<br><br>
# MAGIC 
# MAGIC <div style="img align: center; line-height:0; padding-top:9px;">
# MAGIC   <img src="https://user-images.githubusercontent.com/20408077/87175470-4d8e1580-c29e-11ea-8f33-0ee14348a2c1.png" width="500"/>
# MAGIC </div>
# MAGIC 
# MAGIC 一言で言えば、Delta Lakeは、データレイクに**信頼性とパフォーマンス**をもたらす、オープンソースのストレージレイヤーです。Delta Lakeは、ACIDトランザクション、スケーラブルなメタデータ処理を提供し、ストリーミングとバッチデータを統一的に処理します。
# MAGIC 
# MAGIC Delta Lakeは既存のデータレイク上で動作し、Apache Spark APIと完全な互換性があります。 <a href="https://docs.databricks.com/delta/delta-intro.html" target="_blank">詳細をご参照ください。 </a>

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="8ce92b68-6e6c-4fd0-8d3c-a57f27e5bdd9"/>
# MAGIC 
# MAGIC ###デルタテーブルの作成 (Creating a Delta Table)
# MAGIC まず、AirbnbのデータセットをSpark DataFrameとして読み込む必要があります。

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.parquet/"
airbnb_df = spark.read.format("parquet").load(file_path)

display(airbnb_df)

# COMMAND ----------

# MAGIC %md <i18n value="c100b529-ac6b-4540-a3ff-4afa63577eee"/>
# MAGIC 
# MAGIC 以下のセルは、Spark DataFrameが提供するスキーマを使用して、データをDelta tableに変換します。

# COMMAND ----------

# Converting Spark DataFrame to Delta Table
dbutils.fs.rm(DA.paths.working_dir, True)
airbnb_df.write.format("delta").mode("overwrite").save(DA.paths.working_dir)

# COMMAND ----------

# MAGIC %md <i18n value="090a31f6-1082-44cf-8e2a-6c659ea796ea"/>
# MAGIC 
# MAGIC Deltaディレクトリは、メタストアのテーブルとして登録することも可能です。

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {DA.cleaned_username}")
spark.sql(f"USE {DA.cleaned_username}")

airbnb_df.write.format("delta").mode("overwrite").saveAsTable("delta_review")

# COMMAND ----------

# MAGIC %md <i18n value="732577c2-095d-4278-8466-74e494a9c1bd"/>
# MAGIC 
# MAGIC デルタはパーティションに対応しています。パーティショニングを行うと、パーティショニングされたカラムの値が同じデータをその値に対応するディレクトリに置きます。パーティションされたカラムにフィルターをかけると、そのフィルターに一致するディレクトリのみを読み込みます。この最適化をパーティション・プルーニングと呼びます。データのパターンに基づいてパーティションカラムを選択します。例えば、このデータセットでは、地域別にパーティションを設定すると後の処理に役立つかもしれません。

# COMMAND ----------

airbnb_df.write.format("delta").mode("overwrite").partitionBy("neighbourhood_cleansed").option("overwriteSchema", "true").save(DA.paths.working_dir)

# COMMAND ----------

# MAGIC %md <i18n value="e9ce863b-5761-4676-ae0b-95f3f5f027f6"/>
# MAGIC 
# MAGIC ###<a href="https://databricks.com/blog/2019/08/21/diving-into-delta-lake-unpacking-the-transaction-log.html" target="_blank">Transaction Log </a>の理解 (Understanding the Transaction Log)
# MAGIC 
# MAGIC Delta Transaction Logを見てみましょう。Deltaが地域別のパーティションを別々のファイルに保存していることがわかります。さらに、_delta_logというディレクトリも存在しています。

# COMMAND ----------

display(dbutils.fs.ls(DA.paths.working_dir))

# COMMAND ----------

# MAGIC %md <i18n value="ac970bba-1cf6-4aa3-91bb-74a797496eef"/>
# MAGIC 
# MAGIC <div style="img align: center; line-height:0; padding-top:9px;">
# MAGIC   <img src="https://user-images.githubusercontent.com/20408077/87174138-609fe600-c29c-11ea-90cc-84df0c1357f1.png" width="500"/>
# MAGIC </div>
# MAGIC 
# MAGIC ユーザーがDelta Lakeテーブルを作成すると、そのテーブルのトランザクションログは自動的に_delta_logというサブディレクトリに作成されます。そのテーブルを変更すると、全ての変更はトランザクションログとして、順序付きでアトミックにコミットとして記録されます。各コミットは、000000.jsonで始まるJSONファイルとして書き出されます。テーブルを追加変更すると、さらにJSONファイルが生成されます。

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/_delta_log/"))

# COMMAND ----------

# MAGIC %md <i18n value="2905b874-373b-493d-9084-8ff4f7583ccc"/>
# MAGIC 
# MAGIC 次に、トランザクション・ログ・ファイルを見てみましょう。
# MAGIC 
# MAGIC <a href="https://docs.databricks.com/delta/delta-utility.html" target="_blank">4つの列</a>はそれぞれ、テーブルが作成されたDeltaテーブルへの最初のコミットの異なる部分を表します。<br><br>
# MAGIC 
# MAGIC - **add**列には、DataFrame 全体および個々の列に関する統計情報が記載されます。
# MAGIC - **commitInfo**列には、操作の内容（WRITEまたはREAD）と、誰がその操作を実行したかについての有用な情報が記載されます。
# MAGIC - **metaData**列には、カラムのスキーマに関する情報が記載されます。
# MAGIC - protocal versionには、このデルタテーブルへの書き込みまたは読み出しに必要な最小デルタバージョンに関する情報が記載されます。

# COMMAND ----------

display(spark.read.json(f"{DA.paths.working_dir}/_delta_log/00000000000000000000.json"))

# COMMAND ----------

# MAGIC %md <i18n value="8f79d1df-d777-4364-9783-b52bc0eed81a"/>
# MAGIC 
# MAGIC 2つ目のトランザクション・ログには39行のデータがあります。これには、各パーティションのメタデータが含まれます。

# COMMAND ----------

display(spark.read.json(f"{DA.paths.working_dir}/_delta_log/00000000000000000001.json"))

# COMMAND ----------

# MAGIC %md <i18n value="18500df8-b905-4f24-957c-58040920d554"/>
# MAGIC 
# MAGIC 最後に、Neighborhoodパーティションの1つについて、中のファイルを見てみましょう。中のファイルは、_delta_logディレクトリのパーティションコミット（ファイル01）に対応します。

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md <i18n value="9f817cd0-87ec-457b-8776-3fc275521868"/>
# MAGIC 
# MAGIC ### Deltaテーブルからデータの読込 (Reading data from your Delta table)

# COMMAND ----------

df = spark.read.format("delta").load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="faba817b-7cbf-49d4-a32c-36a40f582021"/>
# MAGIC 
# MAGIC #デルタテーブルの更新 (Updating your Delta Table)
# MAGIC 
# MAGIC ホストがスーパーホストである行をフィルタリングしてみましょう。

# COMMAND ----------

df_update = airbnb_df.filter(airbnb_df["host_is_superhost"] == "t")
display(df_update)

# COMMAND ----------

df_update.write.format("delta").mode("overwrite").save(DA.paths.working_dir)

# COMMAND ----------

df = spark.read.format("delta").load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="e4cafdf4-a346-4729-81a6-fdea70f4929a"/>
# MAGIC 
# MAGIC 更新後のBayviewパーティションのファイルを見てみましょう。このディレクトリ内のファイルは、異なるコミットに対応するDataFrameのスナップショットであることをご注意ください。

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md <i18n value="25ca7489-8077-4b23-96af-8d801982367c"/>
# MAGIC 
# MAGIC #Delta Time Travel

# COMMAND ----------

# MAGIC %md <i18n value="c6f2e771-502d-46ed-b8d4-b02e3e4f4134"/>
# MAGIC 
# MAGIC おっと、実は全データセットが必要です!<a href="https://databricks.com/blog/2019/02/04/introducing-delta-time-travel-for-large-scale-data-lakes.html" target="_blank">Delta Time Travel</a>を使用して、デルタテーブルの以前のバージョンにアクセスすることができます。バージョン履歴にアクセスするには、次の2つのセルを使用します。Delta Lakeは、デフォルトで30日間のバージョン履歴を保持しますが、必要であればより長い履歴を保持することもできます。

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS train_delta;
# MAGIC CREATE TABLE train_delta USING DELTA LOCATION '${DA.paths.working_dir}'

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY train_delta

# COMMAND ----------

# MAGIC %md <i18n value="61faa23f-d940-479c-95fe-5aba72c29ddf"/>
# MAGIC 
# MAGIC **`versionAsOf`** オプションを使用すると、デルタテーブルの以前のバージョンに簡単にアクセスすることができます。

# COMMAND ----------

df = spark.read.format("delta").option("versionAsOf", 0).load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="5664be65-8fd2-4746-8065-35ee8b563797"/>
# MAGIC 
# MAGIC また、タイムスタンプを利用して古いバージョンにアクセスすることも可能です。
# MAGIC 
# MAGIC タイムスタンプ文字列をバージョン履歴の情報に置き換えます。なお、必要に応じて、時間情報を含まない日付を使用することができます。

# COMMAND ----------

# Use your own timestamp 
# time_stamp_string = "FILL_IN"

# OR programatically get the first verion's timestamp value
time_stamp_string = str(spark.sql("DESCRIBE HISTORY train_delta").collect()[-1]["timestamp"])

df = spark.read.format("delta").option("timestampAsOf", time_stamp_string).load(DA.paths.working_dir)
display(df)

# COMMAND ----------

# MAGIC %md <i18n value="6cbe5204-fe27-438a-af54-87492c2563b5"/>
# MAGIC 
# MAGIC **`VACUUM`** を使用してディレクトリをクリーンアップすることができます。Vacuumでは、時間単位で保存期間の指定ができます。

# COMMAND ----------

# MAGIC %md <i18n value="4da7827c-b312-4b66-8466-f0245f3787f4"/>
# MAGIC 
# MAGIC あれっ、コードが実行できない！？デフォルトでは、最近のコミットを誤ってvacuumするのを防ぐために、Delta Lakeはユーザーに7日または168時間以内の履歴をvacuumさせないようになっています。一旦vacuumすると、タイムトラベルで以前のコミットに戻ることはできず、最新のDelta Tableのみが保存されます。
# MAGIC 
# MAGIC vacuumのパラメーターを別の値に変更してみてください。

# COMMAND ----------

# from delta.tables import DeltaTable

# delta_table = DeltaTable.forPath(spark, DA.paths.working_dir)
# delta_table.vacuum(0)

# COMMAND ----------

# MAGIC %md <i18n value="1150e320-5ed2-4a38-b39f-b63157bca94f"/>
# MAGIC 
# MAGIC デフォルトの保存期間のチェックを通すようなSpark設定をすることで、これを回避することができます。

# COMMAND ----------

from delta.tables import DeltaTable

spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
delta_table = DeltaTable.forPath(spark, DA.paths.working_dir)
delta_table.vacuum(0)

# COMMAND ----------

# MAGIC %md <i18n value="b845b2ea-2c11-4d6e-b083-d5908b65d313"/>
# MAGIC 
# MAGIC それでは、Delta Tableのファイルを見てみましょう。vacuum後、ディレクトリには直近のDelta Tableコミットのパーティションのみが格納されています。

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.working_dir}/neighbourhood_cleansed=Bayview/"))

# COMMAND ----------

# MAGIC %md <i18n value="a7bcdad3-affb-4b00-b791-07c14f5e59d5"/>
# MAGIC 
# MAGIC vacuumするとDelta Tableが参照する過去履歴のファイルが削除されるため、過去のバージョンにアクセスできなくなります。以下のコードを実行すると、エラーが発生するはずです。

# COMMAND ----------

# df = spark.read.format("delta").option("versionAsOf", 0).load(DA.paths.working_dir)
# display(df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
