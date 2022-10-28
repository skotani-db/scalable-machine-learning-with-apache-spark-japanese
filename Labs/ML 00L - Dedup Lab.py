# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="bfaf8cdf-ed63-45f7-a559-954d478bb0f7"/>
# MAGIC 
# MAGIC # De-Duping Data (重複したデータ削除)
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png)概要説明
# MAGIC 
# MAGIC この演習では、ある顧客から受け取ったファイルに対してETLを実行します。そのファイルには、以下のような人に関するデータが含まれています。
# MAGIC 
# MAGIC * ファーストネーム、ミドルネーム、ラストネーム
# MAGIC * 性別
# MAGIC * 生年月日
# MAGIC * 社会保障番号
# MAGIC * 給与
# MAGIC 
# MAGIC しかし、今回のデータには重複したレコードが含まれています。さらに：
# MAGIC 
# MAGIC * 一部のレコードでは、名前に大文字と小文字が混在（例："Carol"）していますが、他のレコードでは大文字（例："CAROL"）で表記しています。
# MAGIC * ソーシャルセキュリティ番号もフォーマットが統一されていません。ハイフンが入っているもの（例："992-83-4829"）もあれば、ハイフンが抜けているもの（"992834829"）もあります。
# MAGIC 
# MAGIC 名前のフィールドは、文字の大文字小文字を無視すれば一致することが保証されており、生年月日も一致します。給与も一致しています。
# MAGIC 社会保障番号（Social Security Numbers）は、何らかの方法で同じフォーマットにすれば、一致します。
# MAGIC 
# MAGIC この演習のタスクは、重複するレコードを削除することです。具体的な内容は：
# MAGIC 
# MAGIC * 重複を削除します。重複のどの記録を残しても良いが、一つを残すことが重要です。
# MAGIC * カラムの元データ形式を保持します。例えば、姓のカラムをすべて小文字で書いてしまうと、この条件を満たしていないことになります。
# MAGIC * 結果を *dest_file* で指定された Parquet ファイルとして書き出す。
# MAGIC * 最終的なParquet "ファイル "は8つのパーティション・ファイル（拡張子が".parquet "で終わる8つのファイル）から構成されている必要があります。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** 初期データセットには103,000レコードが含まれています。
# MAGIC 重複を排除した結果は、10万件のレコードになります。

# COMMAND ----------

# MAGIC %md <i18n value="8f030328-5dad-4ae9-bf82-30e96ffc38a5"/>
# MAGIC 
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) 演習開始
# MAGIC 
# MAGIC 次のセルを実行して、"classroom"を設定してください。

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md <i18n value="c70ecf76-ddd9-4f68-8c2c-41bbf258419c"/>
# MAGIC 
# MAGIC ##![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) ヒント
# MAGIC 
# MAGIC * <a href="https://spark.apache.org/docs/latest/api/python/index.html" target="_blank">API docs</a> を参考にしてください。具体的には、 
# MAGIC   <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.html?highlight=dataframe#pyspark.sql.DataFrame" target="_blank">DataFrame</a>と
# MAGIC   <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql.html#functions" target="_blank">functions</a>は演習の完成に役に立ちます。
# MAGIC * まずファイルの内容を見て、フォーマットを確認します。 **`dbutils.fs.head()`** (または単に **`%fs head`** ) を使います。

# COMMAND ----------

# TODO

source_file = f"{DA.paths.datasets}/dataframes/people-with-dups.txt"
dest_file = f"{DA.paths.working_dir}/people.parquet"

# In case it already exists
dbutils.fs.rm(dest_file, True)

# COMMAND ----------

# MAGIC %md <i18n value="3616cc9c-788f-431a-a895-fde4e0983366"/>
# MAGIC 
# MAGIC ##![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) 回答を検証します
# MAGIC 
# MAGIC 最低限の確認として、parquetファイルを**dest_file**に書き出したことと、正しいレコード数を検証します。
# MAGIC 
# MAGIC 以下のセルを実行して、結果を確認します。

# COMMAND ----------

part_files = len(list(filter(lambda f: f.path.endswith(".parquet"), dbutils.fs.ls(dest_file))))

final_df = spark.read.parquet(dest_file)
final_count = final_df.count()

clearYourResults()
validateYourAnswer("01 Parquet File Exists", 1276280174, part_files)
validateYourAnswer("02 Expected 100000 Records", 972882115, final_count)
summarizeYourResults()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
