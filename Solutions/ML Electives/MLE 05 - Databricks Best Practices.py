# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="64f90be2-bcc0-40c9-bbe7-3b501323e71c"/>
# MAGIC 
# MAGIC # Databricks のベストプラクティス (Databricks Best Practices)
# MAGIC 
# MAGIC このノートブックでは、Databricksを使用する際のさまざまなベストプラクティスを紹介します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンで以下を行います。<br>
# MAGIC  - 動作の遅いジョブをデバッグするための一般的なフレームワークを探索する
# MAGIC  - 様々なデータアクセスパラダイムが持つセキュリティ上の意味を理解する
# MAGIC  - マシンタイプ、ライブラリ、ジョブなど、さまざまなクラスタ構成の問題を判断する
# MAGIC  - Databricksのノートブックとジョブをバージョン管理およびCLIと統合する

# COMMAND ----------

# MAGIC %md <i18n value="0c063b0a-ccbb-486f-8568-1fe52cfa8971"/>
# MAGIC 
# MAGIC ## 実行速度が遅いジョブ (Slow Running Jobs)
# MAGIC 
# MAGIC ジョブの実行速度が遅い場合の最も一般的な問題：<br>
# MAGIC 
# MAGIC - **`Spill`** (スピル、メモリ溢れ): メモリーからディスクにデータが溢れること。解決策：より多くのメモリを持つクラスタを使用する。
# MAGIC - **`Shuffle`** : 大量データがクラスタ間で転送されていること。 解決策：Joinを最適化するか、シャッフルを回避するためにコードをリファクタリングする。
# MAGIC - **`スキュー/ストラグラー`** : パーティショニングされたデータ（ファイルまたはメモリ内）が均等に分散されていない、一部のパーティションの実行に時間がかかる「最後のreducerの呪い」が発生します。 解決策：使用可能なコアの倍数に再分割するか、スキューヒントを使用する。
# MAGIC - **`Small/Large Files`** : 小さなファイルが多すぎて、各ファイルの読み込みに専用のスレッドが必要なため、クラスタのリソースが枯渇しているか、大きなファイルのせいで未使用のスレッドが発生しています。 解決策：より最適な方法でデータを書き直すか、Deltaファイルのコンパクションを実行する。
# MAGIC 
# MAGIC デバッグのツールキット:<br>
# MAGIC 
# MAGIC - CPU、ネットワーク、メモリのリソースをクラスタまたはノードレベルで管理するGanglia
# MAGIC - その他を監視するSpark UI（特にストレージとエグゼキュータータブ）
# MAGIC - ドライバやワーカーのログにエラーがないか（特にバックグラウンドプロセスの場合）
# MAGIC - クラスタ・セクションのノートブックタブで、インターンが再びクラスタを占拠しているかどうか

# COMMAND ----------

# MAGIC %md <i18n value="35f52f8a-6a95-4273-8e04-ead835c2c184"/>
# MAGIC 
# MAGIC ## データアクセスとセキュリティ (Data Access and Security)
# MAGIC 
# MAGIC データ・アクセスに関するいくつかの注意点:<br>
# MAGIC 
# MAGIC * <a href="https://docs.databricks.com/data/databricks-file-system.html#mount-storage" target="_blank">簡単にアクセスできるためデータをマウントする</a>
# MAGIC * <a href="https://docs.databricks.com/dev-tools/cli/secrets-cli.html#secrets-cli" target="_blank">Secretを使って認証情報を保護する</a> (これは認証情報をコードに残さないようにするものです)
# MAGIC <a href="https://docs.databricks.com/dev-tools/cli/secrets-cli.html#secrets-cli" target="_blank">AWS</a> でも <a href="https://docs.microsoft.com/en-us/azure/databricks/security/credential-passthrough/adls-passthrough" target="_blank">Azure</a> でもCredential passthrough機能が使える。

# COMMAND ----------

# MAGIC %md <i18n value="2c6e2b76-709f-43e9-9fd2-731713fe30a7"/>
# MAGIC 
# MAGIC ## クラスタの構成、ライブラリ、ジョブ (Cluster Configuration, Libraries, and Jobs)
# MAGIC 
# MAGIC クラスタータイプ：
# MAGIC 
# MAGIC - メモリ最適化（<a href="https://docs.databricks.com/delta/optimizations/delta-cache.html" target="_blank">Delta Cache Acceleration</a>あり/なし
# MAGIC - コンピューティング最適化
# MAGIC - ストレージ最適化
# MAGIC - GPU最適化
# MAGIC - 汎用
# MAGIC 
# MAGIC 一般的な経験則:<br>
# MAGIC 
# MAGIC - 機械学習用に大きなマシンタイプの小型クラスタ
# MAGIC - 本番ワークロードごとに1クラスタ
# MAGIC - ML学習用のクラスタを共有しない（開発中であっても）
# MAGIC - <a href="https://docs.databricks.com/clusters/configure.html" target="_blank">詳しくはドキュメントをご覧ください。</a>

# COMMAND ----------

# MAGIC %md <i18n value="6368d08e-4f54-4504-8a83-5e099c7aeb34"/>
# MAGIC 
# MAGIC ライブラリインストールのベストプラクティス:<br>
# MAGIC   
# MAGIC - <a href="https://docs.databricks.com/libraries/notebooks-python-libraries.html" target="_blank">Notebook-scoped Python libraries</a> は、同じクラスタのユーザが異なるライブラリを持つことができることを保証します。 また、ライブラリに依存するノートブックを保存するのにも便利です。
# MAGIC - <a href="https://docs.databricks.com/clusters/init-scripts.html" target="_blank">初期化スクリプト</a>は、JVMの起動前にコードが実行されることを保証します（特定のライブラリまたは環境構成に適しています）。
# MAGIC - いくつかの構成変数は、クラスタ起動時に設定する必要があります。

# COMMAND ----------

# MAGIC %md <i18n value="dd0026c2-92e2-4761-9308-75ad353649d4"/>
# MAGIC 
# MAGIC ジョブのベストプラクティス:<br>
# MAGIC 
# MAGIC - <a href="https://docs.databricks.com/notebooks/notebook-workflows.html" target="_blank">ノートブックワークフロー</a>を使用します。
# MAGIC - <a href="https://docs.databricks.com/notebooks/widgets.html" target="_blank">ウィジェット</a>はパラメータ渡しのために使用します。
# MAGIC - jars や wheelsも実行することができます。
# MAGIC - オーケストレーションツール（Airflowなど）のCLIを利用します。
# MAGIC - <a href="https://docs.databricks.com/jobs.html" target="_blank">詳しくはドキュメントをご覧ください</a>。
# MAGIC - 無限にジョブが実行されるのを防ぐため、必ずタイムアウト時間を指定します。

# COMMAND ----------

# MAGIC %md <i18n value="ea44ac8c-88c8-443a-a370-b4671af6f1e9"/>
# MAGIC 
# MAGIC ## CLIとバージョン管理 (CLI and Version Control)
# MAGIC 
# MAGIC <a href="https://github.com/databricks/databricks-cli" target="_blank">Databricks CLI</a>:<br>
# MAGIC 
# MAGIC  * プログラム的にすべてのノートブックをエクスポートし、githubにチェックインします。
# MAGIC  * データのインポート/エクスポート、ジョブの実行、クラスタの作成、その他ほとんどのWorkspaceタスクの実行が可能です。
# MAGIC 
# MAGIC Git の統合は、いくつかの方法で実現できます:<br>
# MAGIC 
# MAGIC  * CLIを使ってノートブックをインポート/エクスポートし、gitに手動でチェックインする。
# MAGIC  * <a href="https://docs.databricks.com/notebooks/github-version-control.html" target="_blank">ビルドインの git統合機能を使用する</a>。
# MAGIC  * <a href="https://www.youtube.com/watch?v=HsfMmBfQtvI" target="_blank">プロジェクト統合の代替に次世代ワークスペースを使用する</a>。

# COMMAND ----------

# MAGIC %md <i18n value="4bbc8017-a03b-4b3e-810f-9375e5afd7e2"/>
# MAGIC 
# MAGIC 時間が許す限り、<a href="https://docs.databricks.com/administration-guide/index.html" target="_blank">管理コンソール</a>を探索してください。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
