# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="04aa5a94-e0d3-4bec-a9b5-a0590c33a257"/>
# MAGIC 
# MAGIC # モデルレジストリ (Model Registry)
# MAGIC 
# MAGIC MLflow Model Registryは、チームでMLモデルの共有、実験からオンラインテスト、本番までの共同作業、承認およびガバナンスワークフローとの統合、MLデプロイメントとそのパフォーマンスの監視を行うことができるコラボレーションハブです。 このレッスンでは、MLflow モデルレジストリを使用してモデルを管理する方法について説明します。
# MAGIC 
# MAGIC このデモノートブックではAirbnbのデータセットにscikit-learnを使用しますが、ラボではMLlibを使用します。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンで以下を行います。<br>
# MAGIC  - MLflowを使ったモデルの登録
# MAGIC  - モデルライフサイクルの管理
# MAGIC  - モデルのアーカイブと削除
# MAGIC  
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> モデルサービングのエンドポイントを立ち上げたい場合は、<a href="https://docs.databricks.com/applications/mlflow/model-serving.html#requirements" target="_blank">クラスターの作成</a> 権限が必要です。

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="5802ff47-58b5-4789-973d-2fb855bf347a"/>
# MAGIC 
# MAGIC ### モデルレジストリ (Model Registry)
# MAGIC 
# MAGIC MLflow Model Registry コンポーネントは、MLflow モデルの全ライフサイクルを共同管理するための集中型モデルストア、API のセット、および UI です。モデルのリネージ（どの MLflow Experiment と Run でモデルを作成したか）、モデルのバージョン管理、ステージ遷移（例：ステージングから本番へ）、アノテーション（例：コメントやタグ）、デプロイ管理（例：どの本番ジョブから特定のモデルバージョンをリクエストしたか）を提供します。
# MAGIC 
# MAGIC モデルレジストリは以下の機能を備えています：
# MAGIC 
# MAGIC * **中央リポジトリ:** MLflow Model RegistryにMLflowモデルを登録します。登録されたモデルは、一意の名前、バージョン、ステージ、その他のメタデータを持ちます。
# MAGIC * **モデルのバージョン管理：** 登録されたモデルの更新時にバージョンを自動的に追跡します。
# MAGIC * **モデルステージ：** モデルのライフサイクルを表現するために、「ステージング」や「プロダクション」のように、各モデルのバージョンにプリセットまたはカスタムのステージを割り当てます。
# MAGIC * **モデルステージの遷移：** モデルの新規登録または変更のイベントを記録します。ユーザー、変更内容、およびコメントなどの追加メタデータが自動的に記録されます。
# MAGIC * **CI/CDワークフローの統合:** コントロールとガバナンス強化のため、CI/CDパイプラインの一部として、モデルステージの遷移、リクエスト、レビューと承認変更を記録します。
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry.png" style="height:400px; margin:20px"/></div>
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> モデルレジストリの詳細については <a href="https://mlflow.org/docs/latest/registry.html" target="_blank">MLflow docs</a> を参照してください。

# COMMAND ----------

# MAGIC %md <i18n value="7f34f7da-b5d2-42af-b24d-54e1730db95f"/>
# MAGIC 
# MAGIC ### モデルの登録 (Registering a Model)
# MAGIC 
# MAGIC 以下のワークフローは、UIでも純粋なPythonでも動作します。 このノートブックでは、純粋なPythonを使用します。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> 画面左側の「モデル」タブをクリックし、UIを探索してください。

# COMMAND ----------

# MAGIC %md <i18n value="cbc59424-e45b-4179-a586-8c14a66a61a1"/>
# MAGIC 
# MAGIC モデルを学習し、<a href="https://docs.databricks.com/applications/mlflow/databricks-autologging.html" target="_blank">autologging (オートロギング)</a> を使って、MLflowにログを記録します。オートロギングにより、明示的なログステートメントを使用せずに、メトリクス、パラメータ、モデルを記録することができます。
# MAGIC 
# MAGIC オートロギングを使うには、いくつかの方法があります。
# MAGIC 
# MAGIC 1. 学習コードの前に **`mlflow.autolog()`** を呼び出します。これにより、インポートすると同時に、インストールした各サポートしているライブラリのオートロギングが有効になります。
# MAGIC 1. 管理コンソールからワークスペース・レベルでオートロギングを有効にします
# MAGIC 1. コードで使用する各ライブラリには、ライブラリ固有の autolog 呼び出しを使用します。(例: **`mlflow.spark.autolog()`** )
# MAGIC 
# MAGIC ここでは、数値特徴のみを使用して簡単なランダムフォレストを構築します。

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv(f"{DA.paths.datasets}/airbnb/sf-listings/airbnb-cleaned-mlflow.csv".replace("dbfs:/", "/dbfs/"))
X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

with mlflow.start_run(run_name="LR Model") as run:
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, log_models=True)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    signature = infer_signature(X_train, lr.predict(X_train))

# COMMAND ----------

# MAGIC %md <i18n value="1322cac5-9638-4cc9-b050-3545958f3936"/>
# MAGIC 
# MAGIC ワークスペースのほかのユーザーとぶつからないように、ユニークなモデル名を作成します。
# MAGIC 
# MAGIC モデル名は、空でないUTF-8文字列でなければならず、フォワードスラッシュ(/)、ピリオド(.)、コロン(:)を含むことができないことに注意してください。

# COMMAND ----------

model_name = f"{DA.cleaned_username}_sklearn_lr"
model_name

# COMMAND ----------

# MAGIC %md <i18n value="0777e3f5-ba7c-41c4-a477-9f0a5a809664"/>
# MAGIC 
# MAGIC モデルを登録します。

# COMMAND ----------

run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="22756858-ff7f-4392-826f-f401a81230c4"/>
# MAGIC 
# MAGIC **画面左の*モデル*タブを開いて、登録されているモデルを探索します。** 以下の点に注意してください。
# MAGIC 
# MAGIC * 誰がモデルを学習させたか、どのコードを使用したかを記録します。
# MAGIC * このモデルで行われたアクションの履歴を記録します。
# MAGIC * このモデルをバージョン１として記録します。
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/301/registered_model_new.png" style="height:600px; margin:20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="481cba23-661f-4de7-a1d8-06b6be8c57d3"/>
# MAGIC 
# MAGIC 状態を確認します。

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

# MAGIC %md <i18n value="10556266-2903-4afc-8af9-3213d244aa21"/>
# MAGIC 
# MAGIC 次に、モデルの説明を追加します。

# COMMAND ----------

client.update_registered_model(
    name=model_details.name,
    description="This model forecasts Airbnb housing list prices based on various listing inputs."
)

# COMMAND ----------

# MAGIC %md <i18n value="5abeafb2-fd60-4b0d-bf52-79320c10d402"/>
# MAGIC 
# MAGIC バージョンに応じた説明を追加します。

# COMMAND ----------

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="This model version was built using OLS linear regression with sklearn."
)

# COMMAND ----------

# MAGIC %md <i18n value="aaac467f-3a52-4428-a119-8286cb0ac158"/>
# MAGIC 
# MAGIC ### モデルのデプロイメント (Deploying a Model)
# MAGIC 
# MAGIC MLflow Model Registryは、4つのモデルステージを定義しています。 **`None`** ,  **`Staging`** , **`Production`** と **`Archived`** です。それぞれのステージには、固有の意味があります。例えば、 **`Staging`** はモデルのテスト用で、 **`Production`** はテストまたはレビュープロセスを完了し、アプリケーションにデプロイされたモデルのものです。
# MAGIC 
# MAGIC 適切な権限を持つユーザーは、ステージ間でモデルを移行させることができます。

# COMMAND ----------

# MAGIC %md <i18n value="dff93671-f891-4779-9e41-a0960739516f"/>
# MAGIC 
# MAGIC ステージの遷移について学んだので、モデルを **`Production`** ステージに遷移させます。

# COMMAND ----------

import time

time.sleep(10) # In case the registration is still pending

# COMMAND ----------

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md <i18n value="4dc7e8b7-da38-4ce1-a238-39cad74d97c5"/>
# MAGIC 
# MAGIC モデルの現在の状態を取得します。

# COMMAND ----------

model_version_details = client.get_model_version(
    name=model_details.name,
    version=model_details.version
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

# MAGIC %md <i18n value="ba563293-bb74-4318-9618-a1dcf86ec7a3"/>
# MAGIC 
# MAGIC **`pyfunc`** を使って最新のモデルを取得します。 このようにモデルをロードすると、トレーニングに使用したパッケージに関係なく、モデルを使用することができます。
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> 特定のバージョンのモデルも読み込むことができます。

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

# MAGIC %md <i18n value="e1bb8ae5-6cf3-42c2-aebd-bde925a9ef30"/>
# MAGIC 
# MAGIC モデルを適用します。

# COMMAND ----------

model_version_1.predict(X_test)

# COMMAND ----------

# MAGIC %md <i18n value="75a9c277-0115-4cef-b4aa-dd69a0a5d8a0"/>
# MAGIC 
# MAGIC ### 新しいバージョンのモデルのデプロイ (Deploying a New Model Version)
# MAGIC 
# MAGIC MLflow Model Registryでは、登録された1つのモデルに対応する複数のモデルバージョンを作成することができます。ステージ遷移を行うことで、新しいバージョンのモデルをステージング環境またはプロダクション環境にシームレスに統合することができます。

# COMMAND ----------

# MAGIC %md <i18n value="2ef7acd0-422a-4449-ad27-3a26f217ab15"/>
# MAGIC 
# MAGIC 新しいモデルのバージョンを作成し、ログに記録されたときにそのモデルを登録します。

# COMMAND ----------

from sklearn.linear_model import Ridge

with mlflow.start_run(run_name="LR Ridge Model") as run:
    alpha = .9
    ridge_regression = Ridge(alpha=alpha)
    ridge_regression.fit(X_train, y_train)

    # Specify the `registered_model_name` parameter of the `mlflow.sklearn.log_model()`
    # function to register the model with the MLflow Model Registry. This automatically
    # creates a new model version

    mlflow.sklearn.log_model(
        sk_model=ridge_regression,
        artifact_path="sklearn-ridge-model",
        registered_model_name=model_name,
    )

    mlflow.log_params(ridge_regression.get_params())
    mlflow.log_metric("mse", mean_squared_error(y_test, ridge_regression.predict(X_test)))

# COMMAND ----------

# MAGIC %md <i18n value="dc1dd6b4-9e9e-45be-93c4-5500a10191ed"/>
# MAGIC 
# MAGIC 新モデルをステージングにします。

# COMMAND ----------

import time

time.sleep(10)

client.transition_model_version_stage(
    name=model_details.name,
    version=2,
    stage="Staging"
)

# COMMAND ----------

# MAGIC %md-sandbox <i18n value="fe857eeb-6119-4927-ad79-77eaa7bffe3a"/>
# MAGIC 
# MAGIC 新モデルのバージョンはUIで確認します。
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/301/model_version_new.png" style="height:600px; margin:20px"/></div>

# COMMAND ----------

# MAGIC %md <i18n value="6f568dd2-0413-4b78-baf6-23debb8a5118"/>
# MAGIC 
# MAGIC 検索機能を使って、モデルの最新バージョンを取得できます。

# COMMAND ----------

model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

# MAGIC %md <i18n value="4fb5d7c9-b0c0-49d5-a313-ac95da7e0f91"/>
# MAGIC 
# MAGIC この新しいバージョンに説明を追加します。

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=new_model_version,
    description=f"This model version is a ridge regression model with an alpha value of {alpha} that was trained in scikit-learn."
)

# COMMAND ----------

# MAGIC %md <i18n value="10adff21-8116-4a01-a309-ce5a7d233fcf"/>
# MAGIC 
# MAGIC このモデルは現在ステージングにあるので、自動化CI/CDパイプラインを実行して、本番に入る前にモデルをテストすることができます。 それが完成すれば、そのモデルを本番に移行させることができます。

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=new_model_version,
    stage="Production", 
    archive_existing_versions=True # Archieve existing model in production 
)

# COMMAND ----------

# MAGIC %md <i18n value="e3caaf08-a721-425b-8765-050c757d1d2e"/>
# MAGIC 
# MAGIC バージョン1を削除します。 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> アーカイブされていないモデルを削除することはできません。

# COMMAND ----------

client.delete_model_version(
    name=model_name,
    version=1
)

# COMMAND ----------

# MAGIC %md <i18n value="a896f3e5-d83c-4328-821f-a67d60699f0e"/>
# MAGIC 
# MAGIC モデルのバージョン2もアーカイブします。

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Archived"
)

# COMMAND ----------

# MAGIC %md <i18n value="0eb4929d-648b-4ae6-bca3-aff8af50f15f"/>
# MAGIC 
# MAGIC ここで、登録されているモデル全体を削除します。

# COMMAND ----------

client.delete_registered_model(model_name)

# COMMAND ----------

# MAGIC %md <i18n value="6fe495ec-f481-4181-a006-bea55a6cef09"/>
# MAGIC 
# MAGIC ## レビュー (Review)
# MAGIC * **質問：** MLflow のトラッキングは、モデルレジストリとはどのように違うのですか？ 
# MAGIC * **回答：** トラッキングは実験と開発のためのものです。 モデルレジストリは、モデルをトラッキングサーバからステージングを経てプロダクションに移行させるためのものです。 これは、データエンジニアや機械学習エンジニアがデプロイメントプロセスの中でよく担当する役割です。
# MAGIC 
# MAGIC * **質問：** なぜ、モデルレジストリが必要なのでしょうか？ 
# MAGIC * **回答：** MLflowのトラッキングが機械学習のトレーニングプロセスにエンドツーエンドの再現性を提供するように、モデルレジストリはデプロイメントプロセスの再現性とガバナンスを提供します。 本番システムはミッションクリティカルであるため、コンポーネントをACLで分離し、特定のユーザーのみが本番モデルを変更できるようにすることが可能です。 バージョン管理およびCI/CDワークフローの統合も、モデルを本番環境にデプロイする上で重要な側面です。
# MAGIC 
# MAGIC * **質問：** UIと比較して、プログラムでできることは何ですか？ 
# MAGIC * **回答：** ほとんどの操作は、UIを使用するか、純粋なPythonで行うことができます。 モデルのトラッキングはPythonで行う必要がありますが、そこから先はどちらでも可能です。 例えば、MLflow のトラッキング API を使って記録されたモデルは、UI を使って登録し、本番環境にプッシュすることができます。

# COMMAND ----------

# MAGIC %md <i18n value="ecf5132e-f80d-4374-a325-28b4e96d5b61"/>
# MAGIC 
# MAGIC ## その他のトピックとリソース (Additional Topics & Resources)
# MAGIC 
# MAGIC **Q:** MLflow Model Registry の詳細資料は、どこにありますか？ 
# MAGIC **A:** <a href="https://mlflow.org/docs/latest/registry.html" target="_blank">MLflowドキュメンテーション</a> をご覧ください。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
