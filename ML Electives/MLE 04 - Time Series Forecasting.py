# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md <i18n value="54c3040f-38b6-4562-8dd3-61a8bb6aeba1"/>
# MAGIC 
# MAGIC # 時系列予測 (Time Series Forecasting)
# MAGIC 
# MAGIC 時系列データを扱うことは、データサイエンスにおいて重要なスキルです。 このノートでは、時系列に対する3つの主要なアプローチについて学びます: Prophet、ARIMA、指数平滑化。
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) このレッスンでは、次のことを行います。<br>
# MAGIC - 時系列における主な概念の紹介
# MAGIC - Prophetを使ったCOVIDデータの予測
# MAGIC - ARIMAによる予測
# MAGIC - 指数平滑化法による予測
# MAGIC 
# MAGIC このノートブックでは、韓国のCOVID‑19感染者に関する<a href="https://www.kaggle.com/kimjihoo/coronavirusdataset" target="_blank">Coronavirus dataset</a> を使用します。

# COMMAND ----------

# MAGIC %pip install --upgrade pystan==2.19.1.1 fbprophet

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md <i18n value="5537f13d-b402-464f-8814-bb981709ffb2"/>
# MAGIC 
# MAGIC ### <a href="https://en.wikipedia.org/wiki/Time_series" target="_blank">時系列 (Time Series)</a>
# MAGIC 
# MAGIC 時系列とは、時間順に索引付け（またはリスト化、グラフ化）された一連のデータポイントのことです。最も一般的な時系列は、時間的に等間隔に連続して取得されたデータ系列です。したがって、これは離散時間データのシーケンスです。時系列の例：<br>
# MAGIC 
# MAGIC - 海洋潮汐の高さ
# MAGIC - 太陽黒点の数
# MAGIC - ダウ・ジョーンズ工業の平均株価の日次終値
# MAGIC 
# MAGIC このノートブックでは、時系列予測、つまり過去に観測された値から構築したモデルで将来の値を予測することに焦点を当てます。

# COMMAND ----------

file_path = f"{DA.paths.datasets}/COVID/coronavirusdataset/Time.csv"

spark_df = (spark
            .read
            .option("inferSchema", True)
            .option("header", True)
            .csv(file_path)
           )
  
display(spark_df)

# COMMAND ----------

# MAGIC %md <i18n value="91681688-70e2-4eee-b18a-4afa353bce3f"/>
# MAGIC 
# MAGIC Spark DataFrameをPandas DataFrameに変換します。

# COMMAND ----------

df = spark_df.toPandas()

# COMMAND ----------

# MAGIC %md <i18n value="920f1e35-2a54-4588-b4bf-72c6bed85e07"/>
# MAGIC 
# MAGIC データを見ると、時間列（データが観察された時間）は今回の予測に特に関係がないので、そのまま削除してもよいでしょう。

# COMMAND ----------

df = df.drop(columns="time")
df.head()

# COMMAND ----------

# MAGIC %md <i18n value="f5c365d6-4d8b-49a2-a8be-36aa3232d6c1"/>
# MAGIC 
# MAGIC ### Prophet
# MAGIC <a href="https://facebook.github.io/prophet/" target="_blank">FacebookのProphetパッケージ</a>は、ユーザーに代わってヘビーな作業を行ってくれるため、時系列予測タスクによく採用されています。今回のデータセットを使ってProphetの使い方を見てみましょう。

# COMMAND ----------

import pandas as pd
from fbprophet import Prophet
import logging

# Suppresses `java_gateway` messages from Prophet as it runs.
logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md <i18n value="a0a43507-9db1-41da-b7bf-8d5c2f4b2a67"/>
# MAGIC 
# MAGIC Prophetの入力DataFrameに特定のカラム名が必要です。日付の列はdsに、予測する列はyに改名する必要があります。それでは、韓国での確定患者数を予想してみましょう。

# COMMAND ----------

prophet_df = pd.DataFrame()
prophet_df["ds"] = pd.to_datetime(df["date"])
prophet_df["y"] = df["confirmed"]
prophet_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="daf04369-b1c3-4c84-80c1-0f7da47fc3e6"/>
# MAGIC 
# MAGIC 次に、何日分の予測を行うかを指定しましょう。これは  **`Prophet.make_future_dataframe`** メソッドを用いて行うことができます。データの規模が大きいので、1ヵ月後の数字を見てみましょう。
# MAGIC 
# MAGIC 1ヶ月先までの日付を見ることができます。

# COMMAND ----------

prophet_obj = Prophet()
prophet_obj.fit(prophet_df)
prophet_future = prophet_obj.make_future_dataframe(periods=30)
prophet_future.tail()

# COMMAND ----------

# MAGIC %md <i18n value="b79ef0fd-8017-4d04-a1f4-f4e8f04dfc87"/>
# MAGIC 
# MAGIC 最後に、 **`predict`** メソッドを使って、データポイントを予測します。**`yhat`** 列には、予測値が格納されます。また、DataFrameにProphetが生成する他の値を確認することができます。

# COMMAND ----------

prophet_forecast = prophet_obj.predict(prophet_future)
prophet_forecast[['ds', 'yhat']].tail()

# COMMAND ----------

# MAGIC %md <i18n value="02352d36-96cb-4c11-aca6-d47a194f9942"/>
# MAGIC 
# MAGIC では、 **`plot`** を使って、予測結果のグラフ表示を見てみましょう。

# COMMAND ----------

prophet_plot = prophet_obj.plot(prophet_forecast)

# COMMAND ----------

# MAGIC %md <i18n value="d260f48c-7aaa-4cf2-8ab1-50c3a9c7318d"/>
# MAGIC 
# MAGIC また、 **`plot_components`** を使用して、予測結果をより詳細に見ることができます。

# COMMAND ----------

prophet_plot2 = prophet_obj.plot_components(prophet_forecast)

# COMMAND ----------

# MAGIC %md <i18n value="44baaf89-7f68-48a7-9cfd-8ed431613bfa"/>
# MAGIC 
# MAGIC Prophetを使って、<a href="https://facebook.github.io/prophet/docs/trend_changepoints.html" target="_blank">チェンジポイント</a>（データセットが急激に変化したポイント）を特定することもできます。これは、感染者が急増した時期を特定できるためとても有用な情報になります。

# COMMAND ----------

from fbprophet.plot import add_changepoints_to_plot

prophet_plot = prophet_obj.plot(prophet_forecast)
changepts = add_changepoints_to_plot(prophet_plot.gca(), prophet_obj, prophet_forecast)

# COMMAND ----------

print(prophet_obj.changepoints)

# COMMAND ----------

# MAGIC %md <i18n value="93d9af60-11de-472f-8a26-f9a679ff29f2"/>
# MAGIC 
# MAGIC 次に、韓国での休日と感染者数の増加の相関を確認します。ビルトインの **`add_country_holidays`** <a href="https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#built-in-country-holidays" target="_blank">メソッド</a> を使って、休日に関するあらゆる傾向について調べることができます。
# MAGIC 
# MAGIC 国コードの完全なリストは、<a href="https://github.com/dr-prodigy/python-holidays/blob/master/holidays/countries/" target="_blank">こちら</a>で確認できます。

# COMMAND ----------

holidays = pd.DataFrame({"ds": [], "holiday": []})
prophet_holiday = Prophet(holidays=holidays)

prophet_holiday.add_country_holidays(country_name='KR')
prophet_holiday.fit(prophet_df)

# COMMAND ----------

# MAGIC %md <i18n value="0a5f7168-3877-4457-81d4-4ceebce8ec02"/>
# MAGIC 
# MAGIC どのような祝日が含まれているかは、以下のセルを実行することで確認できます。

# COMMAND ----------

prophet_holiday.train_holiday_names

# COMMAND ----------

prophet_future = prophet_holiday.make_future_dataframe(periods=30)
prophet_forecast = prophet_holiday.predict(prophet_future)
prophet_plot_holiday = prophet_holiday.plot_components(prophet_forecast)

# COMMAND ----------

# MAGIC %md <i18n value="81681565-2eb4-467c-8c5e-c6546c7230aa"/>
# MAGIC 
# MAGIC ### ARIMA
# MAGIC 
# MAGIC ARIMAとは、Auto-Regressive (AR) Integrated (I) Moving Average (MA)の略です。ARIMAモデルは回帰分析の一種で、データ系列のある点とその直近時点の値との関係性を分析する手法です。
# MAGIC 
# MAGIC ARIMAは、Prophetと同様に、データセットの過去の値に基づいて将来の値を予測します。ARIMAはプロフェットと違い、設定作業が多くなりますが、様々な時系列に適用することが可能です。
# MAGIC 
# MAGIC ARIMAモデルを作成するために、以下のパラメータを求める必要があります。
# MAGIC 
# MAGIC - **`p`** (自己回帰パラメータ) :モデルに含まれるラグ観測の数で、ラグ次数とも呼ばれる。
# MAGIC - **`d`** (差分の階数):生の観測値が差分される回数で、差分化の度合いとも呼ばれます。
# MAGIC - **`q`** (移動平均パラメータ):移動平均の窓の大きさで、移動平均の次数とも呼ばれる。

# COMMAND ----------

# MAGIC %md <i18n value="99b5826a-6cf1-4bb0-a859-8cce45c50f74"/>
# MAGIC 
# MAGIC まず、新しいARIMA DataFrameを作成します。確定症例はすでにProphetを使って予測したので、隔離解除の患者数の予測を見てみましょう。

# COMMAND ----------

arima_df = pd.DataFrame()
arima_df["date"] = pd.to_datetime(df["date"])
arima_df["released"] = df["released"]
arima_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="286ec724-d52c-46ff-bd35-84201bede0a5"/>
# MAGIC 
# MAGIC ARIMAモデルを作成する最初のステップは、データセットが定常であることを確認し、dパラメータを求めます。これは、 **`statsmodels`** ライブラリの<a href="https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test" target="_blank">Augmented Dickey Fuller Test</a>を使って簡単に確認することができます。
# MAGIC 
# MAGIC P値がADF統計量より大きいので、データセットを差分する必要があります。差分化することで、データセットの平均値を安定させ、過去のトレンドや季節性の影響を排除することができます。

# COMMAND ----------

from statsmodels.tsa.stattools import adfuller
from numpy import log

result = adfuller(df.released.dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# COMMAND ----------

# MAGIC %md <i18n value="8766e03d-4623-40c2-a5fa-25684db75670"/>
# MAGIC 
# MAGIC データセットを差分化するには、valueカラムに対して **`diff`** を呼び出します。定義された平均とかなり早くゼロに到達するACFプロットの周りを歩き回る、ほぼ定常的な系列を探しています。グラフを見ると、dパラメータは1か2のどちらかにすべきことがわかります。

# COMMAND ----------

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams.update({"figure.figsize":(9,7), "figure.dpi":120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(arima_df.released); axes[0, 0].set_title('Original Series')
plot_acf(arima_df.released, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(arima_df.released.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(arima_df.released.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(arima_df.released.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(arima_df.released.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="7e2a2b58-4516-4036-9f93-e6514c529de5"/>
# MAGIC 
# MAGIC 次に、部分自己補正グラフ(Partial Autocorrection Plot)を用いて、必要なAR項の数を求めます。これがpパラメータです。
# MAGIC 
# MAGIC 部分自己補正とは、ある系列とそのラグとの相関を表すものです。グラフから、pパラメータは1にすべきです。

# COMMAND ----------

plt.rcParams.update({"figure.figsize":(9,3), "figure.dpi":120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(arima_df.released.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(arima_df.released.diff().dropna(), ax=axes[1])

plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="42204bfb-229b-4695-82f1-56b34ad04ba2"/>
# MAGIC 
# MAGIC 最後に、ACFプロットを見ながら、移動平均の項数を求めて、qパラメータを求めます。移動平均は、観測値と遅延した観測値に適用される残差との間の依存性を組み込んでいます。グラフによると、qパラメータは1にすべきです。

# COMMAND ----------

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(arima_df.released.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(arima_df.released.diff().dropna(), ax=axes[1])

plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="7855a2df-0091-4a6d-b05f-538c20784b66"/>
# MAGIC 
# MAGIC p、d、qのパラメータ値が決まったら、パラメータを渡してARIMAモデルのフィッティングを行うことができます。次のセルは、データセット情報とモデル係数を含むモデルの概要を示している。

# COMMAND ----------

from statsmodels.tsa.arima_model import ARIMA

# p, d, q
# 1, 2, 1 ARIMA Model
model = ARIMA(arima_df.released, order=(1,2,1))
arima_fit = model.fit(disp=0)
print(arima_fit.summary())

# COMMAND ----------

# MAGIC %md <i18n value="b787949f-d55a-4178-baf2-66b023567904"/>
# MAGIC 
# MAGIC 最後に、モデルの精度を検証するために、データを学習データとテストデータに分割してみましょう。時系列でデータを分割する必要があるため、sklearnの**train_test_split`**のような関数はここでは使えないことに注意してください。

# COMMAND ----------

split_ind = int(len(arima_df)*.7)
train_df = arima_df[ :split_ind]
test_df = arima_df[split_ind: ]
#train_df.tail()
#test_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="5e50d9a6-321f-4577-afa3-e146ad7a38ab"/>
# MAGIC 
# MAGIC 予測には、サンプル外交差検証(Out of Sample Cross Validation)を使用します。グラフに示したように、予測値が実際の値よりも若干線形になっているが、全体としてはかなり実測値に近い値になっています。

# COMMAND ----------

train_model = ARIMA(train_df.released, order=(1,2,1))  
train_fit = train_model.fit()  

fc, se, conf = train_fit.forecast(int(len(arima_df)-split_ind))

fc_series = pd.Series(fc, index=test_df.index)

plt.plot(train_df.released, label='train', color="dodgerblue")
plt.plot(test_df.released, label='actual', color="orange")
plt.plot(fc_series, label='forecast', color="green")
plt.title('Forecast vs Actuals')
plt.ylabel("Number of Released Patients")
plt.xlabel("Day Number")
plt.legend(loc='upper left', fontsize=8)
plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="ce127e7b-c717-4d46-b354-77bf6a0f8dc0"/>
# MAGIC 
# MAGIC ### 指数平滑化 (Exponential Smoothing)
# MAGIC 
# MAGIC <a href="https://en.wikipedia.org/wiki/Exponential_smoothing" target="_blank">指数平滑化</a>とは、指数窓関数を用いて時系列データを平滑化する経験則の手法です。単純移動平均では過去の観測値が均等に重み付けされるのに対し、指数関数は時間の経過とともに指数的に減少する重みを割り当てるために使用されます。季節性など、ユーザーによる事前の想定に基づいて何らかの判断を行うための手順で、簡単に習得でき、簡単に適用できます。時系列データの解析には、指数平滑化がよく使われます。
# MAGIC 
# MAGIC 指数平滑化には3つのタイプがあります。<br>
# MAGIC - 単純指数平滑化(SES)
# MAGIC   - トレンドや季節性のないデータセットに使用します。
# MAGIC - 二重指数平滑化（別名：ホルトの線形平滑化）
# MAGIC   - トレンドはあるが季節性がないデータセットに使用します。
# MAGIC - 三重指数平滑化（ホルト・ウィンタース指数平滑化とも呼ばれる）
# MAGIC   - トレンドと季節性の両方を持つデータセットに使用されます。
# MAGIC 
# MAGIC 今回の場合、コロナウィルスのデータセットには明確なトレンドがありますが、季節性は特に重要ではないので、二重指数平滑化を使用することにします。

# COMMAND ----------

# MAGIC %md <i18n value="294eaf9b-8ba0-4137-bb52-d27b39f3d34f"/>
# MAGIC 
# MAGIC 他の2つのコラムはすでに予測済みなので、新型コロナウイルスに感染した死亡者数の予測を見てみよう。

# COMMAND ----------

exp_df = pd.DataFrame()
exp_df["date"] = pd.to_datetime(df["date"])
exp_df["deceased"] = df["deceased"]
exp_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="7ddb8a5e-453b-491d-ac0d-3087a2c7f955"/>
# MAGIC 
# MAGIC ホルトの線形平滑化はデータポイントが0より多いものにしか働かないので、該当する行を削除する必要があります(実装では、データポイントが0ではないものを抜き出し)。さらに、DataFrameのインデックスを日付カラムに設定する必要があります。

# COMMAND ----------

exp_df = exp_df[exp_df["deceased"] != 0]
exp_df = exp_df.set_index("date")
exp_df.head()

# COMMAND ----------

# MAGIC %md <i18n value="b4fb0e59-5554-44b0-81b3-efbfeca88e33"/>
# MAGIC 
# MAGIC 幸いなことに、statsmodelがそのほとんどのワークをやってくれています。しかし、正確な予測を得るためには、まだパラメータを微調整する必要があります。ここで利用可能なパラメータはα(または **`smoothing_level`** )とβ(または **`smoothing_slope`** )です。αは平滑化定数を、βはトレンド係数として定義されます。
# MAGIC 
# MAGIC 下のセルでは、3種類の予測を試しています。1つ目のHolt's Linear Trendは、直線的なトレンドで予測します。2つ目のExponential Trendは、指数的なトレンドで予測します。3つ目のAdditive Damped Trendは、予測トレンドを線形に減衰させるものです。

# COMMAND ----------

from statsmodels.tsa.holtwinters import Holt

exp_fit1 = Holt(exp_df.deceased).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
exp_forecast1 = exp_fit1.forecast(30).rename("Holt's linear trend")

exp_fit2 = Holt(exp_df.deceased, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
exp_forecast2 = exp_fit2.forecast(30).rename("Exponential trend")

exp_fit3 = Holt(exp_df.deceased, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
exp_forecast3 = exp_fit3.forecast(30).rename("Additive damped trend")

# COMMAND ----------

# MAGIC %md <i18n value="4e3c58f7-afdc-42c6-805f-6f34770ea4d8"/>
# MAGIC 
# MAGIC 3つのモデルをプロットした結果、標準的なHoltのLinearとExponentialのトレンドラインは非常によく似た予測をするのに対し、Additive Dampedのトレンドは死亡患者数をわずかに低くなっていることがわかります。

# COMMAND ----------

exp_fit1.fittedvalues.plot(color="orange", label="Holt's linear trend")
exp_fit2.fittedvalues.plot(color="red", label="Exponential trend")
exp_fit3.fittedvalues.plot(color="green", label="Additive damped trend")

plt.legend()
plt.ylabel("Number of Deceased Patients")
plt.xlabel("Day Number")
plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="f3bbd647-3586-482f-880a-369258cfc7d0"/>
# MAGIC 
# MAGIC グラフの予測部分を拡大することで、より詳細なグラフを見ることができます。
# MAGIC 
# MAGIC 指数トレンドラインは、線形トレンドラインと一直線に始まり、グラフの終盤で徐々に指数トレンドに似てきていることがわかります。減衰したトレンドラインは、他のトレンドラインより下に始まり、下に終わります。

# COMMAND ----------

exp_forecast1.plot(legend=True, color="orange")
exp_forecast2.plot(legend=True, color="red")
exp_forecast3.plot(legend=True, color="green")

plt.ylabel("Number of Deceased Patients")
plt.xlabel("Day Number")
plt.show()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
