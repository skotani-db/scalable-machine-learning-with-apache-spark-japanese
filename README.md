# Scalable Machine Learning With Apache Spark [Japanese]

このリポジトリには、様々なラボとその解答に加え、このコースを教える講師に学生がついて行くために必要なリソースが含まれています。

**Special Note:** This course is published in multiple languages via different repos.
* For the English version, see https://github.com/databricks-academy/scalable-machine-learning-with-apache-spark-english
* For the Japanese version, see https://github.com/databricks-academy/scalable-machine-learning-with-apache-spark-japanese (this repo)

開始するには2つの方法があります（Databricks Reposを使用する方法と使用しない方法）。講師は、あなたがいつ、どちらの手順を使うべきかを指示します。

このリポジトリでは両方の手順を日本語で文書化しています。

# 始めよう！
workspaceへアセットをインポートする方法は2つあります。

1つ目の方法は <a href="https://docs.databricks.com/repos.html" target="_blank">Databricks Repos</a>を使います。この機能はGitプロバイダとリポジトリレベルで統合する機能です(この場合は<a href="https://github.com/" target="_blank">GitHub</a>)。<br/>
<img src="https://files.training.databricks.com/images/icon_note_32.png"> GitHubアカウントはこの方法には必要 **ありません**。

2つ目の方法は、DBCファイル（ノートブックのコレクションを含むアーカイブファイル）をワークスペースにインポートすることです。

Databricks Academyでは、従来から2番目の方法で配布していましたが、現在ではDatabricks Reposが好ましい方法であり、
可能な限りDatabricks Academyが強く推奨する方法です。


# Databricks Reposで始める
1. GitHub上で:
   1.  **Code** をクリック
   1.  **HTTPS** オプションを選択
   1.  **Copy** アイコンをクリックして、リポジトリのURLをクリップボードへコピーします<br/>
   
   ![](https://github.com/skotani-db/data-engineering-with-databricks/blob/753dc9d03532fc8cee87081310bef6824552d135/images/copy-url.png)

1. Databricksで、 左のペインにある **Repos** アイコン![](https://github.com/shotaroktn-db/data-engineering-with-databricks/blob/main/images/repos-icon.png)をクリック
1. デフォルトで */Repos/* 配下にあるユーザー個別のフォルダへ移動します
1.  **Add Repo** ボタンをクリックします<br/>
![](https://github.com/skotani-db/data-engineering-with-databricks/blob/753dc9d03532fc8cee87081310bef6824552d135/images/add-repo.png)
1. 
   **Add Repo** のダイアログボックスで:
   1.  **Clone remote Git repo** を選択
   1. Step1のGithubリポジトリURLを貼り付け
   1.  **Git provider** フィールドは自動で入力されます
   1.  **Repo name** フィールドも同様に自動で入力されますが、好きなように名前を変更いただいて構いません
   1.  **Create** をクリック
1. この教材を見るにはリポジトリを表すフォルダに移動してください
1. コースのインストラクションまたは講師から提供されたノートブックから始めましょう

# DBCファイルで始める (Databricks Reposを使わない)
1. GitHubで右側のペインにある **Releases** の下、 **Latest** リンクをクリックする<br/>
![](https://github.com/skotani-db/data-engineering-with-databricks/blob/753dc9d03532fc8cee87081310bef6824552d135/images/latest.png)
2.  **Assets** の下でDBCファイルのリンクを探します<br/>
![](https://github.com/skotani-db/data-engineering-with-databricks/blob/753dc9d03532fc8cee87081310bef6824552d135/images/dbc.png)
4. DBCファイルのリンクを右クリックして、ロケーションのリンクをコピーします(このファイルをダウンロードする必要はありません)<br/>
![](https://github.com/skotani-db/data-engineering-with-databricks/blob/753dc9d03532fc8cee87081310bef6824552d135/images/clone.png)
4. Databricksに戻って、 **Workspace** アイコン![](https://github.com/skotani-db/data-engineering-with-databricks/blob/753dc9d03532fc8cee87081310bef6824552d135/images/workspace.png)をクリック 
5. 左のナビゲーションペインから出てきた**Workspace** の右上,  **Home** ボタンをクリックしてユーザーのホームフォルダを開きます<br/> **/Users/student@example.com** のように　**/Users/your-email-address** のフォルダが開かれるはずです
6. あなたのemailアドレスのスイムレーンで、逆V字のマークをクリックして **Import** を選びます<br/>
![](https://github.com/skotani-db/data-engineering-with-databricks/blob/753dc9d03532fc8cee87081310bef6824552d135/images/import.png)<br/>
   **Import Notebooks** ダイアログで
   1.  **URL** を選択
   2. 上のStep3でコピーしたURLを貼り付け
   3.  **Import** をクリック<br/>
![](https://github.com/skotani-db/data-engineering-with-databricks/blob/753dc9d03532fc8cee87081310bef6824552d135/images/import-notebook.png)
8. 一度インポートが完了したら、コースのnotebookを見るために新しくできたフォルダを選択します
9. どのnotebookから始めるかはコースやインストラクター次第です


