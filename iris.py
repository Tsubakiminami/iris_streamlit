## キカガク
## Python アプリケーション開発コースStreamlit でのアプリ開発Streamlit の基礎
## iris.py
## 2023/09/21-09/22
##
## 実行 streamlit run iris.py
## 保存したファイル iris.py を実行する
## 保存 cmd+s

# 基本ライブラリーをインポート
from typing_extensions import dataclass_transform
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# 【注意】
# scikit-learn がローカルにインストールできていない場合、エラーが表示されます。
# 各種シェル上から、scikit-learn をローカルにインストールしておきましょう。
# pip3 install scikit-learn

st.write('Pandas version',pd.__version__)
st.write('Numpy version',np.__version__)
st.write('Streamlit version',st.__version__)

# データセット読み込み
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 目標値
df['target'] = iris.target

# 目標値を数字から花の名前に変更
df.loc[df['target'] == 0, 'target'] = 'setosa'
df.loc[df['target'] == 1, 'target'] = 'versicolor'
df.loc[df['target'] == 2, 'target'] = 'virginica'
# df = df.astype({'target':'categorical'})

# 予測モデルの構築
x = iris.data[:, [0, 2]]    # 全ての行で、0番目と2番目の列を抽出
y = iris.target

# ロジスティック回帰モデルの構築
clf = LogisticRegression()
clf.fit(x, y)

# サイドバーの入力画面
st.sidebar.header('Input features')

sepalValue = st.sidebar.slider('Sepal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider('Petal length (cm)', min_value=0.0, max_value=10.0, step=0.1)

# メインパネル
st.title('Iris classifier')
st.write('## Input value')

# インプットデータ（1行のデータフレーム）
value_df = pd.DataFrame([],columns=['data','sepal length (cm)','petal length (cm)'])
# record = pd.Series(['data',sepalValue, petalValue], index=value_df.columns)
record = pd.DataFrame([['data:',sepalValue, petalValue]],columns=value_df.columns)
# value_df = value_df.append(record, ignore_ind:ex=True)
value_df = pd.concat([value_df, record], ignore_index=True)
value_df.set_index('data',inplace=True)


# 入力値の値
st.write(value_df)

# # 予測値のデータフレーム
pred_probs = clf.predict_proba(value_df)    # array([[0.00697739, 0.64845897, 0.34456364]])
pred_df = pd.DataFrame(pred_probs,columns=['setosa','versicolor','virginica'],index=['probability'])

st.write('## Prediction')
st.write(pred_df)

# 予測結果の出力
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write('このアイリスはきっと',str(name[0]),'です!')
