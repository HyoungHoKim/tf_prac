# 대학 합격 예측 AI만들기
import pandas as pd # 행과 열로 된 데이터를 다루기위해 쓰는 라이브러리 

data = pd.read_csv('gpascore.csv')
print(data)

# 데이터 전처리하기
# print(data.isnull().sum()) # 빈 데이터를 찾는 함수
data = data.dropna() # 빈 데이터 열 지우기
# data.fillna(100) # 빈 칸에 설정한 값(100) 대입

y데이터 = data['admit'].values
x데이터 = []

for i, rows in data.iterrows():
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']])

# 딥 러닝
import numpy as np
import tensorflow as tf

# Sequential을 쓰면 신경망 레이어들을 쉽게 만들어줌
# deep Learning model을 만든것임
# 일반적인 히든 레이어 : tf.keras.layers.Dense()
model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(64, activation='tanh'),	 
	tf.keras.layers.Dense(128, activation='tanh'),
	tf.keras.layers.Dense(1, activation='sigmoid'),
])

# binary_crossentropy = 분류 및 확률 문제에서 많이 씀
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(x데이터(데이터), y데이터(답), epochs=10) => 데이터를 바탕으로 10번 학습 시킨다.

model.fit(np.array(x데이터), np.array(y데이터), epochs=1000)

# 예측
예측값 = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(예측값)