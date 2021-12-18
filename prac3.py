import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import EarlyStopping

# input_shape : 어떤 형태의 데이터가 들어가는지
model = Sequential([
    Flatten(input_shape=(28, 28)),	
	Dense(units=128, activation='relu'),
	Dense(units=10, activation='softmax')
])

# optimizer : 러닝 과정 중 오차(loss)를 최소화 시키는 알고리즘 
# Loss : 오차 측정 알고리즘
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# mnist : 데이터 예제
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 조기 종료
# moniter : val_accuracy 관찰, paitence : 3번까지 개선되지 않을 시 종료
# min_delta : 기준 미달시 조기 종료, 3epoch동안 손실이 0.05이상 개선되지 않을 경우 훈련 중지
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.05)

# model.fit(x_train, y_train) : 훈련 함수
# x_train : feature(data)
# y_train : lable(target)
# epochs : 훈련 회수
# batch_size : 한번의 배치마다 전체 데이터에서 일부를 불러오는 사이즈
histoty = model.fit(x_train, y_train, epochs=10, batch_size=64, callbacks=[early_stopping])

# 훈련과 검증 동시에
# 기존 훈련 데이터에서 20%짤라서 검증을 함께 수행
# model.fit(x, y, validation_split=0.2, epochs=5, batch_size=64)

# 평가(정확도)
loss, accuracy = model.evaluate(x_test, y_test)
print("Test 데이터 정확도: ", accuracy)

"""
1. 모델 가중치 저장
from tensorflow.keras.callbacks import ModelCheckpoint

cp_path = 'model_save/cp.ckpt'
checkpoint = ModelCheckpoint(filepath=cp_path,
                save_best_only=True,
                save_weights_only=True
                verbose=1)
                
model.fit(x_train, y_train, epochs=10, batch_size=64, callbacks=[checkpoint])

1-1. 저장된 모델 가중치 불러오기
model.load_weights(cp_path)

2. 모델 저장 
from tensorflow.keras.callbacks import ModelCheckpoint

# 방법1
model = load_model()
checkpoint = ModelCheckpoint('model_save.h5') # save_weights_only=False
model.fit(x_train, y_train, epochs=10, callbacks=[checkpoint])

# 방법2
model = load_model()
model.fit(x_train, y_train, epochs=3)
model.save('model_save2.h5') 

2-1. 모델 불러오기
from tensorflow.keras.models import load_model

model = load_model('model_save2.h5')
model.evaluate(x_test, y_test)
"""