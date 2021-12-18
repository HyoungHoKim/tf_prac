# 키로 신발 크기 추정 AI

import tensorflow as tf

키 = 170
신발 = 260
# 신발 = 키 * a + b

# weight
a = tf.Variable(0.1)
b = tf.Variable(0.2)

def 손실함수():
    return tf.square(260 - (키 * a + b)) # 손실 값 = (실제값 - 예측값)^2

# 경사 하강법
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
	opt.minimize(손실함수, var_list=[a,b])
	print(a.numpy(), b.numpy())