import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# 1. 데이터 불러오기
df = pd.read_csv('car data.csv')

# 2. 필요한 특성(X)과 정답(y) 분리
X = df[['Year', 'Kms_Driven']]
y = df['Selling_Price']

# 3. 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 딥러닝 회귀 모델 설계
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 6. 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("🤖 딥러닝 모델 학습을 시작합니다...")
model.fit(X_train_scaled, y_train, epochs=50, batch_size=4, validation_split=0.1, verbose=1)

# 7. 최신 포맷(.keras)으로 모델 및 스케일러 저장
model.save('car_price_model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("🎉 모델학습 완료! 파일이 생성되었습니다.")
