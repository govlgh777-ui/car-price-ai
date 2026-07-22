import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# 데이터 불러오기 및 분리
df = pd.read_csv('car data.csv')
X = df[['Year', 'Kms_Driven']]
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 가벼운 회귀 모델로 변경 (텐서플로우 사용 안 함)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 파일 저장 (확장자를 일반 피클 파일로 변경)
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("🎉 초경량 모델 생성 완료!")
