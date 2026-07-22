import os; os.system('python train.py')
import streamlit as st
import numpy as np
import pickle
import time
import plotly.graph_objects as go

st.set_page_config(page_title="중고차 시세 예측 AI", page_icon="🚗", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>📊 머신러닝 기반 중고차 적정 시세 대시보드</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6B7280;'>Kaggle Vehicle Dataset을 활용한 선형 회귀(Linear Regression) 모델 예측 시스템</p>", unsafe_allow_html=True)
st.write("---")

@st.cache_resource
def load_assets():
    # 텐서플로우 대신 피클로 모델 로드
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ 모델 파일을 불러오지 못했습니다.")
    st.stop()

col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown("### 📋 차량 상세 정보 입력")
    with st.container(border=True):
        year = st.slider("📅 제조 연식 (년도)", 2000, 2026, 2015)
        kms = st.number_input("🛣️ 총 주행 거리 (km)", min_value=0, max_value=500000, value=50000, step=5000)
        st.selectbox("⛽ 연료 종류 (참고용)", ["휘발유 (Petrol)", "경유 (Diesel)", "LPG"])
        st.radio("⚙️ 변속기 종류 (참고용)", ["자동 (Automatic)", "수동 (Manual)"], horizontal=True)

    predict_btn = st.button("🔮 AI 실시간 시세 예측하기", use_container_width=True, type="primary")

with col2:
    st.markdown("### 📈 AI 분석 및 시각화 리포트")
    if predict_btn:
        with st.spinner("모델이 시세를 연산 중입니다..."):
            time.sleep(0.2)
            features = np.array([[year, kms]])
            features_scaled = scaler.transform(features)
            # 예측 수행
            prediction = model.predict(features_scaled)
            predicted_price_won = max(0.0, float(prediction) * 200)

        st.write("#### 🤖 AI 추천 적정 판매가")
        st.metric(label="예상 시세 (원)", value=f"약 {predicted_price_won:,.0f} 만원")

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_price_won,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "시장 가치 스펙트럼 (단위: 만원)", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [None, 4000], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1E3A8A"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 1000], 'color': '#FEE2E2'},
                    {'range': [1000, 2500], 'color': '#FEF3C7'},
                    {'range': [2500, 4000], 'color': '#D1FAE5'}
                ],
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=80, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.write("#### 💡 AI 감가상각 리포트")
        car_age = 2026 - year
        if car_age <= 3:
            st.info(f"✨ 해당 차량은 연식({year}년형)이 짧아 **신차급 가격 방어력**을 유지하고 있습니다.")
        elif kms > 100000:
            st.warning(f"⚠️ 주행거리가 **{kms:,}km**로 다소 긴 편에 속하여 시세에 감가 요인이 반영되었습니다.")
        else:
            st.success("✅ 연식 대비 주행거리가 적당하여 안정적인 시장 평균 시세를 형성하고 있습니다.")
    else:
        st.info("👈 왼쪽에서 차량 정보를 입력한 뒤 **[AI 실시간 시세 예측하기]** 버튼을 누르면 이 자리에 화려한 비주얼 리포트가 출력됩니다.")
        labels = ['휘발유 (Petrol)', '경유 (Diesel)', 'LPG / 기타']
        values = [58.4, 35.2, 6.4]
        colors = ['#1E3A8A', '#3B82F6', '#93C5FD']
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=colors)])
        fig_pie.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
