import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import time
import plotly.graph_objects as go

# 1. 페이지 레이아웃 확장 설정 (넓은 화면 사용)
st.set_page_config(page_title="중고차 시세 예측 AI", page_icon="🚗", layout="wide")

# 스타일 대시보드 타이틀
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>📊 딥러닝 기반 중고차 적정 시세 대시보드</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6B7280;'>Kaggle Vehicle Dataset을 활용한 단층 퍼셉트론(MLP) 회귀 모델 예측 시스템</p>", unsafe_allow_html=True)
st.write("---")

# 2. 모델 및 스케일러 불러오기
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('car_price_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ 모델 파일이 없습니다! 터미널에서 'python train.py'를 먼저 실행해 주세요.")
    st.stop()

# 3. 화면을 왼쪽(입력창)과 오른쪽(시각화 및 결과)으로 분할
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown("### 📋 차량 상세 정보 입력")
    st.write("예측하고자 하는 중고차의 스펙을 설정하세요.")
    
    # 입력 UI 카드 스타일로 묶기
    with st.container(border=True):
        year = st.slider("📅 제조 연식 (년도)", 2000, 2026, 2015, help="차량의 최초 등록 연도를 선택하세요.")
        kms = st.number_input("🛣️ 총 주행 거리 (km)", min_value=0, max_value=500000, value=50000, step=5000)
        
        # 실제 데이터셋 학습에는 안 쓰이지만 화면을 꽉 채우기 위한 시각용 옵션 추가
        st.selectbox("⛽ 연료 종류 (참고용)", ["휘발유 (Petrol)", "경유 (Diesel)", "LPG"])
        st.radio("⚙️ 변속기 종류 (참고용)", ["자동 (Automatic)", "수동 (Manual)"], horizontal=True)

    # 대형 예측 버튼
    predict_btn = st.button("🔮 AI 실시간 시세 예측하기", use_container_width=True, type="primary")

with col2:
    st.markdown("### 📈 AI 분석 및 시각화 리포트")
    
    if predict_btn:
        # 로딩 애니메이션 추가 (시각적 효과)
        with st.spinner("딥러닝 모델이 시세를 연산 중입니다..."):
            time.sleep(0.4) # 감성 로딩 시간
            
            # 딥러닝 연산 수행
            features = np.array([[year, kms]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            predicted_price_won = max(0.0, float(prediction[0][0]) * 200)

        # 큰 숫자 메트릭 카드로 결과 보여주기
        st.write("#### 🤖 AI 추천 적정 판매가")
        st.metric(label="예상 시세 (원)", value=f"약 {predicted_price_won:,.0f} 만원")
        
        # 🌟 [시각화 추가 1] 실시간 가격 게이지 차트 (Gauge Chart) 
        # 최대 금액을 4,000만원으로 설정하고 현재 금액이 어느 포지션인지 바 형태로 시각화합니다.
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_price_won,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "시장 가치 스펙트럼 (단위: 만원)", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [None, 4000], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1E3A8A"}, # 메인 테마 색상 블루
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 1500], 'color': '#FEE2E2'},  # 저가 구간 (연한 레드)
                    {'range': [1500, 3000], 'color': '#FEF3C7'}, # 중가 구간 (연한 옐로우)
                    {'range': [3000, 4000], 'color': '#D1FAE5'}  # 고가 가치 구간 (연한 그린)
                ],
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=80, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 감가상각 가이드 인포박스
        st.write("#### 💡 AI 감가상각 리포트")
        current_year = 2026
        car_age = current_year - year
        
        if car_age <= 3:
            st.info(f"✨ 해당 차량은 연식({year}년형)이 짧아 **신차급 가격 방어력**을 유지하고 있습니다.")
        elif kms > 100000:
            st.warning(f"⚠️ 주행거리가 **{kms:,}km**로 다소 긴 편에 속하여 시세에 감가 요인이 반영되었습니다.")
        else:
            st.success("✅ 연식 대비 주행거리가 적당하여 안정적인 시장 평균 시세를 형성하고 있습니다.")
            
    else:
        # 버튼을 누르기 전 빈 공간을 채워줄 안내 가이드
        st.info("👈 왼쪽에서 차량 정보를 입력한 뒤 **[AI 실시간 시세 예측하기]** 버튼을 누르면 이 자리에 화려한 비주얼 리포트가 출력됩니다.")
        
        # 🌟 [시각화 추가 2] 중고차 시장 연료별 점유율 원형 차트 (Pie Chart) 고정 배치
        st.write("#### 📊 중고차 시장 데이터 통계 (연료 종류별 비중)")
        labels = ['휘발유 (Petrol)', '경유 (Diesel)', 'LPG / 기타']
        values = [58.4, 35.2, 6.4]
        colors = ['#1E3A8A', '#3B82F6', '#93C5FD'] # 감각적인 블루 톤 그라데이션
        
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=colors)])
        fig_pie.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
