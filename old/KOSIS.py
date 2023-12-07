import streamlit as st
import pandas as pd
import seaborn as sns
import statool.eda as eda  # eda 모듈 임포트
import numpy as np

st.header("🌲Wep app for EDA")
st.success("🎈KOSIS에서 받은 데이터와 같이 원자료가 아닌 경우, 여기에 업로드하세요!")

# 데이터 업로드
uploaded_file = st.file_uploader("파일을 업로드해주세요!", type=["csv"], help = 'csv파일만 업로드됩니다😥')
if uploaded_file:
    df = pd.read_csv(uploaded_file)