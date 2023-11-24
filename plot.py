import streamlit as st
import pandas as pd
import seaborn as sns
import statool.eda as eda  # eda 모듈 임포트
import datetime
import numpy as np

st.header("🌲Wep app for EDA")
st.success("🎈EDA(Exploratory Data Analysis, 탐색적 데이터 분석)이란 간단한 그래프로 데이터의 특징과 패턴을 찾아내어 데이터를 탐구하기 위한 과정입니다. 왼쪽의 사이드바에서 데이터를 선택하거나 업로드하고, 순서에 따라 탐색을 진행해보세요. \n\n✉버그 및 제안사항 등 문의: sbhath17@gmail.com")

# 스트림릿 세션 상태 초기화
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'selected_columns' not in st.session_state:
    st.session_state['selected_columns'] = None
if 'user_column_types' not in st.session_state:
    st.session_state['user_column_types'] = None
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'columns_selected' not in st.session_state:
    st.session_state['columns_selected'] = False
if 'types_set' not in st.session_state:
    st.session_state['types_set'] = False
if 'transformations' not in st.session_state:
    st.session_state['transformations'] = {}
if 'viz' not in st.session_state:
    st.session_state['viz'] = {} 

dataset_name = st.sidebar.selectbox("분석하고 싶은 데이터를 선택해주세요!",
    sns.get_dataset_names(), index = 16, help = "처음이시라면, 귀여운 펭귄들의 데이터인 'penguins'를 추천드려요😀")
with st.sidebar:
    uploaded_file = st.file_uploader("혹은, 파일을 업로드해주세요!", type=["csv"], help = 'csv파일 업로드는 준비중입니다😥')
with st.sidebar:
    if uploaded_file is not None:
        mydata = "업로드한 데이터"
    else:
        mydata = dataset_name
    if st.checkbox(f'**{mydata}** 불러오기'):
        df = eda.load_data(dataset_name, uploaded_file)
        df = st.session_state['df']
    # # 버튼을 통해 캐시 클리어
    # if st.button('새로운 데이터를 탐색하려면 버튼을 눌러주세요. '):
    #     st.cache_data.clear()  # 모든 memo 캐시 클리어
    #     st.cache_resource.clear()  # 모든 singleton 캐시 클리어
    #     st.write("모든 데이터가 삭제되었습니다.")
    #     st.session_state['data_loaded'] = None
    #     st.session_state['df'] = None
       
st.subheader("👀 데이터 확인하기")

try:
    if df is not None:
        st.session_state['df'] = df
        st.session_state['data_loaded'] = True
        st.write("데이터 로드 완료! 불러온 데이터셋은 다음과 같습니다. ")
        st.write(df.head())
        with st.expander('전체 데이터 보기'):
            st.write(df)
except:
    st.error("사이드바에서 먼저 데이터를 선택 후 <데이터 불러오기> 버튼을 클릭해주세요. ")

# 2. 열 선택
if st.session_state['data_loaded']:
    df = st.session_state['df']
    st.subheader("👈 분석할 열 선택하기")
    st.success("위의 데이터셋에서, 분석할 변수만 선택해주세요.")
    if st.checkbox('모든 열 선택하기', key='select_all', value = df.columns.all()):
        default_columns = df.columns.tolist() if 'select_all' in st.session_state and st.session_state['select_all'] else []
    else:
        default_columns = df.columns.tolist() if 'selected_columns' not in st.session_state else st.session_state['selected_columns']

    selected_columns = st.multiselect('분석하고자 하는 열을 선택하세요:', st.session_state['df'].columns.tolist(), default=default_columns)
    st.write(df[selected_columns].head())

    st.session_state['selected_columns'] = selected_columns
    if st.button('열 선택 완료!'):
        st.session_state['columns_selected'] = True
        st.success("열 선택 완료!")

# 3. 데이터 유형 변경
if st.session_state['columns_selected']:
    st.subheader("🙄 데이터 유형 변경")
    st.success("데이터를 살펴보고, 각 변수가 수치형인지, 범주형인지 확인해보세요.")
    if st.session_state['selected_columns'] is not None:
        df_selected = st.session_state['df'][st.session_state['selected_columns']]
        inferred_types = eda.infer_column_types(df_selected)
        user_column_types = {}

        options_en = ['Numeric', 'Categorical']
        options_kr = ["수치형", "범주형"]
        options_dic = {'수치형': 'Numeric', '범주형': 'Categorical'}
        
        # 반반 나눠서 나열
        col1, col2 = st.columns(2)
        keys = list(inferred_types.keys())
        half = len(keys) // 2 

        dict1 = {key: inferred_types[key] for key in keys[:half]}
        dict2 = {key: inferred_types[key] for key in keys[half:]}

        with col1:
            for column, col_type in dict2.items():
                default_index = options_en.index(col_type)
                user_col_type = st.radio(
                    f"'{column}'의 유형:",
                    options_kr,
                    index=default_index,
                    key=column
                )
                user_column_types[column] = options_dic[user_col_type]



        with col2:
            for column, col_type in dict1.items():
                default_index = options_en.index(col_type)
                user_col_type = st.radio(
                    f"'{column}'의 유형:",
                    options_kr,
                    index=default_index,
                    key=column
                )
                user_column_types[column] = options_dic[user_col_type]
        # for col in df_selected.columns:
        #     col_type = st.selectbox(f"{col} 유형 선택", ['Numeric', 'Categorical'], index=0 if inferred_types[col] == 'Numeric' else 1, key=col)
        #     user_column_types[col] = col_type
        if st.button('유형 변경 완료!'):
            st.session_state['user_column_types'] = user_column_types
            st.session_state['types_set'] = True
            st.success("데이터 유형 변경 완료!")

# 4. 데이터 시각화
if st.session_state['types_set']:
    st.subheader("📊 데이터 요약과 시각화")
    converted_df = eda.convert_column_types(df_selected, st.session_state['user_column_types'])
    st.session_state['converted_df'] = converted_df
    # st.write(converted_df.head(2))
    tab1, tab2  = st.tabs(['기술통계량 확인하기', '데이터 시각화'])
    with tab1:
        st.warning("각 변수마다 일변량, 이변량 데이터를 시각화하고 있어요. 오래 걸릴 수 있으니 기다려주세요!")
        eda.모든_그래프_그리기(converted_df)
        st.session_state['viz'] = True
    with tab2:
        # 각 열에 대한 기술통계량 또는 빈도표 생성
        for column, col_type in user_column_types.items():
            st.write(f"**{column}** ({col_type})")
            if col_type == 'Numeric':
                st.write(pd.DataFrame(converted_df[column].describe()).T)
            elif col_type == 'Categorical':
                st.write(pd.DataFrame(converted_df[column].value_counts()).T.style.background_gradient(axis=1))


# 5. 재표현하기
if st.session_state['viz']:
    st.subheader("🤓 재표현하기")
    st.success("수치형 데이터의 경우, 한쪽으로 쏠린 분포를 띤다면 변환을 통해 분석을 쉽게 하기도 합니다. 원하는 변수만 변환을 해보세요.")
    converted_df = st.session_state['converted_df']
    # st.write(eda.infer_column_types(converted_df))

    df_transformed = converted_df.copy()
    for col in st.session_state['selected_columns']:
        if st.session_state['user_column_types'][col] == 'Numeric':
            transformation = st.radio(f"{col} 변환 선택:", ['그대로', '로그변환', '제곱근', '제곱'], key=f"trans_{col}")
            st.session_state['transformations'][col] = transformation
            if transformation != '그대로':
                df_transformed = eda.transform_numeric_data(df_transformed, col, transformation)
    st.write(df_transformed.head())
    if st.button('재표현 후 시각화하기'):
        eda.모든_그래프_그리기(df_transformed)

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import koreanize_matplotlib

if st.session_state['viz']:
    st.subheader("🔍 회귀선과 잔차 살펴보기")
    
    # 데이터프레임 복사
    df_residual = converted_df.copy()

    # 수치형 열 선택
    num_col = []
    for col in st.session_state['selected_columns']:
        if st.session_state['user_column_types'][col] == 'Numeric':
            num_col.append(col)

    # 열 선택
    selected_col = st.multiselect("두 열을 선택해주세요.", num_col)

    if len(selected_col) > 1:
        x = selected_col[0]
        y = selected_col[1]
        eda.plot_residuals(df_residual[[x,y]], x, y)
        # df_residual = df_residual[selected_col]
        # st.write(df_residual)

        # # 선형 회귀 모델 피팅
        # model = LinearRegression()
        # model.fit(df_residual[[selected_col[0]]], df_residual[selected_col[1]])

        # # 예측 및 잔차 계산
        # df_residual['predicted'] = model.predict(df_residual[[selected_col[0]]])
        # df_residual['residuals'] = df_residual[selected_col[1]] - df_residual['predicted']

        # # 잔차 그래프 생성
        # fig, ax = plt.subplots()
        # ax.scatter(df_residual[selected_col[0]], df_residual['residuals'])
        # ax.axhline(y=0, color='r', linestyle='--')
        # ax.set_xlabel(selected_col[0])
        # ax.set_ylabel('Residuals')
        # st.pyplot(fig)



# st.header("7. 다양한 시각화 살펴보기")