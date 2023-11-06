import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.mosaicplot import mosaic  
import numpy as np
import koreanize_matplotlib

# Streamlit 앱 제목 설정
st.title("📊 데이터 시각화 및 분석")

# 데이터셋 불러오기 또는 파일 업로드 선택 창
st.subheader("1️⃣ 데이터 불러오기")

tab1, tab2 = st.tabs(["seaborn 데이터셋", "파일 업로드"])

with tab1:
    dataset_name = st.text_input('데이터 예시: titanic, tips, taxis, penguins, iris...:')
    if st.button('seaborn 데이터 불러오기'):
        with st.spinner('샘플 데이터를 불러오는 중 입니다...'):
            try:
                df = sns.load_dataset(dataset_name)
                st.session_state['df'] = df
                st.write(df.head())
            except ValueError:
                st.error("⚠ 데이터셋 이름을 다시 확인해주세요!")

with tab2:
    uploaded_file = st.file_uploader("분석하고 싶은 파일을 업로드해주세요.", type=["csv", "xlsx"])
    if st.button('파일 업로드하기'):
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            st.session_state['df'] = df
            st.write(df.head())
        else:
            st.error("⚠ 파일을 업로드해주세요!")

# 데이터가 불러와졌는지 확인
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # 열 선택
    st.subheader('2️⃣ 열 선택')
    if st.checkbox('모든 열 선택하기', key='select_all'):
        if 'select_all' in st.session_state and st.session_state['select_all']:
            default_columns = df.columns.tolist()
        else:
            default_columns = []
    else:
        default_columns = df.columns.tolist() if 'selected_columns' not in st.session_state else st.session_state['selected_columns']
    
    selected_columns = st.multiselect('분석하고자 하는 열을 선택하세요:', df.columns.tolist(), default=default_columns)
    st.session_state['selected_columns'] = selected_columns

    if selected_columns:
        df = df[selected_columns]
        st.write(df.head())

        # 데이터 유형 설정
        st.subheader('3️⃣ 데이터 유형 설정')
        # if st.button('데이터 유형 업데이트'):
        # 각 열의 데이터 유형을 추론하는 함수
        def infer_column_types(df):
            column_types = {}
            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    column_types[column] = 'Numeric'
                else:
                    column_types[column] = 'Categorical'
            return column_types

        column_types = infer_column_types(df)
        st.session_state['column_types'] = column_types

        # 사용자가 각 열의 데이터 유형을 설정할 수 있도록 입력 받기
        user_column_types = {}
        options_en = ['Numeric', 'Categorical']
        # options_kr = {'Numeric': '수치형', 'Categorical': '범주형'}
        options_kr = ["수치형", "범주형"]
        options_dic = {'수치형':'Numeric',  '범주형':'Categorical'}
        # col1, col2 = st.columns(2):
        
        # 반반 나눠서 나열
        col1, col2 = st.columns(2)
        keys = list(column_types.keys())
        half = len(keys) // 2 

        dict1 = {key: column_types[key] for key in keys[:half]}
        dict2 = {key: column_types[key] for key in keys[half:]}


        with col1:
            for column, col_type in dict1.items():
                default_index = options_en.index(col_type)  # Get the index of the current col_type
                user_col_type = st.radio(
                    f"'{column}'의 유형:",
                    options_kr,
                    index=default_index,  # Set the default value
                    key=column
                )
                user_column_types[column] = user_col_type
        with col2:
            for column, col_type in dict2.items():
                default_index = options_en.index(col_type)  # Get the index of the current col_type
                user_col_type = st.radio(
                    f"'{column}'의 유형:",
                    options_kr,
                    index=default_index,  # Set the default value
                    key=column
                )
                user_column_types[column] = user_col_type

        # 사용자의 입력에 따라 DataFrame의 열 유형을 변환
        def convert_column_types(df, user_column_types):
            for column, col_type in user_column_types.items():
                if col_type == 'Numeric':
                    df[column] = pd.to_numeric(df[column], errors='coerce')  # 범주형을 수치형으로 변환
                elif col_type == 'Categorical':
                    df[column] = df[column].astype('category')  # 수치형을 범주형으로 변환
            return df

        # 열 유형 변환 실행
        df = convert_column_types(df, user_column_types)            
        # ... (데이터 유형 변환 코드)
        st.session_state['df_updated'] = df
        st.session_state['user_column_types'] = user_column_types
        st.success('데이터 유형이 업데이트되었습니다.')
            # st.write(user_column_types)

if 'df_updated' in st.session_state:
    df_updated = st.session_state['df_updated']
    user_column_types = st.session_state['user_column_types']
    # 기술통계량 확인
    st.subheader('4️⃣ 기술통계량 및 자동 데이터 시각화')

    tab1, tab2  = st.tabs(['기술통계량 확인하기', '자동 데이터 시각화'])
    with tab1:
        # 각 열에 대한 기술통계량 또는 빈도표 생성
        for column, col_type in user_column_types.items():
            st.write(f"**{column}** ({col_type})")
            if col_type == '수치형':
                st.write(pd.DataFrame(df_updated[column].describe()).T)
            elif col_type == '범주형':
                st.write(pd.DataFrame(df_updated[column].value_counts()).T.style.background_gradient(axis=1))
    with tab2:
        n = len(df_updated.columns)
        total_plots = n * n
        completed_plots = 0
        pal = sns.color_palette("Set2")

        # 프로그레스 바 초기화
        progress_bar = st.progress(0)

        fig, axes = plt.subplots(n, n, figsize=(5 * n, 5 * n))

        for i, col1 in enumerate(df_updated.columns):
            for j, col2 in enumerate(df_updated.columns):
                ax = axes[i, j]
                if i != j:
                    # 수치형 * 수치형 = 산점도
                    if user_column_types[col1] == '수치형' and user_column_types[col2] == '수치형':
                        sns.scatterplot(data=df_updated, x=col1, y=col2, ax=ax, palette = pal)
                    # 범주형 * 수치형 = 상자그림
                    elif user_column_types[col1] == '범주형' and user_column_types[col2] == '수치형':
                        sns.boxplot(data=df_updated, x=col1, y=col2, ax=ax, palette = pal)
                    # 수치형 * 범주형 = 상자그림 (순서 바꿈)
                    elif user_column_types[col1] == '수치형' and user_column_types[col2] == '범주형':
                        sns.boxplot(data=df_updated, x=col2, y=col1, ax=ax, palette = pal)
                    # 범주형 * 범주형 = 모자이크플롯
                    elif user_column_types[col1] == '범주형' and user_column_types[col2] == '범주형':
                        mosaic(df_updated, [col1, col2], ax=ax)
                    # 그래프 제목 설정
                    ax.set_title(f'{col1} vs {col2}')
                    completed_plots += 1
                    progress_bar.progress((completed_plots / total_plots) )
                else:
                    # 같은 열의 조합에는 히스토그램 또는 카운트 플롯
                    if user_column_types[col1] == '수치형':
                        sns.histplot(df_updated[col1], kde=True, ax=ax, color = 'r')
                    else:
                        sns.countplot(x=df_updated[col1], ax=ax, palette = pal)
                    # 대각선 그래프 제목 설정
                    ax.set_title(f'Distribution of {col1}')

        # 서브플롯들 사이의 여백을 조정
        plt.tight_layout()
        st.pyplot(fig)
        # 프로그레스 바 완료
        progress_bar.empty()
