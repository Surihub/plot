import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
import numpy as np
from statsmodels.graphics.mosaicplot import mosaic  

# Streamlit 앱 제목 설정
st.title("📊 데이터 시각화 및 분석")

# 데이터셋 불러오기 또는 파일 업로드 선택 창
st.subheader("1️⃣ 데이터 불러오기")

tab1, tab2 = st.tabs(["seaborn 데이터셋", "파일 업로드"])

# 데이터를 불러오는 로직
with tab1:
    dataset_name = st.text_input('데이터 예시: titanic, tips, taxis, penguins, iris...:')
    sample_checked = st.checkbox('seaborn 데이터 확인하기')

    if sample_checked:
        with st.spinner('샘플 데이터를 불러오는 중 입니다...'):
            try:
                df = sns.load_dataset(dataset_name)
                st.write(df.head(3))
            except:
                st.write("⚠데이터셋 이름을 다시 확인해주세요!")

with tab2:
    st.write("단, 이 방법은 csv 파일만 지원합니다.")
    custom_data = st.file_uploader("분석하고 싶은 파12일을 업로드해주세요.", type=["csv", "xlsx"])
    if custom_data:
        custom_data = pd.read_csv(custom_data, encoding = 'euc-kr')
        st.session_state['custom_data'] = custom_data

    upload_checked = st.checkbox('업로드한 파일 확인하기!')
    if upload_checked:
        with st.spinner('중복을 확인하는 중입니다...'):
            try:
                st.write(custom_data.head(5))
                df = custom_data
            except:
                st.write("⚠올바른 파일을 업로드하셨는지 확인해주세요!")

try:
    # 열 이름 가져오기
    column_names = df.columns.tolist()

    # 열 이름을 리스트로 나열하고 클릭 가능하게 만들기
    selected_columns = st.multiselect('분석하고자 하는 열을 선택하세요:', column_names, default=column_names)
    df = df[selected_columns]
    # 선택된 열의 데이터 표시

    if selected_columns:
        st.write(df.head(3))
#################################
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

        # 사용자가 각 열의 데이터 유형을 설정할 수 있도록 입력 받기
        user_column_types = {}
        for column, col_type in column_types.items():
            user_col_type = st.selectbox(
                f"Select the column type for '{column}' (Current: {col_type})",
                ['Numeric', 'Categorical'],
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

        # st.write('Data with updated column types:')
        # st.dataframe(df)
#################################

# ["기술통계량", "데이터 시각화"]
    
    st.subheader('2️⃣ 기술통계량')
    
# ########################
    # 각 열에 대한 기술통계량 또는 빈도표 생성
    for column, col_type in user_column_types.items():
        st.write(f"### {column} ({col_type})")
        if col_type == 'Numeric':
            st.write(pd.DataFrame(df[column].describe()).T)
        elif col_type == 'Categorical':
            st.write(pd.DataFrame(df[column].value_counts()).T.style.background_gradient(axis=1))
#################################
    st.subheader('3️⃣ 시각화 : 일변량')
    # 각 열에 대한 시각화 생성
    for column, col_type in column_types.items():
        with st.container():  # st.container를 사용하여 그래프들을 감싼다.
            with st.expander(f"{column} - {col_type} 데이터 시각화"):
                # 하나의 figure에 두 개의 subplot을 생성한다.
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

                if col_type == 'Numeric':
                    # 첫 번째 subplot에 히스토그램을 그린다.
                    sns.histplot(df[column], kde=False, ax=ax1)
                    ax1.set_title(f"Histogram of {column}")

                    # 두 번째 subplot에 상자그림을 그린다.
                    sns.boxplot(x=df[column], ax=ax2)
                    ax2.set_title(f"Boxplot of {column}")

                elif col_type == 'Categorical':
                    # 첫 번째 subplot에 막대그래프를 그린다.
                    sns.countplot(x=df[column], ax=ax1)
                    ax1.set_title(f"Countplot of {column}")

                    # 두 번째 subplot에 원 그래프를 그린다.
                    df[column].value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%')
                    ax2.set_title(f"Pie chart of {column}")

                # 그래프를 스트림릿에 표시한다.
                st.pyplot(fig)
    st.subheader("test")
    # 격자의 크기를 정의합니다. n x n 격자
    n = len(df.columns)
    fig, axes = plt.subplots(n, n, figsize=(5 * n, 5 * n))

    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            ax = axes[i, j]
            if i != j:
                # 수치형 * 수치형 = 산점도
                if column_types[col1] == 'Numeric' and column_types[col2] == 'Numeric':
                    sns.scatterplot(data=df, x=col1, y=col2, ax=ax)
                # 범주형 * 수치형 = 상자그림
                elif column_types[col1] == 'Categorical' and column_types[col2] == 'Numeric':
                    sns.boxplot(data=df, x=col1, y=col2, ax=ax)
                # 수치형 * 범주형 = 상자그림 (순서 바꿈)
                elif column_types[col1] == 'Numeric' and column_types[col2] == 'Categorical':
                    sns.boxplot(data=df, x=col2, y=col1, ax=ax)
                # 범주형 * 범주형 = 모자이크플롯
                elif column_types[col1] == 'Categorical' and column_types[col2] == 'Categorical':
                    mosaic(df, [col1, col2], ax=ax)
                # 그래프 제목 설정
                ax.set_title(f'{col1} vs {col2}')
            else:
                # 같은 열의 조합에는 히스토그램 또는 카운트 플롯
                if column_types[col1] == 'Numeric':
                    sns.histplot(df[col1], kde=False, ax=ax)
                else:
                    sns.countplot(x=df[col1], ax=ax)
                # 대각선 그래프 제목 설정
                ax.set_title(f'Distribution of {col1}')

    # 서브플롯들 사이의 여백을 조정
    plt.tight_layout()
    st.pyplot(fig)



    st.subheader('3️⃣ 시각화 : 이변량')
    # 각 열에 대한 시각화 생성
    # 모든 열 쌍에 대한 그래프를 그린다
    for col1 in df.columns:
        for col2 in df.columns:
            # 동일한 열을 비교하지 않는다
            if col1 != col2:
                # 수치형 * 수치형 = 산점도
                if column_types[col1] == 'Numeric' and column_types[col2] == 'Numeric':
                    st.write(f"### {col1} and {col2} - 산점도")
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x=col1, y=col2, ax=ax)
                    st.pyplot(fig)

                # 범주형 * 수치형 = 상자그림
                elif column_types[col1] == 'Categorical' and column_types[col2] == 'Numeric':
                    st.write(f"### {col1} and {col2} - 상자그림")
                    fig, ax = plt.subplots()
                    sns.boxplot(data=df, x=col1, y=col2, ax=ax)
                    st.pyplot(fig)

                elif column_types[col1] == 'Numeric' and column_types[col2] == 'Categorical':
                    st.write(f"### {col2} and {col1} - 상자그림")
                    fig, ax = plt.subplots()
                    sns.boxplot(data=df, x=col2, y=col1, ax=ax)
                    st.pyplot(fig)

                # 범주형 * 범주형 = 모자이크플롯
                elif column_types[col1] == 'Categorical' and column_types[col2] == 'Categorical':
                    from statsmodels.graphics.mosaicplot import mosaic  
                    st.write(f"### {col1} and {col2} - 모자이크플롯")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    mosaic(df, [col1, col2], ax=ax)
                    plt.title(f'Mosaic plot of {col1} and {col2}')
                    st.pyplot(fig)

except NameError as e:
    st.warning('데이터를 먼저 입력해주세요. ')