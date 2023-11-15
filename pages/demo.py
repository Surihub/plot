import streamlit as st
import pandas as pd
import seaborn as sns
import statool.eda as eda  # eda 모듈 임포트
import datetime
import numpy as np

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


# 1. 데이터 불러오기 및 확인
st.header("1. 데이터 불러오기 및 확인")
with st.expander("샘플 데이터 목록을 보려면 펼치세요."):
    st.text(sns.get_dataset_names())

dataset_name = st.text_input('데이터셋 이름 (예: titanic, tips, taxis, penguins, iris):')
uploaded_file = st.file_uploader("파일 업로드", type=["csv"])

if st.button('데이터 불러오기'):
    df = eda.load_data(dataset_name, uploaded_file)
    if df is not None:
        st.session_state['df'] = df
        st.session_state['data_loaded'] = True
        st.write("데이터 로드 완료!")


# 2. 열 선택
if st.session_state['data_loaded']:

    st.header("2. 열 선택")
    if st.checkbox('모든 열 선택하기', key='select_all'):
        default_columns = st.session_state['df'].columns.tolist() if 'select_all' in st.session_state and st.session_state['select_all'] else []
    else:
        default_columns = st.session_state['df'].columns.tolist() if 'selected_columns' not in st.session_state else st.session_state['selected_columns']

    selected_columns = st.multiselect('분석하고자 하는 열을 선택하세요:', st.session_state['df'].columns.tolist(), default=default_columns)
    st.write(st.session_state['df'][selected_columns].head())

    st.session_state['selected_columns'] = selected_columns
    if st.button('열 선택 완료!'):
        st.session_state['columns_selected'] = True
        st.success("열 선택 완료!")

# 3. 데이터 유형 변경
if st.session_state['columns_selected']:
    st.header("3. 데이터 유형 변경")
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
            for column, col_type in dict1.items():
                default_index = options_en.index(col_type)
                user_col_type = st.radio(
                    f"'{column}'의 유형:",
                    options_kr,
                    index=default_index,
                    key=column
                )
                user_column_types[column] = options_dic[user_col_type]

        with col2:
            for column, col_type in dict2.items():
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
    st.header("4. 데이터 요약과 시각화")
    converted_df = eda.convert_column_types(df_selected, st.session_state['user_column_types'])
    st.session_state['converted_df'] = converted_df
    st.write(converted_df.head(2))
    tab1, tab2  = st.tabs(['기술통계량 확인하기', '데이터 시각화'])
    with tab1:
        # 각 열에 대한 기술통계량 또는 빈도표 생성
        for column, col_type in user_column_types.items():
            st.write(f"**{column}** ({col_type})")
            if col_type == 'Numeric':
                st.write(pd.DataFrame(converted_df[column].describe()).T)
            elif col_type == 'Categorical':
                st.write(pd.DataFrame(converted_df[column].value_counts()).T.style.background_gradient(axis=1))
    with tab2:
        
        eda.pairviz(converted_df)
        st.session_state['viz'] = True


# 5. 재표현하기
if st.session_state['viz']:
    st.header("5. 재표현하기")
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
        eda.pairviz(df_transformed)

st.header("6. 회귀선과 잔차 살펴보기")

st.header("7. 시각화 하나씩 살펴보기")