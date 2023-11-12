import streamlit as st
import pandas as pd
import seaborn as sns
import statool.eda as eda  # eda 모듈 임포트
import datetime
import numpy as np
import openpyxl

# # 데이터 로드 함수
# def load_data(dataset_name, uploaded_file):
    
#     if dataset_name:
#         return sns.load_dataset(dataset_name)
#     elif uploaded_file:
#         if uploaded_file.name.endswith('.csv'):
#             return pd.read_csv(uploaded_file)
#         elif uploaded_file.name.endswith('.xlsx'):
#             return pd.read_excel(uploaded_file)

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


# 1. 데이터 불러오기 및 확인
st.header("1. 데이터 불러오기 및 확인")
with st.expander("샘플 데이터 목록을 보려면 펼치세요."):
    st.text(sns.get_dataset_names())

dataset_name = st.text_input('데이터셋 이름 (예: titanic, tips, taxis, penguins, iris):')
uploaded_file = st.file_uploader("파일 업로드", type=["csv", "xlsx"])

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
st.text(eda.palet(5))
if st.session_state['types_set']:
    st.header("4. 데이터 시각화")
    if st.button("시각화하기"):
        converted_df = eda.convert_column_types(df_selected, st.session_state['user_column_types'])
        eda.pairviz(converted_df)

# 5. 재표현하기
if st.session_state['types_set']:
    st.header("5. 재표현하기")
    df_transformed = st.session_state['df'][st.session_state['selected_columns']].copy()
    eda.infer_column_types(df_transformed)
    st.write()
    st.write()
    # st.write(df_transformed.select_dtypes(include=[np.number]))
    # df_transformed = st.session_state['df'][st.session_state['selected_columns']].copy()
    # for col in st.session_state['selected_columns']:
    #     if st.session_state['user_column_types'][col] == 'Numeric':
    #         transformation = st.radio(f"{col} 변환 선택:", ['그대로', '로그변환', '제곱근', '제곱'], key=f"trans_{col}")
    #         st.session_state['transformations'][col] = transformation
    #         if transformation != '그대로':
    #             df_transformed = eda.transform_numeric_data(df_transformed, col, transformation)
    # if st.button('재표현 후 시각화하기'):
    #     eda.pairviz(df_transformed)
# # app.py
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import statool.eda as eda  # eda 모듈 임포트
# import datetime
# # 데이터 로드

# # 위젯 호출 및 데이터 로드
# tab1, tab2 = st.tabs(["seaborn 데이터셋", "파일 업로드"])
# with tab1:
#     dataset_name = st.text_input('데이터 예시: titanic, tips, taxis, penguins, iris...:')
# with tab2:
#     uploaded_file = st.file_uploader("분석하고 싶은 파일을 업로드해주세요.", type=["csv", "xlsx"])

# if st.button('데이터 불러오기'):
#     df = eda.load_data(dataset_name, uploaded_file)
#     if df is not None:
#         st.session_state['df'] = df
#         st.write(df.head())
#         current_time = datetime.datetime.now()
#         st.write("현재 시간:", current_time)


#     if st.button('데이터 유형 검토하기'):
#         # 데이터 유형 추론 및 사용자 입력을 통한 커스터마이징
#         inferred_types = eda.infer_column_types(df)
#         user_column_types = {}
#         for col in df.columns:
#             col_type = st.selectbox(f"{col}의 유형 선택", ['Numeric', 'Categorical'], index=0 if inferred_types[col] == 'Numeric' else 1)
#             user_column_types[col] = col_type

#         # 데이터 유형에 따른 변환
#         converted_df = eda.convert_column_types(df, user_column_types)

#         if st.button('데이터 기초통계량 확인하기'):
#             st.write(eda.summarize(df))
        
#             if st.button("대략적인 시각화하기"):

#                 # 데이터 시각화
#                 eda.pairviz(converted_df)
