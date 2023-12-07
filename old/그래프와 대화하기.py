import streamlit as st
import pandas as pd
import seaborn as sns
import statool.eda as eda  # eda 모듈 임포트
import datetime
import numpy as np

st.header("🌲Wep app for EDA : talk to graph")


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
    uploaded_file = st.file_uploader("혹은, 파일을 업로드해주세요!", type=["csv"], help = 'csv파일만 업로드됩니다😥')
with st.sidebar:
    if uploaded_file is not None:
        mydata = "업로드한 데이터"
    else:
        mydata = dataset_name
    if st.checkbox(f'**{mydata}** 불러오기'):
        # df = sns.load_dataset(dataset_name)
        df = eda.load_data(dataset_name, uploaded_file)
        # df = st.session_state['df']
    # # 버튼을 통해 캐시 클리어
    # if st.button('새로운 데이터를 탐색하려면 버튼을 눌러주세요. '):
    #     st.cache_data.clear()  # 모든 memo 캐시 클리어
    #     st.cache_resource.clear()  # 모든 singleton 캐시 클리어
    #     st.write("모든 데이터가 삭제되었습니다.")
    #     st.session_state['data_loaded'] = None
    #     st.session_state['df'] = None
       
st.subheader("👀 데이터 확인하기")
# st.write(df)
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
# st.write(st.session_state['data_loaded'])
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
    if len(selected_columns)>2:
        st.warning("데이터와 대화할 때에는 열을 2개 이하로 선택해주세요.")
    else:
        st.success(selected_columns)
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
            st.success("데이터 유형 변경완료!")

cola = st.session_state('user_column_types')
st.write(cola)