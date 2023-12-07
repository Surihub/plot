# eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
from statsmodels.graphics.mosaicplot import mosaic
import datetime
import streamlit as st
import numpy as np

@st.cache_data
# 데이터 로드 함수 정의
# def load_data(dataset_name):
#     df = sns.load_dataset(dataset_name)
#     return df
def load_data(dataset_name, uploaded_file):
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding = 'utf-8')
        else:
            st.warning("csv 파일만 업로드 가능합니다. ")
        return df
    elif dataset_name:
        try:
            df = sns.load_dataset(dataset_name)
            return df
        except ValueError:
            st.error("⚠ 데이터셋 이름을 다시 확인해주세요!")

# @st.cache_data
# def select_columns(df):
    



# def load_data(dataset_name, uploaded_file):
#     if dataset_name:
#         try:
#             df = sns.load_dataset(dataset_name)
#             return df
#         except ValueError:
#             st.error("⚠ 데이터셋 이름을 다시 확인해주세요!")
#     elif uploaded_file:
#         if uploaded_file.name.endswith('.csv'):
#             df = pd.read_csv(uploaded_file)
#         else:
#             st.warning("csv 파일만 업로드 가능합니다. ")
#         return df

@st.cache_data
def summarize(df):
    # 기초 통계량 요약 함수
    summ = df.describe()
    summ.index = ['개수', '평균', '표준편차', '최솟값', '제1사분위수', '중앙값', '제3사분위수', '최댓값']
    return summ

@st.cache_data
def convert_column_types(df, user_column_types):
    # 사용자 입력에 따른 데이터 유형 변환
    for column, col_type in user_column_types.items():
        if col_type == 'Numeric':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif col_type == 'Categorical':
            df[column] = df[column].astype('category')
    return df

@st.cache_data
def infer_column_types(df):
    column_types = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            column_types[column] = 'Numeric'
        else:
            column_types[column] = 'Categorical'
    return column_types

    # st.session_state['column_types'] = column_types

    # # 사용자가 각 열의 데이터 유형을 설정할 수 있도록 입력 받기
    # user_column_types = {}
    # options_en = ['Numeric', 'Categorical']
    # options_kr = ["수치형", "범주형"]
    # options_dic = {'수치형': 'Numeric', '범주형': 'Categorical'}
    
    # # 반반 나눠서 나열
    # col1, col2 = st.columns(2)
    # keys = list(column_types.keys())
    # half = len(keys) // 2 

    # dict1 = {key: column_types[key] for key in keys[:half]}
    # dict2 = {key: column_types[key] for key in keys[half:]}

    # with col1:
    #     for column, col_type in dict1.items():
    #         default_index = options_en.index(col_type)
    #         user_col_type = st.radio(
    #             f"'{column}'의 유형:",
    #             options_kr,
    #             index=default_index,
    #             key=column
    #         )
    #         user_column_types[column] = options_dic[user_col_type]

    # with col2:
    #     for column, col_type in dict2.items():
    #         default_index = options_en.index(col_type)
    #         user_col_type = st.radio(
    #             f"'{column}'의 유형:",
    #             options_kr,
    #             index=default_index,
    #             key=column
    #         )
    #         user_column_types[column] = options_dic[user_col_type]

    # return user_column_types


@st.cache_data
# 수치형 데이터 변환
def transform_numeric_data(df, column, transformation):
    if transformation == '로그변환':
        df[column + '_log'] = np.log(df[column])
        transformed_column = column + '_log'
    elif transformation == '제곱근':
        df[column + '_sqrt'] = np.sqrt(df[column])
        transformed_column = column + '_sqrt'
    elif transformation == '제곱':
        df[column + '_squared'] = np.square(df[column])
        transformed_column = column + '_squared'
    else:
        transformed_column = column  # 변환 없을 경우 원본 열 이름을 그대로 사용

    # 원본 데이터 열 삭제
    df = df.drop(column, axis = 1)

    return df

pal = sns.color_palette(['#FB8500', '#FFB703', '#8E8444', '#1B536F', '#219EBC', '#A7D0E2'])

def palet(num_categories):
    if num_categories <= 6:
        p = sns.color_palette(['#FB8500', '#FFB703', '#8E8444', '#1B536F', '#219EBC', '#A7D0E2'])
        
    else:
        p = sns.color_palette("Set2", n_colors=num_categories)
    return p

import time
@st.cache_data
def 모든_그래프_그리기(df):
    user_column_types = infer_column_types(df)
    n = len(df.columns)
    # 범주의 수에 따라 팔레트 선택
    # 전체 그래프 개수 계산
    if n > 1:
        st.warning("각 변수마다 일변량, 이변량 데이터를 시각화하고 있어요. 오래 걸릴 수 있으니 기다려주세요!")
        progress_text = "📈 그래프를 그리는 중입니다...."
        count = 0
        # bar = st.progress(count , text=progress_text)
        fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
        for i, col1 in enumerate(df.columns):
            # toast = st.toast(f"{col1}의 그래프를 그리는 중!", icon = '🍞')
            for j, col2 in enumerate(df.columns):
                # toast.toast(f"{col1}과 {col2}의 그래프", icon = '🥞')
                ax = axes[i, j]
                if i != j:
                    if user_column_types[col1] == 'Numeric' and user_column_types[col2] == 'Numeric':
                        sns.scatterplot(data=df, x=col1, y=col2, ax=ax, color = pal[0])
                    elif user_column_types[col1] == 'Categorical' and user_column_types[col2] == 'Numeric':
                        sns.boxplot(data=df, x=col1, y=col2, ax=ax, palette=pal)
                    elif user_column_types[col1] == 'Numeric' and user_column_types[col2] == 'Categorical':
                        # sns.histplot(data=df, x=col1, hue=col2, ax=ax, palette=pal)  # 여기를 수정
                        sns.kdeplot(data=df, x=col1, hue=col2, ax=ax, palette=pal)  # 여기를 수정
                    elif user_column_types[col1] == 'Categorical' and user_column_types[col2] == 'Categorical':
                        unique_values = df[col2].unique().astype(str)
                        # st.write(unique_values)
                        # 색상 매핑 생성
                        color_mapping = {val: color for val, color in zip(unique_values, palet(len(unique_values)))}
                        mosaic(df, [col1, col2], ax=ax, properties=lambda key: {'color': color_mapping[key[1]]}, gap=0.05)

                    ax.set_title(f'{col1} vs {col2}')
                else:
                    if user_column_types[col1] == 'Numeric':
                        sns.histplot(df[col1], ax=ax, color=pal[0])
                    else:
                        sns.countplot(x=df[col1], ax=ax, palette=pal)
                    ax.set_title(f'Distribution of {col1}')
                count = count + 1
                # bar.progress(count /(n*n), text=progress_text)
                # st.text(f'그려진 그래프: {completed_plots} / 총 그래프: {total_plots}')  # 진행 상황 업데이트
                time.sleep(0.1)
                # placeholder.empty()
        # st.toast("거의 다 그렸어요!", icon = "🍽")

        plt.tight_layout()
        # bar.empty()
        st.pyplot(fig)
    if n==1:
        st.warning("열을 하나만 선택하셨군요! 아래의 데이터 하나씩 시각화 영역에서 시각화하세요!")



from stemgraphic import stem_graphic

@st.cache_data
def 하나씩_그래프_그리기(df, width, height):
    user_column_types = infer_column_types(df)
    # 범주의 수에 따라 팔레트 선택
    # 전체 그래프 개수 계산s
    progress_text = "📈 그래프를 그리는 중입니다...."
    col = df.columns[0]
    # 범주형일 때, 막대, 원, 띠
    if user_column_types[col] == "Categorical":
        fig, axes = plt.subplots(1, 3, figsize=(width, height))

        # 막대 그래프
        sns.countplot(x=df[col], ax=axes[0], palette=pal)
        axes[0].set_title(f'{col} bar chart')

        # 원 그래프
        axes[1].pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct='%1.1f%%', startangle=90,  colors=pal)
        axes[1].set_title(f'{col} pie chart')

        # 띠 그래프
        # 데이터프레임에서 특정 열에 대한 값의 비율을 계산합니다.
        ddi = df.copy()
        ddi = ddi.dropna()
        ddi = pd.DataFrame(ddi[col])
        ddi['temp'] = '_'
        
        ddi_2 = pd.pivot_table(ddi, columns = col, aggfunc= 'count')
        # ddi_2.plot.bar(stacked = True, ax = axes[2])

        # 각 값이 전체 합계에 대한 비율이 되도록 변환합니다.
        ddi_percent = ddi_2.divide(ddi_2.sum(axis=1), axis=0)

        # 막대 그래프를 가로로 그리고, 누적해서 표시합니다.
        ddi_percent.plot(kind='barh', stacked=True, ax=axes[2], legend=False, color = pal)

        # 범례 설정
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend(handles, [label.split(', ')[-1][:-1] for label in labels], loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(labels), frameon=False)

        # x축 레이블을 퍼센트로 표시
        axes[2].set_xlabel('(%)')

        # y축의 눈금과 레이블 제거
        axes[2].yaxis.set_ticks([])
        axes[2].yaxis.set_ticklabels([])

        # 그래프 제목 설정
        axes[2].set_title(f'{col} ribbon graph')

        plt.tight_layout()
        st.pyplot(fig)

    # 수치형일 때, 줄기잎, 히스토, 도다, 상자그림
    else:

        fig, axes = plt.subplots(2, 2, figsize=(width, height*2))            
        
        # 줄기잎그림

        # 히스토그램
        # 도다
        # 상자그림
        stem_graphic(df[col], ax = axes[0,0])
        sns.histplot(data = df, x = col, ax = axes[0,1], color=pal[0])
        # sns.boxplot(data = df, x = col, ax = axes[1,0], palette=pal)
        sns.boxplot(data = df, x = col, ax = axes[1,1], palette = pal)


        # 데이터를 히스토그램으로 나누어 계급 구하기
        df_copy = df.dropna()
        counts, bin_edges = np.histogram(df_copy[col], bins=10)

        # 도수분포다각형을 그리기 위한 x값(계급의 중앙값) 계산
        # 양 끝의 계급에 대한 도수를 0으로 추가
        counts = np.insert(counts, 0, 0)
        counts = np.append(counts, 0)
        bin_edges = np.insert(bin_edges, 0, bin_edges[0] - (bin_edges[1] - bin_edges[0]))
        bin_edges = np.append(bin_edges, bin_edges[-1] + (bin_edges[-1] - bin_edges[-2]))

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # 도수분포다각형 그리기
        axes[1,0].plot(bin_centers, counts, marker='o', linestyle='-')

        plt.tight_layout()
        st.pyplot(fig)

    # fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
    # for i, col1 in enumerate(df.columns):
    #     # toast = st.toast(f"{col1}의 그래프를 그리는 중!", icon = '🍞')
    #     for j, col2 in enumerate(df.columns):
    #         # toast.toast(f"{col1}과 {col2}의 그래프", icon = '🥞')
    #         ax = axes[i, j]
    #         if i != j:
    #             if user_column_types[col1] == 'Numeric' and user_column_types[col2] == 'Numeric':
    #                 sns.scatterplot(data=df, x=col1, y=col2, ax=ax, color = pal[0])
    #             elif user_column_types[col1] == 'Categorical' and user_column_types[col2] == 'Numeric':
    #                 sns.boxplot(data=df, x=col1, y=col2, ax=ax, palette=pal)
    #             elif user_column_types[col1] == 'Numeric' and user_column_types[col2] == 'Categorical':
    #                 # sns.histplot(data=df, x=col1, hue=col2, ax=ax, palette=pal)  # 여기를 수정
    #                 sns.kdeplot(data=df, x=col1, hue=col2, ax=ax, palette=pal)  # 여기를 수정
    #             elif user_column_types[col1] == 'Categorical' and user_column_types[col2] == 'Categorical':
    #                 unique_values = df[col2].unique().astype(str)
    #                 # st.write(unique_values)
    #                 # 색상 매핑 생성
    #                 color_mapping = {val: color for val, color in zip(unique_values, palet(len(unique_values)))}
    #                 mosaic(df, [col1, col2], ax=ax, properties=lambda key: {'color': color_mapping[key[1]]}, gap=0.05)

    #             ax.set_title(f'{col1} vs {col2}')
    #         else:
    #             if user_column_types[col1] == 'Numeric':
    #                 sns.histplot(df[col1], ax=ax, color=pal[0])
    #             else:
    #                 sns.countplot(x=df[col1], ax=ax, palette=pal)
    #             ax.set_title(f'Distribution of {col1}')
    #         count = count + 1
    #         # bar.progress(count /(n*n), text=progress_text)
    #         # st.text(f'그려진 그래프: {completed_plots} / 총 그래프: {total_plots}')  # 진행 상황 업데이트
    #         time.sleep(0.1)
    #         # placeholder.empty()
    # # st.toast("거의 다 그렸어요!", icon = "🍽")

    # plt.tight_layout()
    # # bar.empty()
    # st.pyplot(fig)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# import koreanize_matplotlib
@st.cache_data
def plot_residuals(df, x, y):

    # DataFrame의 복사본 생성
    df_copy = df.copy()
    df_copy = df_copy.dropna()

    # 선형 회귀 모델 생성 및 학습
    model = LinearRegression()
    model.fit(df_copy[[x]], df_copy[y])

    # 예측 및 잔차 계산
    df_copy['Predicted'] = model.predict(df_copy[[x]])
    df_copy['Residuals'] = df_copy[y] - df_copy['Predicted']

    fig, axs = plt.subplots(ncols=2, figsize=(8, 3))
    sns.regplot(x=x, y=y, data=df_copy, ax = axs[0], color = '#557C55', line_kws = {'color' : '#FA7070'})
    sns.regplot(data = df_copy, x='Predicted', y='Residuals', lowess=True,  color = "#A6CF98",line_kws={'color': '#FA7070'}, ax = axs[1])
    axs[1].hlines(y=0, xmin=df_copy['Predicted'].min(), xmax=df_copy['Predicted'].max(), color='#FA7070', linestyles = 'dashed')
    axs[0].set_title(f'Regression')
    axs[1].set_title('Residual plot')    
    st.pyplot(fig)