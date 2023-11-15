import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

# Streamlit 앱 제목 설정
st.title('원하는 그래프를 만들어드려요!')

# openai.api_key = st.secrets['openai']['key']
openai.api_key = st.text_input("openai API의 key를 입력해주세요.")
if openai.api_key is not None:
    st.success("API key 불러오기 성공!")
# Penguins 데이터셋 불러오기
data = sns.load_dataset('penguins')

# 데이터셋 표시
st.write("Penguins 데이터셋 미리보기:")
st.write(data.head(3))

######################
from openai import OpenAI

client = OpenAI(api_key = openai.api_key)

prompt_2 = st.text_input("원하는 그래프를 구체적으로 얘기해주세요.")
#서식지에 따른 펭귄의 특성을 시각화하는 그래프를 그리는 코드를 작성해줘
if st.button("그래프를 그려줘!"):
    response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="예를 들어 sns.barplot(data=data, x='species', y='bill_length_mm')와 같이 seaborn의 penguins 데이터(데이터 이름 : data)에 대해 내가 다음에 할 얘기를 실행할 수 있는 seaborn, plt 라이브러리를 사용한 코드를 작성하고, 주석은 모두 앞에 '#'을 달아야해. 너의 코드를 바로 실행할거여서말이야. . "+prompt_2, 
    max_tokens = 500
    )

    st.code(response.choices[0].text)
    # 사용자로부터 Seaborn 코드 수정
    st.write("Seaborn 그래프 코드를 입력해주세요. 예: sns.barplot(data=data, x='species', y='bill_length_mm')")
    user_input = st.text_area("코드 입력:", value=response.choices[0].text)
    # "sns.barplot(data=data, x='species', y='bill_length_mm')"
    # 실행 버튼
    # 입력된 코드 실행
    try:
        fig, ax = plt.subplots()
        exec(user_input)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"오류 발생: {e}")

    ######################
