import streamlit as st
import google.generativeai as genai 
from geopy.geocoders import Nominatim
import pandas as pd
import re

import folium
from streamlit_folium import st_folium



from langchain.chains import LLMChain, SequentialChain, RetrievalQA
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import Field
from langchain_core.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, ValidationError
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA, LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

@st.cache_resource

# 모델 로딩
def load_model():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                            temperature=0.2,
                            verbose=True,
                            google_api_key='AIzaSyAsaP7rNv74yVna-W1x-kMevK6lzkefOMk')
    return model

model = load_model()

# 벡터 스토어 로딩
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

modelPath = "jhgan/ko-sroberta-multitask"
model_kwargs = {'device':'cuda'}
# model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': True} # 벡터간의 cosine similarity 계산에서 더 유리하다.

embeddings_model = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
vectorstore_8k = FAISS.load_local("res_vectorstore", embeddings_model, allow_dangerous_deserialization=True)
vectorstore_geo = FAISS.load_local("addr_vectorstore", embeddings_model, allow_dangerous_deserialization=True)
vectorstore_tour = FAISS.load_local("tour_vectorstore", embeddings_model, allow_dangerous_deserialization=True)

# # 리트리버 로딩
# llm
res_retriever = vectorstore_8k.as_retriever(search_kwargs={"k": 3})
# 벡터스토어 서칭 - 좌표 기준
geo_retriever = vectorstore_geo.as_retriever(search_kwargs={"k": 100})
# 벡터스토어 서칭 - geo의 결과로 나온 관광지 기준
tour_retriever = vectorstore_tour.as_retriever(search_kwargs={"k": 1})

#지도 표시
# Geolocator 초기화
geolocator = Nominatim(user_agent = 'South Korea', timeout=None)

# 주소를 위도와 경도로 변환하는 함수
def get_lat_lon(address):
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None
    

def make_sub(docs, name):
    lines = docs[0].page_content.split('\n')

    # 딕셔너리를 생성
    data_dict = {}

    # 각 라인에 대해 처리
    for line in lines:
        key_value = line.split(": ", 1) 
        if len(key_value) == 2:
            key, value = key_value
            try:
                data_dict[key] = float(value)
            except ValueError:
                data_dict[key] = value

    # 결과 출력
    place = data_dict.get(name)
    place_latitude = data_dict.get('Latitude')
    place_longitude = data_dict.get('Longitude')
    
    return place, place_latitude, place_longitude

def generate_response(user_questions):
    

    # Step 1: 음식점 추천 체인
    prompt_chain1 = ChatPromptTemplate.from_messages([
        ("assistant", "당신은 제주도 음식점 추천 도우미입니다. 사용자 선호에 맞는 식당을 추천하기 위해 제공된 정보를 고려하여 적절한 식당명과 좌표를 무조건 출력형식에 맞게 반환하세요."),
        ("assistant", "음식점에 데이터에는 여러 약어로 된 칼럼명이 포함되어 있습니다. 각 칼럼명은 다음과 같은 의미를 가집니다:\n"
                    "월(month)_(요일(day of the week))or(그 외 내용)_추가내용(use, price, visit과 같은)순서로 칼럼이 구성되어 있습니다.\n"
                    "1. 시간대: n시m시 → `ntom` (예: `12to13`은 12시부터 13시까지를 의미합니다.)\n"
                    "2. 연령대: `n대`는 특정 연령대를 의미합니다 (예: `30대`는 30세 이상 39세 이하를 뜻합니다.) 추가적으로 '20대'는 20대 이하를 나타냅니다.\n"
                    "3. 요일(day of the week): 요일 약어를 사용하며, 월요일은 `월`, 금요일은 `금` 등으로 축약되었습니다.\n"
                    "4. 월(month): 월은 숫자로 표시되며 칼럼명 맨 앞에 `n_`으로 표현합니다 (예: `1_`은 1월을 의미합니다.)\n"
                    "5. 성별: `M`은 남성을, `FM`은 여성을 의미합니다.\n"
                    "6. 이용건수: `u` 'use'의 약자로 특정 시간대나 요일, 성별에 따른 이용 건수를 나타냅니다.\n"
                    "7. 이용금액: `p` 'price'의 약자로 해당 시간대 또는 요일의 이용 금액을 나타냅니다.\n"
                    "8. 평균 이용금액: `avgP`는 특정 시간대의 평균 이용 금액을 나타냅니다.\n"
                    "9. 현지인 이용 비중: `local`은 현지인의 이용 비중을 나타냅니다.\n"
                    "10. 조회수: `v`는 'visit'의 약자로 조회수를 의미하며, `t_v` 'total_visit'의 약자로 요일별 통합 조회수를 나타냅니다.\n"
                    "11. 성수기 여부: `P`는 'Peaked'의 약자로 성수기 여부를 나타냅니다.\n"
                    "12. 개설일: `fo`는 'foundation'의 약자로 가게의 개설일을 의미합니다.\n\n"
                    "13. logitude : 경도입니다"
                    "14. latitude : 위도입니다"

                    "칼럼명 예시:\n"
                    "- `1_12to13_u`: 1월의 12시부터 13시까지의 이용 건수\n"
                    "- `2_14to17_p`: 2월의 14시부터 17시까지의 이용 금액\n"
                    "- `3_20대`: 3월의 20대 이용 비중\n"
                    "- `4_FM`: 4월의 여성 이용 건수\n"
                    "- `5_local`: 5월의 현지인 이용 비율\n\n"

                    "위 정보를 사용하여 사용자 요청에 적합한 식당명과 해당 식당의 위도와 경도 정보를 반환합니다."),
        ('ai', """
        출력 형식:
        - [restaurant_name, latitude, longitude]
        """),

        ("assistant", "'40대가 즐겨찾을 만한 음식점을 추천해줘'와 같이 정보가 적을 경우:\n"
                    "- 주어진 데이터를 기반으로 40대와 관련된 칼럼을 식별합니다.\n"
                    "- 관련된 칼럼에서 비율이 높은 데이터를 기준으로 상위 5개의 후보를 선정합니다.\n"
                    "- 선정된 후보 중 무작위로 하나를 반환합니다."),
        ("assistant", "{context}"),  # 검색된 문서를 넣기 위해 {context} 추가
        ("human", "{question}")
    ])

    chain1 = RetrievalQA.from_chain_type(
        llm=model,
        retriever=res_retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt_chain1,
            "document_variable_name": "context"
        },
        return_source_documents=True,  # 검색된 문서들을 반환하도록 설정
        output_key='restaurant_name'  # 출력: restaurant_name
    )

    inv_1 = chain1.invoke({'query' : user_questions})
    restaurant_location = inv_1.get('restaurant_name')
    res_info_docs  = inv_1.get('source_documents')
    res_info = "\n".join([doc.page_content for doc in res_info_docs])


    res_place, res_latitude, res_longitude = make_sub(res_info_docs, name= "가게명")


    #geo_retriever.get_relevant_documents("해안도로 갯바위 횟집")[0]


    # Step 3: 관광지 상세 정보 추가 체인 (tour_retriever 활용)
    prompt_chain3 = ChatPromptTemplate.from_messages([
        ('system' "해당 프롬프트의 입력값이 비어있을 경우 '검색결과가 없습니다' 라고 반환하십시오."),
        ("assistant", "당신은 선택한 관광지의 상세 정보를 제공하는 도우미입니다."),
        ('assistant', "관광지 정보 관련 칼럼에 대한 설명은 다음과 같습니다:\n"
                    "1. `관광분류`: 관광지의 카테고리를 나타냅니다 (예: 자연, 역사적 명소 등).\n"
                    "2. `평점`: 여행객들이 관광지에 매긴 평균 점수입니다.\n"
                    "3. `score_value`: `평점`의 반올림 값입니다.\n"
                    "4. `word_cnt`: 특정 키워드가 해당 관광지 리뷰에서 얼마나 많이 언급되었는지 나타내며, 값이 높을수록 관광지 평가에 중요하게 작용한 키워드임을 의미합니다.\n"
                    "5. `latitude` : 위도입니다 \n"
                    "6. `longitude` : 경도입니다."

                    "칼럼명 예시:\n"
                    "- `word_cnt`: '자연 경관': 58, '편안한 분위기': 21\n"
                    "word_cnt는 해당 장소의 특징이라고 할 수 있습니다.\n"
                    "선택된 장소가 자연 명소일 경우 자연 풍경을, 역사적 명소일 경우 역사적 배경을 강조하십시오.\n"
                    "주어진 위도와 경도와 가장 가까운 관광지를 검색하여 관련 정보를 반환합니다. 검색 결과는 내부적으로 처리됩니다."),
        ("human", "검색 조건 {question}"),
        ("assistant", "{context}")
        ])

    chain3 = RetrievalQA.from_chain_type(
        llm=model,
        retriever=tour_retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt_chain3,
            "document_variable_name": "context"
        },
        return_source_documents=True,  # 검색된 문서들을 반환하도록 설정
        output_key='tour_details'  # 출력: tour_details
    )
    inv_3 = chain3.invoke({'query' :  f"latitude는 {res_latitude}, longitude는 {res_longitude}"})

    tour_info_docs = inv_3.get('source_documents')  # 관광지 관련 문서
    tour_info = "\n".join([doc.page_content for doc in tour_info_docs])  # 관광지 정보 요약

    tour_place, tour_latitude, tour_longitude = make_sub(tour_info_docs, name= "지명")

    # Step 4: 최종 응답 생성 체인
    prompt_chain4 = ChatPromptTemplate.from_messages([
        ("assistant", "당신은 사용자에게 음식점과 관광지에 대한 테마 여행 계획을 제공하는 도우미입니다."),
        ("assistant", """
        사용자 질문에 따라 종합적인 여행 추천을 제공합니다. 출력에는 다음 정보가 포함됩니다.
        - 추천 음식점 정보
        - 해당 음식점 근처 관광지 정보
        - 사용자 질문을 바탕으로 구성된 종합적인 음식점과 관광지 추천 메시지

        LLM 역할을 위한 중요한 포인트:
        - 제공된 정보만을 사용해 답변을 생성하세요.
        - 질문이 식당 정보에 대한 것이라면, 데이터를 기반으로 해당 질문에 대한 답변만을 진행하세요.
        - 제공된 정보에 기반하지 않은 답변은 생성하지 마세요.
        - 제공된 정보를 바탕으로 식당에 대한 설명을 제공하세요.
        - 답변에는 식당, 메뉴, 주로 방문하는 고객, 식당 근처의 유명한 명소나 경치에 대한 세부 사항이 포함되어야 합니다.
        - 답변에는 식당과 명소(경치)의 이미지도 포함하십시오.
        - 사용자 선호(동반인, 나이, 예산 등)에 가장 적합한 곳을 추천하세요.
        - 항상 한국어로 답변하세요.
        """),


        ("assistant", """
        음식점 정보와 관련된 데이터 구조는 다음과 같습니다:
        - restaurant_docs는 restaurant_name에 대한 정보로 여러 약어로 된 칼럼명이 포함되어 있습니다. 각 칼럼명은 다음과 같은 의미를 가집니다:
        "1. 월(month)_(요일(day of the week))or(그 외 내용)_추가내용(use, price, visit과 같은)순서로 칼럼이 구성되어 있습니다.\n"
        "2. 시간대: n시m시 → `ntom` (예: `12to13`은 12시부터 13시까지를 의미합니다.)\n"
        "3. 연령대: `n대`는 특정 연령대를 의미합니다 (예: `30대`는 30세 이상 39세 이하를 뜻합니다.) 추가적으로 '20대'는 20대 이하를 나타냅니다.\n"
        "4. 요일(day of the week): 요일 약어를 사용하며, 월요일은 `월`, 금요일은 `금` 등으로 축약되었습니다.\n"
        "5. 월(month): 월은 숫자로 표시되며 칼럼명 맨 앞에 `n_`으로 표현합니다 (예: `1_`은 1월을 의미합니다.)\n"
        "6. 성별: `M`은 남성을, `FM`은 여성을 의미합니다.\n"
        "7. 이용건수: `u` 'use'의 약자로 특정 시간대나 요일, 성별에 따른 이용 건수를 나타냅니다.\n"
        "8. 이용금액: `p` 'price'의 약자로 해당 시간대 또는 요일의 이용 금액을 나타냅니다.\n"
        "9. 평균 이용금액: `avgP`는 특정 시간대의 평균 이용 금액을 나타냅니다.\n"
        "10. 현지인 이용 비중: `local`은 현지인의 이용 비중을 나타냅니다.\n"
        "11. 조회수: `v`는 'visit'의 약자로 조회수를 의미하며, `t_v` 'total_visit'의 약자로 요일별 통합 조회수를 나타냅니다.\n"
        "12. 성수기 여부: `P`는 'Peaked'의 약자로 성수기 여부를 나타냅니다.\n"
        "13. 개설일: `fo`는 'foundation'의 약자로 가게의 개설일을 의미합니다.\n"
        "14. longitude : 경도입니다.\n"
        "15. latitude : 위도입니다.\n"

        - 칼럼명 예시:
        - `1_12to13_u`: 1월의 12시부터 13시까지의 이용 건수
        - `2_14to17_p`: 2월의 14시부터 17시까지의 이용 금액
        - `3_20대`: 3월의 20대 이용 비중
        - `4_FM`: 4월의 여성 이용 건수
        - `5_local`: 5월의 현지인 이용 비율
        """),


        ('ai', "특정 식당에 대한 정보를 요구할 경우 요구되는 질문에 대한 관광지 추천을 진행하지 않고 정해진 대답만을 진행한 후 종료합니다.\n"
                    "해당 상황에 대한 대답 출력예시1"
                    "- 질문이 다음과 같은 형식일 경우 : 해안도로 갯바위 횟집의 12시부터 21시 사이의 이용률은 어느정도야?"
                    "- 대답 : 해안도로 갯바위 횟집의 12시부터 21시 사이의 이용률은 0.1905입니다.\n"
                    "해당 상황에 대한 대답 출력예시2"
                    "- 질문이 다음과 같은 형식일 경우 : 12시부터 21시 사이의 이용률이 가장높은 식당은 어디야?"
                    "- 대답 : 12시부터 21시 사이의 이용률이 가장 높은 식당은 0.1905인 '해안도로 갯바위 횟집' 입니다.\n"
        ),
        ('assistant', "식당에 대한 설명을 마친 후 관광지에 대한 설명을 진행해 주세요. 만약 질문이 음식점에 대한 지엽적인 정보일 경우 해당 추천은 진행하지 않습니다.\n"
                    "tour_details는 tour_place_name에 대한 칼럼으로 각 칼럼의 설명은 다음과 같습니다:\n"
                    "1. `관광분류`: 관광지의 카테고리를 나타냅니다 (예: 자연, 역사적 명소 등).\n"
                    "2. `평점`: 여행객들이 관광지에 매긴 평균 점수입니다.\n"
                    "3. `score_value`: `평점`의 반올림 값입니다.\n"
                    "4. `word_cnt`: 특정 키워드가 해당 관광지 리뷰에서 얼마나 많이 언급되었는지 나타내며, 값이 높을수록 관광지 평가에 중요하게 작용한 키워드임을 의미합니다.\n"
                    "5. longitude : 경도입니다.\n"
                    "6. latitude : 위도입니다."
                    "word_cnt 예시:\n"
                    "- `word_cnt`: '자연 경관': 58, '편안한 분위기': 21\n"

                    "word_cnt는 해당 장소의 특징이라고 할 수 있습니다.\n"
                    "선택된 장소가 자연 명소일 경우 자연 풍경을, 역사적 명소일 경우 역사적 배경을 강조하십시오.쇼핑일 경우 쇼핑할 내용거리에 대해 설명하세요.\n"
                    "해당 관광지의 주요 특징을 설명합니다. 'word_cnt'의 높은 값을 가진 키워드를 토대로 해당 관광지를 설명하십시오. 유저의 질문을 고려하여 해당 장소가 왜 어울리는지도 함께 설명하세요. \n"),
        ("human", """
        사용자 질문: {query}
        음식점 정보: {res_info_chain1}
        관광지 정보: {tour_info_chain3}
        """)
    ])

    chain4 = LLMChain(
        llm=model,
        prompt=prompt_chain4,
        output_key='final_response'  # 최종 출력
    )


    response = chain4.invoke({
        'query': user_questions,  # 사용자 질문
        'res_info_chain1': res_info,  # chain1에서 나온 음식점 정보
        'tour_info_chain3': tour_info  # chain3에서 나온 관광지 정보
    })




    return response['final_response'], res_place, res_latitude, res_longitude, tour_place, tour_latitude, tour_longitude



# 제목
st.title('🍊제주도 맛집 추천')


#st.header()
st.subheader('관광지와 테마를 중점으로 설명해 주시면 좋습니다.')

with st.sidebar:
    st.title('⚙️Settings')

    st.write("Menu")
    # selectbox 레이블 공백 제거
    st.markdown(
    """
    <style>
    .stSelectbox label {  /* This targets the label element for selectbox */
        display: none;  /* Hides the label element */
    }
    .stSelectbox div[role='combobox'] {
        margin-top: -20px; /* Adjusts the margin if needed */
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    time = st.sidebar.selectbox("", ["단품요리 전문", "가정식", "커피", "치킨", "베이커리",
                                      "분식", "양식", "일식", "중식", "맥주/요리주점", "피자",
                                        "꼬치구이", "햄버거", "구내식당/푸드코트", "샌드위치/토스트",
                                          "아이스크림/빙수", "떡/한과", "도시락", "주스", "차", "포장마차",
                                            "기타세계요리", "기사식당", "도너츠", "동남아/인도음식",
                                              "패밀리 레스토랑", "스테이크", "민속주점", "부페", "야식", ], key="time")
    
    st.write("주요 이용객")
    st.write("주요 연령")
    st.write("성수기 여부")

# 채팅 메시지 저장
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "ai", "message": "어떤 맛집을 추천해 드릴까요?", "avatar": "🗿"}]

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar=message['avatar']):
           st.markdown(message["message"])
           if "map_html" in message:
               st.components.v1.html(message["map_html"], width=800, height=600)

# 채팅 기록 초기화 함수
def clear_chat_history():
    st.session_state.chat_history = [{"role": "ai", "message": "어떤 맛집을 추천해 드릴까요?", "avatar": "🗿"}]

# 사이드바에 채팅 기록 초기화 버튼 
st.sidebar.button('채팅 기록 초기화', on_click=clear_chat_history)

# 사용자 입력을 처리하는 부분
if prompt := st.chat_input("추천 받고 싶은 제주도의 맛집에 관한 정보를 적어주세요."):
    # 사용자 메시지 추가
    user_message = {'role': 'user', 'message': prompt, 'avatar': "🍚"}
    st.session_state.chat_history.append(user_message)
    
    # 사용자 메시지 표시
    with st.chat_message("user", avatar="🍚"):
        st.markdown(prompt)

    # AI 응답 처리
    #response['final_response'], res_place, res_latitude, res_longitude, tour_place, tour_latitude, tour_longitude

    with st.chat_message("ai", avatar="🗿"):
        with st.spinner("메시지 처리 중입니다."):
            full_response, res_place, res_latitude, res_longitude, tour_place, tour_latitude, tour_longitude = generate_response(user_questions=prompt)


            # 스트림릿 맵 표시
            m = folium.Map(location=[res_latitude, res_longitude], zoom_start=10)
            folium.Marker([res_latitude, res_longitude], popup=res_place,tooltip="음식점 위치").add_to(m)
            folium.Marker([tour_latitude, tour_longitude],popup=tour_place,tooltip="관광지 위치").add_to(m)

            folium.PolyLine(
            locations=[(res_latitude, res_longitude), (tour_latitude, tour_longitude)],
            color="blue", weight=2.5, opacity=1).add_to(m)


            map_html = m._repr_html_()
            st.markdown(full_response)
            

        st.session_state.chat_history.append({
                "role": "ai",
                "message": full_response,
                "avatar": "🗿",
                "map_html": map_html
            })
        st.components.v1.html(map_html, width=800, height=600)


