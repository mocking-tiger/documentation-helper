from backend.core import run_llm  # RAG 기능을 수행하는 run_llm 함수 임포트
import streamlit as st  # Streamlit 웹 앱 프레임워크 임포트
from typing import Set  # 타입 힌팅용 Set 타입 임포트

st.header("LangChain Documentation Helper Bot")  # 앱 헤더 표시

prompt = st.text_input("Prompt", placeholder="Enter your question here...")  # 사용자 입력을 받는 텍스트 입력 필드 생성

# 세션 상태 초기화 (user_prompt_history, chat_answers_history, chat_history가 없으면 빈 리스트 생성)
for k in ("user_prompt_history", "chat_answers_history", "chat_history"):
    if k not in st.session_state:
        st.session_state[k] = []

def create_sources_string(source_urls: Set[str])->str:  # 참고 문서 URL들을 보기 좋게 포맷팅하는 함수
  if not source_urls:  # URL이 없으면 빈 문자열 반환
    return ""
  sources_list = list(source_urls)  # Set을 List로 변환 (정렬 가능하게)
  sources_list.sort()  # URL을 알파벳 순으로 정렬
  sources_string = "\n"  # 결과 문자열 초기화
  for i, source in enumerate(sources_list):  # 각 URL에 번호를 매겨서 추가
    sources_string += f"{i+1}. {source}\n"
  return sources_string  # 포맷팅된 문자열 반환

if prompt:  # 사용자가 질문을 입력했을 때만 실행 (prompt가 비어있지 않으면 True)
    with st.spinner("Generating response..."):  # 로딩 스피너 표시 ("Generating response..." 메시지)
        generated_response = run_llm(query=prompt, chat_history=st.session_state.chat_history)  # RAG 체인 실행: 질문 → 검색 → 증강 → 생성
        sources = set([doc.metadata["source"] for doc in generated_response["context"]])  # 검색된 문서들의 출처 URL을 Set으로 수집 (중복 제거)
        formatted_response = f"**{generated_response['answer']}**\n\n**참고 문서:**\n{create_sources_string(sources)}"  # 답변과 참고 문서를 포맷팅 (마크다운 형식)
        st.session_state.user_prompt_history.append(prompt)  # 현재 질문을 세션 기록에 추가 (채팅 히스토리 유지)
        st.session_state.chat_answers_history.append(formatted_response)  # 생성된 답변을 세션 기록에 추가 (채팅 히스토리 유지)
        st.session_state.chat_history.append(("human", prompt))  # 대화 히스토리에 사용자 질문 추가 (LangChain Memory용)
        st.session_state.chat_history.append(("ai", generated_response["answer"]))  # 대화 히스토리에 AI 답변 추가 (LangChain Memory용)

if st.session_state["chat_answers_history"]:  # 채팅 기록이 있으면 전체 대화 내역 표시
  for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):  # 답변과 질문을 순서대로 묶어서 반복 (zip으로 쌍을 만듦)
    st.chat_message("user").write(user_query)  # 사용자 메시지를 채팅 UI로 표시
    st.chat_message("assistant").write(generated_response)  # AI 어시스턴트 메시지를 채팅 UI로 표시
