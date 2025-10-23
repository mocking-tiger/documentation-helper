# RAG 기능을 수행하는 run_llm 함수 임포트
from backend.core import run_llm
# Streamlit 웹 앱 프레임워크 임포트
import streamlit as st
# 타입 힌팅용 Set 타입 임포트
from typing import Set

# 앱 헤더 표시
st.header("LangChain Documentation Helper Bot")

# 사용자 입력을 받는 텍스트 입력 필드 생성
prompt = st.text_input("Prompt", placeholder="Enter your question here...")

# 세션 상태에 사용자 질문 기록이 없으면 빈 리스트 초기화 (페이지 새로고침 시에도 유지)
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
# 세션 상태에 챗봇 답변 기록이 없으면 빈 리스트 초기화 (페이지 새로고침 시에도 유지)
if "chate_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

# 참고 문서 URL들을 보기 좋게 포맷팅하는 함수
def create_sources_string(source_urls: Set[str])->str:
  # URL이 없으면 빈 문자열 반환
  if not source_urls:
    return ""
  # Set을 List로 변환 (정렬 가능하게)
  sources_list = list(source_urls)
  # URL을 알파벳 순으로 정렬
  sources_list.sort()
  # 결과 문자열 초기화
  sources_string = "\n"
  # 각 URL에 번호를 매겨서 추가
  for i, source in enumerate(sources_list):
    sources_string += f"{i+1}. {source}\n"
  # 포맷팅된 문자열 반환
  return sources_string

# 사용자가 질문을 입력했을 때만 실행 (prompt가 비어있지 않으면 True)
if prompt:
    # 개발 중 디버깅 정보 표시 (현재 주석 처리됨)
    # with st.expander("🐛 디버그 정보"):
    #     st.write("입력:", prompt)
    #     st.write("길이:", len(prompt))

    # 로딩 스피너 표시 ("Generating response..." 메시지)
    with st.spinner("Generating response..."):
        # RAG 체인 실행: 질문 → 검색 → 증강 → 생성
        generated_response = run_llm(query=prompt)
        # 검색된 문서들의 출처 URL을 Set으로 수집 (중복 제거)
        sources = set([doc.metadata["source"] for doc in generated_response["context"]])

        # 답변과 참고 문서를 포맷팅 (마크다운 형식)
        formatted_response = f"**{generated_response['answer']}**\n\n**참고 문서:**\n{create_sources_string(sources)}"

        # 현재 질문을 세션 기록에 추가 (채팅 히스토리 유지)
        st.session_state.user_prompt_history.append(prompt)
        # 생성된 답변을 세션 기록에 추가 (채팅 히스토리 유지)
        st.session_state.chat_answers_history.append(formatted_response)

    # 결과 전체 보기 (현재 주석 처리됨)
    # with st.expander("📊 전체 응답"):
    #     st.json(generated_response)

    # 최신 답변만 표시
    st.write("### 답변:")
    st.write(formatted_response)

# 채팅 기록이 있으면 전체 대화 내역 표시
if st.session_state["chat_answers_history"]:
  # 답변과 질문을 순서대로 묶어서 반복 (zip으로 쌍을 만듦)
  for generated_response, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
    # 사용자 메시지를 채팅 UI로 표시
    st.chat_message("user").write(user_query)
    # AI 어시스턴트 메시지를 채팅 UI로 표시
    st.chat_message("assistant").write(generated_response)