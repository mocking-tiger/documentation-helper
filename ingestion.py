import asyncio  # 비동기 프로그래밍을 위한 asyncio 모듈 임포트
import os  # 운영체제 관련 기능(환경변수 등)을 위한 os 모듈 임포트
import ssl  # SSL/TLS 보안 연결을 위한 ssl 모듈 임포트
from typing import Any, Dict, List  # 타입 힌팅을 위한 typing 모듈의 타입들 임포트
import certifi  # SSL 인증서 경로를 제공하는 certifi 패키지 임포트
from dotenv import load_dotenv  # .env 파일에서 환경변수를 로드하는 함수 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트를 작은 청크로 분할하는 텍스트 스플리터 임포트
from langchain_chroma import Chroma  # Chroma 벡터 데이터베이스 클래스 임포트
from langchain_core.documents import Document  # LangChain의 문서 객체 클래스 임포트
from langchain_openai import OpenAIEmbeddings  # OpenAI 임베딩 모델 클래스 임포트
from langchain_pinecone import PineconeVectorStore  # Pinecone 벡터 저장소 클래스 임포트
from langchain_tavily import TavilyCrawl,TavilyExtract, TavilyMap  # Tavily 웹 크롤링 및 데이터 추출 도구들 임포트
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)  # 커스텀 로깅 함수들 임포트
load_dotenv()  # .env 파일의 환경변수를 시스템에 로드

ssl_context = ssl.create_default_context(cafile=certifi.where())  # certifi의 인증서를 사용하는 기본 SSL 컨텍스트 생성
os.environ["SSL_CERT_FILE"] = certifi.where()  # SSL 인증서 파일 경로를 환경변수로 설정
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()  # requests 라이브러리가 사용할 인증서 번들 경로 설정

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10)  # OpenAI의 text-embedding-3-small 모델을 사용하는 임베딩 객체 생성

# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)  # Chroma 벡터 데이터베이스 초기화 (현재 주석 처리됨)
vectorstore = PineconeVectorStore(index_name="langchain-docs-2025", embedding=embeddings)  # Pinecone 벡터 저장소 초기화 (langchain-docs-2025 인덱스 사용)
tavily_extract = TavilyExtract()  # Tavily 데이터 추출 객체 생성
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)  # Tavily 웹사이트 맵핑 객체 생성 (깊이 5, 너비 20, 최대 1000페이지)
tavily_crawl = TavilyCrawl()  # Tavily 웹 크롤러 객체 생성

async def main():  # 비동기 메인 함수 정의
    """"Main async function to orchestrate the entire process."""
    print("main함수 실행")
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/", Colors.PURPLE)

    res = tavily_crawl.invoke({
      "url": "https://python.langchain.com/",
      "max_depth": 1,
      "extract_depth": "advanced",
      # "instructions": "content on ai agents"
    })
    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in res["results"]]
    log_success(f"TavilyCrawl: Crawled {len(all_docs)} pages")

if __name__ == "__main__":  # 스크립트가 직접 실행될 때만 main() 함수 실행
    asyncio.run(main())  # 비동기 main() 함수를 실행