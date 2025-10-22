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

async def index_documents_async(documents: List[Document], batch_size: int = 50):
  """문서들을 배치 단위로 비동기 처리하여 벡터 DB에 저장하는 함수"""
  log_header("VECTOR STORAGE PHASE")  # ===== 3단계: 벡터 저장 =====
  log_info(f"VectorStore Indexing: Preparing to add {len(documents)} documents to the vector store", Colors.DARKCYAN)  # 저장할 문서 개수 로그

  # 문서들을 batch_size 크기로 나눔 (예: 1000개 문서 → 50개씩 20개 배치로)
  batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
  log_info(f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each")  # 배치 개수 로그

  # 각 배치를 벡터 DB에 추가하는 내부 함수
  async def add_batch(batch: List[Document], batch_num:int):
    try:
      await vectorstore.aadd_documents(batch)  # 비동기로 문서 배치를 Pinecone에 추가 (임베딩 + 저장)
      log_success(f"VectorStore Indexing: Batch {batch_num} completed successfully")  # 성공 로그
    except Exception as e:
      log_error(f"VectorStore Indexing: Batch {batch_num} failed: {e}")  # 실패 로그
      return False  # 실패 표시
    return True  # 성공 표시

  # 모든 배치를 동시에 처리 (병렬 처리)
  tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]  # 각 배치마다 작업 생성
  results = await asyncio.gather(*tasks, return_exceptions=True)  # 모든 작업을 동시에 실행하고 결과 수집

  # 성공한 배치 개수 세기
  successful = sum(1 for result in results if result is True)  # True인 결과만 카운트

  if successful == len(batches):  # 모든 배치가 성공했으면
    log_success(
      f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
    )
  else:  # 일부만 성공했으면
    log_warning(
      f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
    )

async def main():  # 전체 프로세스를 조율하는 메인 비동기 함수
    """"전체 문서 수집 파이프라인: 크롤링 → 분할 → 임베딩 → 저장"""
    print("main함수 실행")
    log_header("DOCUMENTATION INGESTION PIPELINE")  # 파이프라인 시작 헤더

    # ===== 1단계: 웹 크롤링 =====
    log_info("TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/", Colors.PURPLE)

    res = tavily_crawl.invoke({  # Tavily 크롤러로 LangChain 문서 사이트 크롤링
      "url": "https://python.langchain.com/",  # 크롤링 시작 URL
      "max_depth": 1,  # 링크를 1단계 깊이까지만 따라감 (메인 페이지 + 바로 링크된 페이지)
      "extract_depth": "advanced",  # 고급 수준으로 콘텐츠 추출 (더 많은 정보 수집)
      # "instructions": "content on ai agents"  # 특정 주제만 크롤링하려면 활성화 (현재 주석처리)
    })
    # 크롤링 결과를 LangChain Document 객체로 변환 (리스트 컴프리헨션 사용)
    # 각 페이지의 원본 콘텐츠(raw_content)와 URL(metadata)을 저장
    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in res["results"]]
    log_success(f"TavilyCrawl: Crawled {len(all_docs)} pages")  # 크롤링 완료 로그

    # ===== 2단계: 문서 분할 (Chunking) =====
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
      f"Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",  # 분할 설정 정보
      Colors.YELLOW
    )
    # 텍스트 분할기 생성: 각 청크를 4000자로, 앞뒤 200자씩 겹치게 분할
    # chunk_overlap이 있어야 문맥이 끊기지 않음
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    # 모든 문서를 작은 청크로 분할 (긴 문서 → 여러 개의 작은 조각)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents")  # 분할 완료 로그

    # ===== 3단계: 벡터 DB에 저장 (비동기 처리) =====
    await index_documents_async(splitted_docs, batch_size=500)  # 500개씩 배치로 나눠서 Pinecone에 저장

    log_header("PIPELINE COMPLETE")
    log_success("Documentation ingestion pipeline completed successfully!")
    log_info("Summary:", Colors.BOLD)
    # log_info(f"- URLs mapped: {len(all_docs)}")
    log_info(f"- Chunks created: {len(splitted_docs)}")
    log_info(f"- Documents extracted: {len(all_docs)}")

if __name__ == "__main__":  # 스크립트가 직접 실행될 때만 main() 함수 실행
    asyncio.run(main())  # 비동기 main() 함수를 실행