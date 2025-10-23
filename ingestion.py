from dotenv import load_dotenv  # 환경변수 로드를 위한 dotenv 라이브러리 임포트

load_dotenv()  # .env 파일에서 환경변수 로드 (OPENAI_API_KEY, PINECONE_API_KEY 등)

from langchain.text_splitter import RecursiveCharacterTextSplitter  # 문서를 작은 청크로 재귀적으로 분할하는 도구
from langchain_community.document_loaders import ReadTheDocsLoader  # ReadTheDocs 형식의 HTML 문서를 로드하는 도구
from langchain_openai import OpenAIEmbeddings  # OpenAI의 임베딩 모델 (텍스트 → 벡터 변환)
from langchain_pinecone import PineconeVectorStore  # Pinecone 벡터 DB와 연결하는 도구

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # OpenAI의 text-embedding-3-small 임베딩 모델 초기화


def ingest_docs():  # 문서를 크롤링 → 분할 → 임베딩 → Pinecone 저장하는 전체 파이프라인
    loader = ReadTheDocsLoader("langchain-docs/langchain.readthedocs.io/en/v0.1")  # ReadTheDocs 문서 로더 생성 (로컬에 다운로드된 HTML 문서 경로)

    raw_documents = loader.load()  # HTML 문서들을 메모리로 읽어옴 (파싱하여 Document 객체 리스트로 변환)
    print(f"loaded {len(raw_documents)} documents")  # 로드된 문서 개수 출력

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)  # 텍스트 분할기 초기화 (600자 청크, 50자 오버랩으로 문맥 유지)
    documents = text_splitter.split_documents(raw_documents)  # 큰 문서를 작은 청크로 분할 (임베딩 성능 향상)
    for doc in documents:  # 각 문서의 메타데이터(source URL) 수정
        new_url = doc.metadata["source"]  # 기존 로컬 파일 경로 가져오기
        new_url = new_url.replace("langchain-docs", "https:/")  # 로컬 경로를 웹 URL로 변경 (사용자에게 표시할 링크)
        doc.metadata.update({"source": new_url})  # 수정된 URL로 메타데이터 업데이트

    print(f"Going to add {len(documents)} to Pinecone")  # Pinecone에 추가할 문서 개수 출력
    PineconeVectorStore.from_documents(  # 문서들을 임베딩하고 Pinecone 벡터 DB에 저장
        documents, embeddings, index_name="langchain-doc-index"  # documents: 저장할 문서들, embeddings: 임베딩 모델, index_name: Pinecone 인덱스 이름
    )
    print("****Loading to vectorstore done ***")  # 벡터 DB 저장 완료 메시지


if __name__ == "__main__":  # 이 파일을 직접 실행할 때만 작동 (import될 때는 실행 안됨)
    ingest_docs()  # 문서 수집 파이프라인 실행
