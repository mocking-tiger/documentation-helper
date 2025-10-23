from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
load_dotenv()

INDEX_NAME = "langchain-docs-2025"

def run_llm(query: str):
    # OpenAI의 임베딩 모델 초기화 (텍스트를 벡터로 변환하는 역할)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Pinecone 벡터 DB에 연결 (저장된 문서 벡터들과 유사도 검색)
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # ChatGPT 모델 초기화 (답변 생성에 사용, temperature=0으로 일관된 답변)
    chat = ChatOpenAI(verbose=True, temperature=0)

    # LangChain Hub에서 미리 만들어진 RAG용 프롬프트 템플릿 가져오기
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # 검색된 문서들을 LLM에 전달하는 체인 생성 (문서 + 프롬프트 → LLM)
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    # 전체 RAG 체인 생성 (검색 → 문서 처리 → 답변 생성)
    qa = create_retrieval_chain(retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain)

    # 질문을 입력하여 RAG 체인 실행 (검색 → 증강 → 생성)
    result = qa.invoke({"input": query})

    new_result = {
      "query": result["input"],
      "result": result["answer"],
      "source_documents": result["context"],
    }

    # 결과 반환 (input, context, answer 포함)
    return new_result

if __name__ == "__main__":
  res = run_llm(query="What is a LangChain Chain?")
  print('='*60)
  print(res["result"])
  print('='*60)