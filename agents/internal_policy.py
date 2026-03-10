import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLM
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,
)

# 向量库
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(
    persist_directory=r"D:\AIAPPdevelopment\lianxiproject20260121\langchainday\qiyezhishiku\code\chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt（已修复双大括号）
internal_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个企业内部制度咨询助手。
根据检索到的员工手册、奖惩制度、请假考勤制度等内容，给出准确、友好的回复。

输出格式：严格JSON，不要多余文字：
{{
  "answer": "完整、自然的回复文本",
  "sources": ["来源文件1", "来源文件2"]
}}

检索内容：
{context}

用户问题：{question}"""),
    ("human", "{question}")
])

chain = internal_prompt | llm | StrOutputParser()


def handle_internal(question: str) -> dict:
    try:
        docs = retriever.invoke(question)
        context = "\n\n".join(
            [f"来源：{doc.metadata.get('source', '未知')}\n内容：{doc.page_content[:800]}..." for doc in docs])

        raw = chain.invoke({"context": context, "question": question})

        # 清理输出
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        result = json.loads(cleaned)
        return result

    except Exception as e:
        return {"error": str(e), "raw": raw if 'raw' in locals() else ""}


# 测试
if __name__ == "__main__":
    tests = [
        "怎么请假？",
        "迟到一次扣多少钱？",
        "年假有几天？",
        "报销流程是什么？",
        "请假需要提前几天申请？"
    ]
    for q in tests:
        print(f"\n问题: {q}")
        result = handle_internal(q)
        print(json.dumps(result, ensure_ascii=False, indent=2))