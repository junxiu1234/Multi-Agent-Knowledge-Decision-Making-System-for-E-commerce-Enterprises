import os
import json
import warnings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 忽略 DeprecationWarning（可选）
warnings.filterwarnings("ignore", category=DeprecationWarning)

# LLM 配置（通义千问）
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-max",  # 或你的模型名
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.4,
)

# 向量库路径（建议用绝对路径，避免相对路径问题）
CHROMA_PATH = r"D:\AIAPPdevelopment\lianxiproject20260121\langchainday\qiyezhishiku\code\chroma_db"

# 加载嵌入模型和向量库
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

try:
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("向量库加载成功")
except Exception as e:
    print("向量库加载失败:", e)
    raise

# 客诉处理 Prompt（已转义 JSON 示例）
complaint_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的健康电商客服Agent，专门处理客诉（过敏、投诉、退货、食品安全等）。

根据用户问题和检索到的相关文档，给出完整、合规的处理建议。

输出**严格结构化JSON格式**，**不要添加任何前缀、解释或多余文字**，只输出以下JSON对象：

{{
  "severity": "轻微" 或 "中度" 或 "严重" 或 "无法判断",
  "steps": ["步骤1完整描述", "步骤2完整描述", "..."],
  "reply_template": "给用户的完整回复话术（包括安抚、解释、具体方案、后续跟进建议）",
  "need_compliance_check": true 或 false,
  "sources": ["来源文件1", "来源文件2", "..."]
}}

检索到的参考文档：
{context}

用户问题：{question}"""),
    ("human", "{question}")
])

# 链
complaint_chain = complaint_prompt | llm | StrOutputParser()


def handle_complaint(question: str) -> dict:
    """处理客诉问题，返回解析后的JSON字典"""
    try:
        # 检索相关文档
        docs = retriever.invoke(question)
        context = "\n\n".join([
            f"来源：{doc.metadata.get('source', '未知')}\n内容：{doc.page_content[:800]}..."
            for doc in docs
        ])

        # 调用链
        raw_output = complaint_chain.invoke({
            "context": context,
            "question": question
        })

        # 尝试解析JSON（LLM有时会多输出文字，需要清理）
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif cleaned.startswith("{"):
            pass  # 正常JSON
        else:
            cleaned = cleaned.split("{", 1)[1].rsplit("}", 1)[0]
            cleaned = "{" + cleaned + "}"

        result = json.loads(cleaned)
        return result

    except Exception as e:
        return {"error": str(e), "raw_output": raw_output if 'raw_output' in locals() else ""}


# 测试
if __name__ == "__main__":
    test_questions = [
        "我用了你们的面膜过敏了，满脸红疹，怎么办？",
        "这个保健品吃完拉肚子了，是不是你们的问题？",
        "退货需要寄回吗？费用谁出？",
        "客服说可以无理由退货，但现在不给退，怎么办"
    ]

    for q in test_questions:
        print(f"\n{'=' * 80}\n问题: {q}\n{'-' * 80}")
        result = handle_complaint(q)

        if "error" in result:
            print("处理出错：", result["error"])
            if "raw_output" in result:
                print("原始输出：", result["raw_output"])
        else:
            try:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except:
                print("JSON解析失败，原始输出：")
                print(result)
        print("\n")