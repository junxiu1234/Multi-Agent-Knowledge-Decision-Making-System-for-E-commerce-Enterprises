from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# 加载已构建的向量库
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 测试检索
query = "用户面膜过敏怎么处理"
docs = vectorstore.similarity_search(query, k=3)  # 召回Top-3

print("查询：", query)
print("\n召回的文档：")
for i, doc in enumerate(docs, 1):
    print(f"\n--- 文档 {i} ---")
    print("内容：", doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
    print("来源：", doc.metadata.get('source', '未知'))