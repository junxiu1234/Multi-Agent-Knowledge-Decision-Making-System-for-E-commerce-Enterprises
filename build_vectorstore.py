import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 知识库路径（从 code/ 目录相对路径）
KNOWLEDGE_DIR = "../qiyezhishikuziliao"

# 通义千问配置（硬编码，先这样调试，后面可抽取）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 如果系统环境变量有，就取得到
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置系统环境变量 DASHSCOPE_API_KEY")

print("开始加载文档...")

# 加载所有 .md 文件（支持中文文件名）
loader = DirectoryLoader(
    KNOWLEDGE_DIR,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()
print(f"成功加载 {len(docs)} 个文档")

# 文本切分（chunk_size 1000 比较合适，overlap 200 防止断句）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)
split_docs = text_splitter.split_documents(docs)
print(f"切分成 {len(split_docs)} 个 chunk")


# ===================== 新增：可视化切分后的chunk =====================
def print_split_chunks(chunks, show_all=False, show_first_n=5):
    """
    打印切分后的chunk详情
    :param chunks: 切分后的Document列表
    :param show_all: 是否打印所有chunk（如果文档多，建议设为False）
    :param show_first_n: 只打印前N个chunk（避免输出过长）
    """
    # 确定要打印的chunk数量
    print_num = len(chunks) if show_all else min(show_first_n, len(chunks))

    print("=" * 100)
    print(f"📝 切分后的chunk详情（共{len(chunks)}个，展示前{print_num}个）")
    print("=" * 100)

    for i, doc in enumerate(chunks[:print_num]):
        # 提取关键信息
        source = doc.metadata.get('source', '未知文件')  # 来源文件路径
        start_index = doc.metadata.get('start_index', 0)  # 在原文档中的起始位置
        content = doc.page_content  # chunk内容
        content_length = len(content)  # chunk字符长度

        # 打印单个chunk信息
        print(f"\n【Chunk {i + 1}】")
        print(f"📂 来源文件：{source}")
        print(f"📍 原文档起始位置：{start_index}")
        print(f"📏 字符长度：{content_length}")
        print(f"📄 内容：\n{content}")
        print("-" * 80)


# 调用函数查看chunk（默认只看前5个，避免输出过长）
# 如果想查看所有chunk，把 show_all=True 即可
print_split_chunks(split_docs, show_all=False, show_first_n=5)
# 嵌入模型（免费、中文支持好）
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("开始构建 Chroma 向量库...")

# 构建并持久化到 ./chroma_db
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("向量库构建完成！")
print("持久化目录：", os.path.abspath("./chroma_db"))
print("你可以随时用 Chroma(persist_directory='./chroma_db', embedding_function=embeddings) 加载")