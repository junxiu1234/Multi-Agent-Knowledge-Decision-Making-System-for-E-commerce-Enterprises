# import os
# import json
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
#
# # LLM 配置
# llm = ChatOpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     model="qwen-max",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     temperature=0.3,  # 低温度，校验更严谨
# )
#
# # 合规校验 Prompt（已优化：明确要求输出final_reply字段）
# compliance_prompt = ChatPromptTemplate.from_messages([
#     ("system", """你是一个专业的健康产品合规审核Agent，严格遵守广告法、化妆品/保健食品/医疗器械法规。
#
# 输入：用户问题 + 拟回复话术 + 相关检索文档
#
# 任务：
# 1. 检查回复话术是否合规（禁止夸大功效、绝对化用词、医疗术语、误导禁忌人群等）
# 2. 输出结构化JSON：
# {{
#   "is_compliant": true 或 false,
#   "risk_level": "低风险" / "中风险" / "高风险",
#   "issues": ["违规点1描述", "违规点2描述", ...],
#   "suggested_revision": "如果违规，给出修改后的完整回复话术；如果合规，填 "原话术已合规"",
#   "final_reply": "最终可直接使用的完整回复话术（如果合规，用原话术；如果违规，用suggested_revision）",
#   "sources": ["来源文件1", "来源文件2", ...]
# }}
#
# 注意：输出**只包含JSON**，不要任何解释。
#
# 相关文档：
# {context}
#
# 原用户问题：{question}
#
# 拟回复话术：{reply_template}"""),
#     ("human", "开始审核")
# ])
#
# compliance_chain = compliance_prompt | llm | StrOutputParser()
#
# def check_compliance(question: str, reply_template: str, sources: list) -> dict:
#     try:
#         # 简单拼接上下文（后期可加检索合规文档）
#         context = f"来源文件：{', '.join(sources)}"
#
#         raw_output = compliance_chain.invoke({
#             "context": context,
#             "question": question,
#             "reply_template": reply_template
#         })
#
#         # 清理输出为JSON
#         cleaned = raw_output.strip()
#         if cleaned.startswith("```json"):
#             cleaned = cleaned.split("```json")[1].split("```")[0].strip()
#         result = json.loads(cleaned)
#
#         # 强制确保final_reply字段（兼容旧输出）
#         if "final_reply" not in result:
#             if result.get("is_compliant", True):
#                 result["final_reply"] = reply_template
#             else:
#                 result["final_reply"] = result.get("suggested_revision", reply_template)
#
#         return result
#
#     except Exception as e:
#         return {"error": str(e), "raw": raw_output if 'raw_output' in locals() else ""}
#
# # 测试（已调整为合规审核场景）
# if __name__ == "__main__":
#     test_cases = [
#         {
#             "question": "我用了你们的面膜过敏了，满脸红疹，怎么办？",
#             "reply_template": "亲，非常抱歉... 本品可改善敏感肌屏障，建议继续使用...",
#             "sources": ["SOP_过敏投诉处理流.md"]
#         },
#         {
#             "question": "星衡修护面膜能修复敏感肌屏障、根治红血丝、痘印，一敷就好，能做宣传海报吗？",
#             "reply_template": "星衡修护面膜能修复敏感肌屏障、根治红血丝、痘印，一敷就好。",
#             "sources": ["合规_违规宣称示例与正确修改.md", "制度_宣称审核制度.md"]
#         }
#     ]
#
#     for case in test_cases:
#         print(f"\n{'='*80}\n审核问题: {case['question']}\n原话术: {case['reply_template'][:100]}...")
#         result = check_compliance(**case)
#         print(json.dumps(result, ensure_ascii=False, indent=2))


import os
import json
import warnings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 忽略无关警告，提升控制台整洁度
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===================== 1. 基础配置 =====================
# LLM 配置（通义千问）
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,  # 低温度保证审核结果严谨、一致
)

# 向量库配置（对接禁忌词语库）
CHROMA_PATH = r"D:\AIAPPdevelopment\lianxiproject20260121\langchainday\qiyezhishiku\code\chroma_db"
# 加载多语言嵌入模型（适配中文禁忌词文档）
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 初始化向量库和检索器（核心对接禁忌词语库）
try:
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    # 检索器：优先匹配禁忌词库，k=5返回最相关的5条禁忌词规则
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("✅ 禁忌词语向量库加载成功")
except Exception as e:
    print(f"❌ 向量库加载失败: {e}")
    raise

# ===================== 2. 合规审核Prompt（重点适配禁忌词检查） =====================
compliance_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是专业的健康产品合规审核Agent，核心依据「企业宣传禁忌词语库」开展审核，严格遵守广告法、化妆品/保健食品法规。

核心审核规则：
1. 检查拟回复话术是否包含禁忌词库中的违规词汇（如绝对化用词、医疗术语、夸大功效词等）
2. 违规判定需同时参考检索到的禁忌词文档 + 传入的业务来源文档
3. 输出严格结构化JSON，仅返回JSON，无任何多余文字：
{{
  "is_compliant": true 或 false,
  "risk_level": "低风险" / "中风险" / "高风险",
  "issues": ["违规点1（标注涉及的禁忌词）", "违规点2..."],
  "suggested_revision": "违规则给出修改后的话术，合规则填「原话术已合规」",
  "final_reply": "最终可用话术（合规用原话术，违规用修改版）",
  "sources": ["来源文件1", "来源文件2..."]  // 合并检索到的禁忌词文档+传入的业务文档
}}

【检索到的禁忌词语库内容】
{context}

【原用户问题】
{question}

【拟回复话术】
{reply_template}

【业务来源文档】
{sources_text}"""),
    ("human", "基于上述信息，严格审核拟回复话术的合规性")
])

# 构建链式调用
compliance_chain = compliance_prompt | llm | StrOutputParser()


# ===================== 3. 核心审核函数（修复：添加sources参数） =====================
def check_compliance(question: str, reply_template: str, sources: list) -> dict:
    """
    合规审核函数（适配禁忌词库检索+保留sources参数）
    :param question: 用户原始问题
    :param reply_template: 拟回复话术
    :param sources: 业务来源文档列表（如SOP、制度文件）
    :return: 结构化审核结果
    """
    try:
        # 步骤1：格式化传入的业务sources为文本（供Prompt使用）
        sources_text = "\n".join([f"- {source}" for source in sources]) if sources else "无业务来源文档"

        # 步骤2：基于「拟回复话术」检索禁忌词库（核心调用禁忌词资源）
        search_query = f"审核话术是否包含禁忌词：{reply_template}"
        docs = retriever.invoke(search_query)

        # 步骤3：格式化禁忌词库检索结果
        context_list = []
        retrieved_sources = []
        for doc in docs:
            doc_source = doc.metadata.get('source', '禁忌词语库.md')
            doc_content = doc.page_content[:800]  # 截断避免输入过长
            context_list.append(f"【{doc_source}】\n{doc_content}...")
            retrieved_sources.append(doc_source)

        context = "\n\n".join(context_list) if context_list else "未检索到相关禁忌词规则"

        # 步骤4：调用大模型审核（传入context/question/reply_template/sources_text）
        raw_output = compliance_chain.invoke({
            "context": context,
            "question": question,
            "reply_template": reply_template,
            "sources_text": sources_text  # 新增：传入业务来源文本
        })

        # 步骤5：清理并解析JSON
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        result = json.loads(cleaned)

        # 步骤6：字段兜底+合并来源（业务sources + 禁忌词库sources）
        # 保证final_reply必存在
        if "final_reply" not in result:
            result["final_reply"] = reply_template if result.get("is_compliant") else result.get("suggested_revision",
                                                                                                 reply_template)
        # 合并传入的业务来源 + 检索到的禁忌词库来源（去重）
        result["sources"] = list(set(sources + retrieved_sources))

        return result

    except Exception as e:
        # 异常兜底，保证返回格式统一
        return {
            "error": str(e),
            "is_compliant": False,
            "risk_level": "高风险",
            "issues": ["系统审核异常，需人工复核"],
            "suggested_revision": "因系统异常，暂无法生成合规回复，请人工处理",
            "final_reply": "非常抱歉，当前无法为您提供回复，我们会尽快处理",
            "sources": sources  # 保留传入的业务来源
        }


# ===================== 4. 测试代码（保留sources入参） =====================
if __name__ == "__main__":
    # 测试用例：保留sources字段，适配禁忌词库审核场景
    test_cases = [
        {
            "question": "我用了你们的面膜过敏了，满脸红疹，怎么办？",
            "reply_template": "亲，非常抱歉给您带来不适！本品可改善敏感肌屏障，建议继续使用观察~",
            "sources": ["SOP_过敏投诉处理流.md"]
        },
        {
            "question": "星衡修护面膜能修复敏感肌屏障、根治红血丝、痘印，一敷就好，能做宣传海报吗？",
            "reply_template": "星衡修护面膜能修复敏感肌屏障、根治红血丝、痘印，一敷就好，效果立竿见影！",
            "sources": ["化妆品功效宣称评价规范.md"]
        }
    ]

    # 执行测试
    for idx, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"【测试用例 {idx}】")
        print(f"问题: {case['question']}")
        print(f"拟回复话术: {case['reply_template']}")
        print(f"业务来源: {case['sources']}")
        print(f"{'=' * 80}")

        # 调用审核函数（保留sources入参）
        result = check_compliance(**case)

        # 格式化输出结果
        print(json.dumps(result, ensure_ascii=False, indent=2))