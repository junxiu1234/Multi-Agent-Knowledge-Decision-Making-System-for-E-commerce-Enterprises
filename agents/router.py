import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LLM 配置（通义千问兼容模式）
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-max",  # 或你设定的 LLM_MODEL
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,  # 低温度，分类更稳定
)

# 意图分类 Prompt（Few-shot 示例）
# router_prompt = ChatPromptTemplate.from_messages([
#     ("system", """你是一个企业知识决策系统的意图路由器。
# 请根据用户问题，分类成以下类别之一，只输出类别名，不要多说：
#
# - 客诉处理：涉及投诉、过敏、退货、食品安全、售后问题
# - 合规审核：涉及宣称、广告文案、违规词、营销内容审核
# - 产品查询：询问产品成分、禁忌、使用说明、规格
# - 内部制度：员工手册、奖惩、请假、考勤等内部规则
# - 其他：不属于以上，或无法判断
#
# 用户问题：{question}
# 输出格式：只输出一个类别名，例如 "客诉处理" """),
#     ("human", "{question}")
# ])
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个企业知识决策系统的意图路由器。
请根据用户问题，分类成以下类别之一，只输出类别名，不要多说：

- 客诉处理：涉及投诉、过敏、退货、食品安全、售后问题、产品使用不适
- 合规审核：涉及宣称、广告文案、违规词、营销内容审核、朋友圈/宣传话术是否合规
- 产品查询：询问产品成分、禁忌、使用说明、规格
- 内部制度：员工手册、奖惩、请假、考勤、报销、内部规则咨询
- 供应链：采购、商品上架、入库、退货入库、供应商合规、库存问题
- 其他：不属于以上，或无法判断

Few-shot 示例：
用户：怎么请假？ → 内部制度
用户：迟到扣多少钱？ → 内部制度
用户：这款面膜退货怎么入库？ → 供应链
用户：用了面膜过敏了，怎么办？ → 客诉处理
用户：本品可改善敏感肌，能发朋友圈吗？ → 合规审核

用户问题：{question}
输出格式：只输出一个类别名，例如 "内部制度" """),
    ("human", "{question}")
])
#router_prompt.format_prompt(question="怎么请假？")
# 链
router_chain = router_prompt | llm | StrOutputParser()


def classify_intent(question: str) -> str:
    """输入问题，返回意图类别"""
    result = router_chain.invoke({"question": question})
    return result.strip()


# 测试
if __name__ == "__main__":
    test_questions = [
        "用户说用了面膜过敏了，怎么处理？",
        "这个宣称‘改善敏感肌’能不能发朋友圈？",
        "星衡修护面膜的成分是什么？",
        "公司迟到一次扣多少钱？",
        "今天天气怎么样？"
    ]

    for q in test_questions:
        intent = classify_intent(q)
        print(f"问题: {q}")
        print(f"意图: {intent}\n")