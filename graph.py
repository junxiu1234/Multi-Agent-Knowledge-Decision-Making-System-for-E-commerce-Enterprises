# import json
# from typing import TypedDict, Annotated
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langchain_core.messages import BaseMessage
# from agents.router import classify_intent
# from agents.complaint import handle_complaint
# from agents.compliance import check_compliance
# from agents.internal_policy import handle_internal
# from agents.supply_chain import handle_supply
#
# class AgentState(TypedDict):
#     question: str
#     messages: Annotated[list[BaseMessage], add_messages]
#     intent: str
#     result: dict
#     final_reply: str
#     sources: list
#
# def router_node(state: AgentState) -> AgentState:
#     intent = classify_intent(state["question"])
#     return {"intent": intent}
#
# def complaint_node(state: AgentState) -> AgentState:
#     if state["intent"] == "客诉处理":
#         result = handle_complaint(state["question"])
#         return {
#             "result": result,
#             "final_reply": result.get("reply_template", ""),
#             "sources": result.get("sources", [])
#         }
#     return state
#
# def compliance_node(state: AgentState) -> AgentState:
#     reply_template = state.get("final_reply", state["question"])
#     sources = state.get("sources", [])
#
#     compliance_result = check_compliance(
#         question=state["question"],
#         reply_template=reply_template,
#         sources=sources
#     )
#
#     state["final_reply"] = compliance_result.get("final_reply", reply_template)
#     return {"result": compliance_result, "final_reply": state["final_reply"]}
#
# def internal_node(state: AgentState) -> AgentState:
#     if state["intent"] == "内部制度":
#         result = handle_internal(state["question"])
#         return {
#             "result": result,
#             "final_reply": result.get("answer", ""),
#             "sources": result.get("sources", [])
#         }
#     return state
#
# def supply_node(state: AgentState) -> AgentState:
#     if state["intent"] == "供应链":
#         result = handle_supply(state["question"])
#         return {
#             "result": result,
#             "final_reply": result.get("answer", ""),
#             "sources": result.get("sources", [])
#         }
#     return state
#
# def route_intent(state: AgentState):
#     intent = state["intent"]
#     if intent == "客诉处理":
#         return "complaint"
#     elif intent == "内部制度":
#         return "internal"
#     elif intent == "供应链":
#         return "supply"
#     elif intent == "合规审核":
#         return "compliance"
#     else:
#         return END
#
# def route_after_complaint(state: AgentState):
#     if state.get("result", {}).get("need_compliance_check", False):
#         return "compliance"
#     return END
#
# workflow = StateGraph(state_schema=AgentState)
#
# workflow.add_node("router", router_node)
# workflow.add_node("complaint", complaint_node)
# workflow.add_node("compliance", compliance_node)
# workflow.add_node("internal", internal_node)
# workflow.add_node("supply", supply_node)
#
# workflow.set_entry_point("router")
# workflow.add_conditional_edges("router", route_intent, {
#     "complaint": "complaint",
#     "internal": "internal",
#     "supply": "supply",
#     "compliance": "compliance",
#     END: END
# })
#
# workflow.add_conditional_edges("complaint", route_after_complaint, {"compliance": "compliance", END: END})
# workflow.add_edge("compliance", END)
# workflow.add_edge("internal", END)
# workflow.add_edge("supply", END)
#
# # 文件检查点（SQLiteSaver） - 正确方式：with 块包含 compile 和 invoke
# if __name__ == "__main__":
#     with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
#         graph = workflow.compile(checkpointer=checkpointer)
#
#         config = {"configurable": {"thread_id": "test_user_1"}}  # 同一个用户会话
#
#         tests = [
#             "我用了你们的面膜过敏了，满脸红疹，怎么办？",
#             "我用了3次，从上周开始，脸红肿瘙痒，去医院了，医生说是过敏。",
#             "怎么请假？",
#             "这款面膜退货怎么入库？",
#             "星衡修护面膜能修复敏感肌屏障、根治红血丝、痘印，一敷就好，能做宣传海报吗？"
#         ]
#
#         for i, q in enumerate(tests, 1):
#             print(f"\n{'='*80}\n第{i}轮: {q}")
#             result = graph.invoke({"question": q, "messages": []}, config)
#             print(json.dumps(result, ensure_ascii=False, indent=2))
#             print("\n最终回复：")
#             print(result.get("final_reply", "无"))











import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.redis import RedisSaver  # 新增
import redis  # 新增
from langchain_core.messages import BaseMessage
from agents.router import classify_intent
from agents.complaint import handle_complaint
from agents.compliance import check_compliance
from agents.internal_policy import handle_internal
from agents.supply_chain import handle_supply

class AgentState(TypedDict):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    result: dict
    final_reply: str
    sources: list

def router_node(state: AgentState) -> AgentState:
    intent = classify_intent(state["question"])
    return {"intent": intent}

def complaint_node(state: AgentState) -> AgentState:
    if state["intent"] == "客诉处理":
        result = handle_complaint(state["question"])
        return {
            "result": result,
            "final_reply": result.get("reply_template", ""),
            "sources": result.get("sources", [])
        }
    return state

def compliance_node(state: AgentState) -> AgentState:
    reply_template = state.get("final_reply", state["question"])
    sources = state.get("sources", [])

    compliance_result = check_compliance(
        question=state["question"],
        reply_template=reply_template,
        sources=sources
    )

    state["final_reply"] = compliance_result.get("final_reply", reply_template)
    return {"result": compliance_result, "final_reply": state["final_reply"]}

def internal_node(state: AgentState) -> AgentState:
    if state["intent"] == "内部制度":
        result = handle_internal(state["question"])
        return {
            "result": result,
            "final_reply": result.get("answer", ""),
            "sources": result.get("sources", [])
        }
    return state

def supply_node(state: AgentState) -> AgentState:
    if state["intent"] == "供应链":
        result = handle_supply(state["question"])
        return {
            "result": result,
            "final_reply": result.get("answer", ""),
            "sources": result.get("sources", [])
        }
    return state

def route_intent(state: AgentState):
    intent = state["intent"]
    if intent == "客诉处理":
        return "complaint"
    elif intent == "内部制度":
        return "internal"
    elif intent == "供应链":
        return "supply"
    elif intent == "合规审核":
        return "compliance"
    else:
        return END

def route_after_complaint(state: AgentState):
    if state.get("result", {}).get("need_compliance_check", False):
        return "compliance"
    return END

workflow = StateGraph(state_schema=AgentState)

workflow.add_node("router", router_node)
workflow.add_node("complaint", complaint_node)
workflow.add_node("compliance", compliance_node)
workflow.add_node("internal", internal_node)
workflow.add_node("supply", supply_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_intent, {
    "complaint": "complaint",
    "internal": "internal",
    "supply": "supply",
    "compliance": "compliance",
    END: END
})

workflow.add_conditional_edges("complaint", route_after_complaint, {"compliance": "compliance", END: END})
workflow.add_edge("compliance", END)
workflow.add_edge("internal", END)
workflow.add_edge("supply", END)

# ─────────────── RedisSaver 部分 ───────────────
redis_client = redis.Redis(
    host="redis",       # 本地测试用 localhost  redis for docker
    port=6379,
    db=0,
    decode_responses=False  # 必须 False，LangGraph 需要二进制存储
)

checkpointer = RedisSaver(redis_client=redis_client)

# 如果你用的是 docker run 的 redis，可以直接用下面这行更简单（推荐）
# checkpointer = RedisSaver.from_conn_string("redis://localhost:6379/0")

graph = workflow.compile(checkpointer=checkpointer)

# 下面是你原来的测试代码，保持不变
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test_user_1"}}

    tests = [
        "我用了你们的面膜过敏了，满脸红疹，怎么办？",
        "我用了3次，从上周开始，脸红肿瘙痒，去医院了，医生说是过敏。",
        "怎么请假？",
        "这款面膜退货怎么入库？",
        "星衡修护面膜能修复敏感肌屏障、根治红血丝、痘印，一敷就好，能做宣传海报吗？"
    ]

    for i, q in enumerate(tests, 1):
        print(f"\n{'='*80}\n第{i}轮: {q}")
        result = graph.invoke({"question": q, "messages": []}, config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("\n最终回复：")
        print(result.get("final_reply", "无"))