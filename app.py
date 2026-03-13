import streamlit as st
import json
import re
import uuid
import os
from langgraph.checkpoint.redis import RedisSaver
import redis
from langchain_core.messages import HumanMessage, AIMessage
from graph import workflow

# ===================== 全局初始化 =====================
redis_client = redis.Redis(
    host="redis", # docker-compose 启动的 redis 服务 本地测试用 localhost  redis for docker
    port=6379,
    db=0,
    decode_responses=False
)
checkpointer = RedisSaver(redis_client=redis_client)
checkpointer.setup()

graph = workflow.compile(checkpointer=checkpointer)

# ===================== 持久化 thread_id =====================
THREAD_ID_FILE = "current_thread_id.txt"

def get_or_create_thread_id():
    if os.path.exists(THREAD_ID_FILE):
        with open(THREAD_ID_FILE, "r") as f:
            return f.read().strip()
    else:
        new_id = f"user_{str(uuid.uuid4())[:8]}"
        with open(THREAD_ID_FILE, "w") as f:
            f.write(new_id)
        return new_id

# ===================== 从 Redis 恢复历史 =====================
def load_history_from_redis(thread_id: str):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        if state and state.values and "messages" in state.values:
            messages_from_graph = state.values["messages"]
            history = []
            for msg in messages_from_graph:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})
                else:
                    role = "user" if getattr(msg, "type", None) == "human" else "assistant"
                    content = getattr(msg, "content", "")
                    if content:
                        history.append({"role": role, "content": content})
            st.write("从 Redis 加载到 messages 长度：", len(history))
            return history
        else:
            st.info("Redis checkpoint 中没有 messages 字段")
    except Exception as e:
        st.warning(f"恢复历史失败: {str(e)}")
    return []

# ===================== 页面配置 =====================
st.set_page_config(
    page_title="星衡健康多Agent决策系统",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stChatMessage {padding: 1rem; border-radius: 10px; margin-bottom: 1rem;}
    .stSpinner > div {text-align: center;}
    </style>
""", unsafe_allow_html=True)

# ===================== 会话管理 =====================
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = get_or_create_thread_id()

if st.session_state.current_thread_id not in st.session_state.sessions:
    restored = load_history_from_redis(st.session_state.current_thread_id)
    st.session_state.sessions[st.session_state.current_thread_id] = restored or []
    if restored:
        st.success("已从 Redis 恢复历史对话")

with st.sidebar:
    st.title("💬 会话管理")
    st.caption("管理你的对话历史")

    if st.button("➕ 新建对话", use_container_width=True):
        new_id = f"user_{str(uuid.uuid4())[:8]}"
        st.session_state.current_thread_id = new_id
        with open(THREAD_ID_FILE, "w") as f:
            f.write(new_id)
        st.session_state.sessions[new_id] = []
        st.rerun()

    st.divider()
    st.subheader("历史对话")
    for thread_id in list(st.session_state.sessions.keys()):
        msgs = st.session_state.sessions[thread_id]
        if msgs:
            first_msg = msgs[0]["content"][:20] + "..." if len(msgs[0]["content"]) > 20 else msgs[0]["content"]
            label = f"会话 {thread_id[-4:]}: {first_msg}"
        else:
            label = f"会话 {thread_id[-4:]}: (空对话)"
        if st.button(label, key=thread_id, use_container_width=True,
                     disabled=(thread_id == st.session_state.current_thread_id)):
            st.session_state.current_thread_id = thread_id
            with open(THREAD_ID_FILE, "w") as f:
                f.write(thread_id)
            restored = load_history_from_redis(thread_id)
            st.session_state.sessions[thread_id] = restored or []
            st.rerun()

# ===================== 当前会话数据 =====================
current_thread_id = st.session_state.current_thread_id
messages = st.session_state.sessions.get(current_thread_id, [])

# ===================== 页面标题与说明 =====================
st.title("📊 星衡健康多Agent决策系统")
st.caption(f"当前会话：{current_thread_id} | 支持客诉处理、营销合规审核、内部制度咨询、供应链查询 | 基于LangGraph + Redis持久化")
st.divider()

# ===================== 聊天历史展示 =====================
for msg in messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])

# ===================== 用户输入与处理 =====================
question = st.chat_input("请输入你的问题（例如：面膜过敏怎么办？/怎么请假？）")

if question:
    messages.append({"role": "user", "content": question})
    st.session_state.sessions[current_thread_id] = messages

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("系统正在处理，请稍候..."):
            try:
                config = {"configurable": {"thread_id": current_thread_id}}

                # 手动构建 LangChain messages 列表
                langchain_messages = []
                for m in messages[:-1]:  # 排除当前 question
                    if m["role"] == "user":
                        langchain_messages.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=m["content"]))

                input_data = {
                    "question": question,
                    "messages": langchain_messages
                }

                result = graph.invoke(input_data, config=config)

                # ========== 核心修改：正则提取 answer ==========
                intent = result.get("intent", "未知")
                final_reply = result.get("final_reply", "")
                sources = result.get("sources", [])
                raw_result = result.get("result", {})

                if not final_reply:
                    if isinstance(raw_result, dict) and "answer" in raw_result:
                        final_reply = raw_result["answer"]
                        sources = raw_result.get("sources", [])
                    elif isinstance(raw_result, dict) and "raw" in raw_result:
                        raw_str = raw_result["raw"]
                        answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', raw_str)
                        if answer_match:
                            final_reply = answer_match.group(1)
                        sources_match = re.search(r'"sources"\s*:\s*(\[[^\]]*\])', raw_str)
                        if sources_match:
                            try:
                                sources = json.loads(sources_match.group(1))
                            except:
                                sources = []

                if not final_reply:
                    final_reply = "暂无回复内容，请稍后重试或更换问题表述。"

                reply_content = f"""
### 🎯 意图分类：{intent}
### 📝 最终回复：
{final_reply}
                """

                if sources:
                    reply_content += f"""
### 📚 参考来源：
"""
                    for idx, src in enumerate(sources):
                        reply_content += f"{idx+1}. {src}\n"

                if intent == "合规审核" and isinstance(raw_result, dict):
                    compliant = raw_result.get("is_compliant", False)
                    risk_level = raw_result.get("risk_level", "未知")
                    issues = raw_result.get("issues", [])
                    suggested = raw_result.get("suggested_revision", "")

                    status = "✅ 合规" if compliant else "❌ 不合规"
                    reply_content += f"""
### ⚖️ 合规审核结果：
- 审核状态：{status}
- 风险等级：{risk_level}
                    """

                    if issues:
                        reply_content += f"""
- 违规点：
"""
                        for issue in issues:
                            reply_content += f"  - {issue}\n"

                    if suggested and suggested != "原话术已合规":
                        reply_content += f"""
- 推荐修改版：
{suggested}
                        """

                st.markdown(reply_content)

                with st.expander("🔍 完整处理结果（调试用）", expanded=False):
                    st.json(result)

                messages.append({"role": "assistant", "content": reply_content})
                st.session_state.sessions[current_thread_id] = messages

            except Exception as e:
                error_msg = f"❌ 处理失败：{str(e)}"
                st.error(error_msg)
                messages.append({"role": "assistant", "content": error_msg})
                st.session_state.sessions[current_thread_id] = messages

# ===================== 页面底部说明 =====================
st.divider()
st.caption("💡 系统说明：")
st.caption("- 支持客诉处理、合规审核、内部制度咨询、供应链查询四大场景")
st.caption("- 会话记忆基于Redis + LangGraph checkpoint（重启后可恢复）")
st.caption("- 数据依据：chroma_db（向量库）| 通义千问API（DASHSCOPE_API_KEY）")
