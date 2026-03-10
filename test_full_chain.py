import os
import json
import re
import datetime
from json_repair import repair_json
from agents.router import classify_intent
from agents.complaint import handle_complaint
from agents.compliance import check_compliance
from agents.internal_policy import handle_internal
from agents.supply_chain import handle_supply

LOG_FILE = "execution_log.txt"


def safe_parse_json(raw: str):
    if not raw:
        return {"error": "空输出"}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            repaired = repair_json(raw)
            return json.loads(repaired)
        except Exception as e:
            return {"error": f"JSON解析失败: {str(e)}", "raw": raw}


def log_to_file(question, intent, result=None, final_reply="", error_msg=""):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"\n{'=' * 80}\n[{timestamp}] 问题: {question}\n意图: {intent}\n"

    if error_msg:
        log_entry += f"错误: {error_msg}\n"

    if result:
        log_entry += f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}\n"

    log_entry += f"最终回复话术:\n{final_reply.strip()}\n{'=' * 80}\n"

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)


def clean_path(path: str) -> str:
    return re.sub(r'\\+', '/', path).replace('..', '..')


def process_question(question: str):
    print(f"\n{'=' * 100}\n原始问题: {question}\n{'-' * 50}")

    intent = classify_intent(question)
    print(f"意图分类: {intent}")

    final_reply = ""
    result = None

    if intent == "客诉处理":
        complaint_raw = handle_complaint(question)
        result = safe_parse_json(complaint_raw) if isinstance(complaint_raw, str) else complaint_raw

        if "error" in result:
            print("客诉处理出错：", result["error"])
            log_to_file(question, intent, result=result, error_msg=result["error"])
            return

        print("\n客诉初步处理结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        final_reply = result.get("reply_template", "无回复模板")

        if result.get("need_compliance_check", False):
            print("\n需要合规校验，正在调用合规Agent...")
            compliance_raw = check_compliance(
                question=question,
                reply_template=final_reply,
                sources=result.get("sources", [])
            )
            compliance_result = safe_parse_json(compliance_raw) if isinstance(compliance_raw, str) else compliance_raw

            if "error" in compliance_result:
                print("合规校验出错：", compliance_result["error"])
            else:
                print("\n合规校验结果:")
                print(json.dumps(compliance_result, ensure_ascii=False, indent=2))

                if not compliance_result.get("is_compliant", True):
                    final_reply = compliance_result.get("suggested_revision", final_reply)
                    print("\n最终推荐回复话术（已修订）:")
                else:
                    print("\n原话术已通过合规校验，无需修改")

    elif intent == "内部制度":
        print("\n进入内部制度查询模式...")
        result = handle_internal(question)
        if "error" in result:
            print("内部制度查询出错：", result["error"])
        else:
            print("\n内部制度查询结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            final_reply = result.get("answer", "无回复")

    elif intent == "供应链":
        print("\n进入供应链查询模式...")
        result = handle_supply(question)
        if "error" in result:
            print("供应链查询出错：", result["error"])
        else:
            print("\n供应链查询结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            final_reply = result.get("answer", "无回复")

    else:
        print("当前仅支持客诉处理、内部制度、供应链场景，其他意图后续扩展")
        log_to_file(question, intent)
        return

    # 美化输出
    print("\n【最终可使用回复/结果】")
    print(final_reply.strip())

    if result and "steps" in result:
        print("\n【处理步骤】")
        for i, step in enumerate(result["steps"], 1):
            print(f"{i}. {step}")

    print("\n【参考来源】")
    sources = result.get("sources", []) if result else []
    if sources:
        for src in sources:
            print(f"- {clean_path(src)}")
    else:
        print("无来源引用")

    # 保存日志
    log_to_file(question, intent, result, final_reply=final_reply)

    print(f"{'=' * 100}\n")


# 测试用例（覆盖四大职能）
if __name__ == "__main__":
    test_questions = [
        "我用了你们的面膜过敏了，满脸红疹，怎么办？",  # 客诉
        "怎么请假？",  # 内部制度
        "迟到一次扣多少钱？",  # 内部制度
        "这款面膜退货怎么入库？",  # 供应链
        "供应商合规要求是什么？",  # 供应链
        "本品可改善敏感肌，能发朋友圈吗？"  # 合规审核
    ]

    for q in test_questions:
        process_question(q)