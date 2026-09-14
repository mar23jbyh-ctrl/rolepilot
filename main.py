from app.service import InterviewService
from app.branding import PRODUCT_NAME, PRODUCT_DESCRIPTION


def run_interview():
    print(f"{PRODUCT_NAME} · {PRODUCT_DESCRIPTION}")
    service = InterviewService()
    jd = input("\n请输入岗位 JD（职位描述）：\n")
    resume = input("\n请输入候选人简历文本：\n")
    result = service.start_session(jd_text=jd, resume_text=resume)
    session_id = result["session_id"]
    report = (result.get("report") or (result.get("state") or {}).get("evaluation_report"))
    while not report:
        question = result.get("question", "")
        if not question:
            print("当前没有可回答的问题，请检查会话状态。")
            return
        print("\n面试官：" + str(question))
        answer = input("\n候选人：")
        result = service.submit_answer(session_id, answer)
        report = result.get("report")
    print("\n" + "=" * 30)
    print("面试评估报告")
    print("=" * 30)
    print(report.get("text_analysis", ""))
    print("平均分：" + str(report.get("overall_score", "")))
    print("学习路径：" + " -> ".join(report.get("learning_path", [])))
    print("推荐资源：" + "；".join(report.get("resources", [])))


if __name__ == "__main__":
    run_interview()
