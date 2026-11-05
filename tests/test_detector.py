import pytest

from detector import is_question

QUESTIONS = [
    "What is the timeline for the migration?",
    "what is the timeline for the migration",  # Whisper dropped the '?'
    "Why did the job fail",
    "How long will the cut-over take",
    "Can we roll back if it fails",
    "Could you share the budget numbers",
    "Would that work for the team",
    "Do you have the latest numbers",
    "Does the service support vector search",
    "Is it ready for production",
    "Are we on track",
    "So, how does the reconciliation work",
    "Okay and what's the status of phase two",
    "Tell me about the risks.",
    "Please explain the rollback plan.",
    "Walk me through the architecture.",
    "I was wondering about the vendor support.",
    "We finished phase one. When is the cut-over",
    "这个项目的进度怎么样",
    "你们的预算还剩多少",
    "为什么对账任务变慢了",
    "如何回滚",
    "有没有备选方案",
    "是不是已经上线了",
    "能不能下周完成",
    "会不会影响用户",
    "这个方案可行吗",
    "那性能呢",
    "哪个团队负责",
    "是否需要停机",
    "进度？",
]

STATEMENTS = [
    "We finished the schema conversion last week.",
    "The budget is fine.",
    "Let's move on to the next topic.",
    "Thanks everyone for joining.",
    "我们上周完成了数据迁移。",
    "预算没问题。",
    "",
    "   ",
]


@pytest.mark.parametrize("text", QUESTIONS)
def test_questions(text):
    assert is_question(text), text


@pytest.mark.parametrize("text", STATEMENTS)
def test_statements(text):
    assert not is_question(text), text
