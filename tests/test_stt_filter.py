import pytest

from stt import Transcript, clean_text, collapse_repeats


@pytest.mark.parametrize("text", [
    "", "   ", "Thank you.", "thank you", "Thanks for watching!", "you", "...", "♪♪♪", "♪",
    "字幕由Amara.org社区提供", "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目", "Subtitles by the Amara.org community",
    "谢谢观看", "中文字幕志愿者 李宗盛",
])
def test_hallucinations_are_dropped(text):
    assert clean_text(text) is None


@pytest.mark.parametrize("text", [
    "Thank you for the update, what is the timeline?",
    "What is the budget?",
    "我们下周上线。",
    "OK.",
])
def test_real_text_is_kept(text):
    assert clean_text(text) == text


def test_repeats_are_collapsed():
    assert collapse_repeats("the the the the the the plan") == "the the the plan"
    assert clean_text("好的好的好的好的好的好的") == "好的好的好的"


def test_transcript_window_and_previous():
    tr = Transcript()
    a = tr.add(10, "first")
    b = tr.add(100, "second")
    c = tr.add(250, "third")
    assert tr.previous(c.id) is b
    assert tr.previous(a.id) is None
    assert [u.text for u in tr.window(250, 3)] == ["second", "third"]
    assert tr.last() is c and tr.get(b.id) is b
