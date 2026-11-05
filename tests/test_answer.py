import sys
import threading
import time
from pathlib import Path

import pytest

import answer
from answer import (AnswerEngine, AskRequest, ClaudeConfig, CommandBuilder, OneShotClaude, PersistentClaude,
                    Turn, build_message, child_env, obsidian_url)
from stt import Transcript

FAKE = str(Path(__file__).resolve().parent / "fake_claude.py")


# ---- Turn: stream-json parsing ------------------------------------------- #
def text_events(text, step=2):
    yield {"type": "stream_event", "event": {"type": "message_start"}}
    for i in range(0, len(text), step):
        yield {"type": "stream_event",
               "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text[i:i + step]}}}
    yield {"type": "stream_event", "event": {"type": "message_stop"}}


def run_turn(events):
    deltas, resets, tools = [], [], []
    turn = Turn(on_delta=deltas.append, on_reset=lambda: resets.append(1), on_tool=tools.append)
    for e in events:
        turn.handle(e)
    return turn, "".join(deltas), resets, tools


def result(text="", is_error=False):
    return {"type": "result", "subtype": "success", "is_error": is_error, "result": text}


def test_turn_streams_text():
    turn, shown, _, _ = run_turn([*text_events("Yes, on **Nov 20**.\n- a\n- b"), result("x")])
    assert shown == "Yes, on **Nov 20**.\n- a\n- b"
    assert not turn.skipped and turn.done.is_set() and turn.ms_first is not None


@pytest.mark.parametrize("skip", ["SKIP", "SKIP\n", " SKIP.", "**SKIP**"])
def test_turn_skip_is_suppressed(skip):
    turn, shown, _, _ = run_turn([*text_events(skip, step=1), result(skip)])
    assert shown == ""
    assert turn.skipped


def test_turn_words_starting_with_s_are_not_skipped():
    turn, shown, _, _ = run_turn([*text_events("Sure. Skipping tests is fine.", step=1), result()])
    assert shown == "Sure. Skipping tests is fine."
    assert not turn.skipped


def test_turn_short_answer_flushed_at_message_end():
    turn, shown, _, _ = run_turn([*text_events("SK"), result("SK")])
    assert shown == "SK" and not turn.skipped


def test_turn_text_before_tool_call_is_withdrawn():
    events = [
        *list(text_events("Let me check the vault."))[:-1],
        {"type": "stream_event", "event": {"type": "content_block_start",
                                           "content_block": {"type": "tool_use", "name": "Grep"}}},
        {"type": "stream_event", "event": {"type": "message_stop"}},
        *text_events("Budget left: **$110k** [1]."),
        result(),
    ]
    turn, shown, resets, tools = run_turn(events)
    assert tools == ["Grep"]
    assert resets  # UI was told to clear the preamble
    assert turn.visible == "Budget left: **$110k** [1]."


def test_turn_error_result():
    err = {"type": "assistant", "error": "authentication_failed", "is_api_error_message": True,
           "message": {"content": [{"type": "text", "text": "Failed to authenticate"}]}}
    turn, shown, _, _ = run_turn([err, result("Failed to authenticate", is_error=True)])
    assert turn.error == "Failed to authenticate"
    assert shown == "" and not turn.skipped


def test_turn_without_partial_messages_uses_assistant_message():
    events = [{"type": "assistant", "message": {"content": [{"type": "text", "text": "Plain answer."}]}},
              result("Plain answer.")]
    turn, shown, _, _ = run_turn(events)
    assert shown == "Plain answer."


# ---- prompt ------------------------------------------------------------------ #
def test_build_message_format(vault):
    tr = Transcript()
    tr.add(5, "Morning all.")
    u = tr.add(65, "What is the Apollo timeline?")
    hits = vault.search(u.text)
    msg = build_message("I lead the data team.", tr.all(), [{"t": u.t, "text": u.text}], hits)
    for tag in ("<about_me>", "<recent_transcript>", "<questions>", "<notes>"):
        assert tag in msg
    assert "[00:05] Morning all." in msg
    assert "[01:05] What is the Apollo timeline?" in msg
    assert "[1] Projects/Apollo Migration.md › Apollo Migration › Timeline" in msg
    assert "<mode>manual</mode>" not in msg
    assert "<mode>manual</mode>" in build_message("", [], [{"t": None, "text": "q"}], [], manual=True)


def test_obsidian_url():
    assert obsidian_url("My Vault", "Projects/Apollo Migration.md") == \
        "obsidian://open?vault=My%20Vault&file=Projects%2FApollo%20Migration.md"


def test_child_env_drops_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "x")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "keep-me")
    env = child_env()
    assert "ANTHROPIC_API_KEY" not in env and "CLAUDECODE" not in env and "CLAUDE_CODE_ENTRYPOINT" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "keep-me"


# ---- command line -------------------------------------------------------------- #
_RealBuilder = CommandBuilder


def fake_builder(cfg=None, vault_path=None, exclude=()):
    """A CommandBuilder whose 'claude' is tests/fake_claude.py."""
    cfg = cfg or ClaudeConfig(cli=sys.executable)
    answer._HELP_CACHE[" ".join(answer.resolve_cli(cfg.cli))] = (
        "--input-format stream-json --include-partial-messages --tools --no-session-persistence "
        "--strict-mcp-config --disable-slash-commands --effort --settings"
    )
    b = _RealBuilder(cfg, vault_path, exclude)
    b.argv0 = [sys.executable, FAKE]
    b.via_shell_shim = False
    return b


def test_command_line_flags(vault_path):
    b = fake_builder(ClaudeConfig(cli=sys.executable, model="haiku"), str(vault_path))
    cmd = b.persistent()
    s = " ".join(cmd)
    assert cmd[2:5] == ["-p", "--input-format", "stream-json"]
    for flag in ("--output-format stream-json", "--verbose", "--include-partial-messages", "--model haiku",
                 "--allowedTools Read,Grep,Glob", "--add-dir"):
        assert flag in s
    assert "--append-system-prompt" in cmd
    assert cmd[cmd.index("--append-system-prompt") + 1] == answer.SYSTEM_RULES
    assert "--disallowedTools" not in cmd
    b2 = fake_builder(ClaudeConfig(cli=sys.executable, allow_vault_tools=False))
    assert "--add-dir" not in b2.oneshot() and b2.oneshot()[b2.oneshot().index("--tools") + 1] == ""


def test_excluded_folders_become_deny_rules(vault_path):
    assert answer.deny_rules(["NDA/**", "Private", "secret/", "*.excalidraw.md", "a/b.md"]) == [
        "Read(./NDA/**)", "Read(./Private/**)", "Read(./secret/**)", "Read(./**/*.excalidraw.md)", "Read(./a/b.md)"]
    cmd = fake_builder(ClaudeConfig(cli=sys.executable), str(vault_path), ["NDA/**"]).persistent()
    i = cmd.index("--disallowedTools")
    assert cmd[i + 1] == "Read(./NDA/**)" and cmd[i + 2].startswith("--")
    no_tools = fake_builder(ClaudeConfig(cli=sys.executable, allow_vault_tools=False), str(vault_path), ["NDA/**"])
    assert "--disallowedTools" not in no_tools.persistent()


# ---- runners against the fake CLI ---------------------------------------------- #
def ask(runner, text):
    deltas = []
    t = Turn(on_delta=deltas.append)
    runner.ask(text, t)
    return t, "".join(deltas)


def test_persistent_process_multiple_turns():
    r = PersistentClaude(fake_builder(), max_minutes=30, timeout=20)
    r.start()
    try:
        pid = r.proc.pid
        t1, s1 = ask(r, "<questions>\nQ one\n</questions>")
        t2, s2 = ask(r, "<questions>\nQ two\n</questions>")
        t3, s3 = ask(r, "<questions>\nSMALLTALK\n</questions>")
        assert s1.startswith("Answer: Q one") and s2.startswith("Answer: Q two")
        assert t3.skipped and s3 == ""
        assert r.proc.pid == pid  # same process for every turn
    finally:
        r.stop()


def test_persistent_restarts_after_crash():
    r = PersistentClaude(fake_builder(), timeout=20)
    r.start()
    try:
        t, _ = ask(r, "CRASH")
        assert t.error and "exited" in t.error
        t2, s2 = ask(r, "<questions>\nafter crash\n</questions>")
        assert s2.startswith("Answer: after crash") and not t2.error
    finally:
        r.stop()


def test_persistent_recycles_old_session():
    r = PersistentClaude(fake_builder(), max_minutes=30, timeout=20)
    r.start()
    try:
        pid = r.proc.pid
        r.started_at -= 31 * 60
        assert r.expired
        _, s = ask(r, "<questions>\nfresh\n</questions>")
        assert s.startswith("Answer: fresh") and r.proc.pid != pid
    finally:
        r.stop()


def test_oneshot_runner():
    r = OneShotClaude(fake_builder(), timeout=20)
    t, s = ask(r, "<questions>\none shot\n</questions>")
    assert s.startswith("Answer: one shot")


# ---- engine ------------------------------------------------------------------------ #
class Recorder:
    def __init__(self):
        self.events = []
        self.cv = threading.Condition()

    def __call__(self, event, data):
        with self.cv:
            self.events.append((event, data))
            self.cv.notify_all()

    def wait_for(self, event, n=1, timeout=20):
        deadline = time.time() + timeout
        with self.cv:
            while sum(1 for e, _ in self.events if e == event) < n:
                left = deadline - time.time()
                if left <= 0:
                    raise AssertionError(f"timed out waiting for {event}: {[e for e, _ in self.events]}")
                self.cv.wait(left)
        return [d for e, d in self.events if e == event]


def test_engine_dry_run_shows_sources(vault):
    tr = Transcript()
    tr.add(1, "Let's talk about Apollo.")
    u = tr.add(4, "How much budget is left?")
    rec = Recorder()
    eng = AnswerEngine(ClaudeConfig(), vault, tr, rec, dry_run=True)
    eng.start()
    eng.submit(AskRequest(utt_id=u.id))
    start = rec.wait_for("answer_start")[0]
    end = rec.wait_for("answer_end")[0]
    eng.stop()
    assert start["questions"][0]["text"] == u.text
    assert start["sources"][0]["title"] == "Apollo Migration › Budget"
    assert start["sources"][0]["url"].startswith("obsidian://open?vault=vault&file=Projects%2F")
    assert end["dry_run"] is True
    assert not any(e == "answer_delta" for e, _ in rec.events)


def test_engine_streams_merges_and_skips(vault, monkeypatch):
    tr = Transcript()
    rec = Recorder()
    cfg = ClaudeConfig(cli=sys.executable, warmup=True)
    eng = AnswerEngine(cfg, vault, tr, rec)
    monkeypatch.setattr(answer, "CommandBuilder", lambda c, p, e=(): fake_builder(c, p, e))
    # two questions arrive while Claude is still busy (here: warming up) -> one merged request
    q1 = tr.add(10, "What is the Apollo timeline?")
    q2 = tr.add(14, "And how much budget is left?")
    eng.submit(AskRequest(utt_id=q2.id))
    eng.submit(AskRequest(utt_id=q1.id))
    eng.submit(AskRequest(utt_id=q1.id))  # duplicate is ignored
    eng.start()
    start = rec.wait_for("answer_start")[0]
    end = rec.wait_for("answer_end")[0]
    assert [q["utt_id"] for q in start["questions"]] == [q1.id, q2.id]  # merged into one request
    text = "".join(d.get("text", "") for e, d in rec.events if e == "answer_delta" and d["id"] == start["id"])
    assert text.startswith("Answer: [00:10] What is the Apollo timeline?")
    assert end["ms_first"] is not None

    s = tr.add(20, "SMALLTALK how was your weekend?")
    eng.submit(AskRequest(utt_id=s.id))
    skip = rec.wait_for("answer_skip")[0]
    eng.stop()
    assert skip["id"] == start["id"] + 1
    assert eng.status()["claude_mode"] == "persistent"
