"""A stand-in for `claude -p ... --output-format stream-json` used by the tests.

Reply rules (based on the user message):
  contains "<warmup/>" or "SMALLTALK"  -> "SKIP"
  contains "USE_TOOL"                    -> preamble text, a Grep tool call, then the answer
  contains "CRASH"                       -> exit(3) without answering
  otherwise                              -> "Answer: <first line of <questions>>"
"""
import json
import re
import sys
import time
import uuid

SESSION = str(uuid.uuid4())


def out(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def stream_text_message(text, stop_reason="end_turn", tool=None):
    out({"type": "stream_event", "event": {"type": "message_start", "message": {"id": "m", "content": []}}})
    out({"type": "stream_event", "event": {"type": "content_block_start", "index": 0,
                                           "content_block": {"type": "text", "text": ""}}})
    for i in range(0, len(text), 3):
        out({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                               "delta": {"type": "text_delta", "text": text[i:i + 3]}}})
        time.sleep(0.002)
    out({"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}})
    content = [{"type": "text", "text": text}]
    if tool:
        out({"type": "stream_event", "event": {"type": "content_block_start", "index": 1,
                                               "content_block": {"type": "tool_use", "name": tool, "id": "t1",
                                                                 "input": {}}}})
        out({"type": "stream_event", "event": {"type": "content_block_stop", "index": 1}})
        content.append({"type": "tool_use", "name": tool, "id": "t1", "input": {}})
    out({"type": "stream_event", "event": {"type": "message_delta", "delta": {"stop_reason": stop_reason}}})
    out({"type": "stream_event", "event": {"type": "message_stop"}})
    out({"type": "assistant", "message": {"role": "assistant", "content": content}, "session_id": SESSION})


def reply(prompt):
    out({"type": "system", "subtype": "init", "session_id": SESSION, "tools": []})
    if "CRASH" in prompt:
        sys.exit(3)
    if "<warmup/>" in prompt or "SMALLTALK" in prompt:
        answer = "SKIP"
    else:
        if "USE_TOOL" in prompt:
            stream_text_message("Let me search the vault.", stop_reason="tool_use", tool="Grep")
            out({"type": "user", "message": {"role": "user", "content": [{"type": "tool_result"}]}})
        m = re.search(r"<questions>\n(.*?)\n", prompt)
        q = m.group(1) if m else prompt[:40]
        answer = f"Answer: {q}\n- point **one** [1]\n- point two"
    stream_text_message(answer)
    out({"type": "result", "subtype": "success", "is_error": False, "result": answer,
         "duration_ms": 5, "session_id": SESSION})


def main():
    args = sys.argv[1:]
    persistent = "--input-format" in args and args[args.index("--input-format") + 1] == "stream-json"
    if persistent:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            content = msg["message"]["content"]
            if isinstance(content, list):
                content = "".join(c.get("text", "") for c in content)
            reply(content)
    else:
        reply(sys.stdin.read())


if __name__ == "__main__":
    main()
