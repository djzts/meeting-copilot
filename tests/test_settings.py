import argparse
import json
import threading
import tomllib
import urllib.error
import urllib.request

import pytest

import copilot
import settings
from settings import SettingsError, update_toml_text, validate_changes

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_validate_coerces_types():
    out = validate_changes({
        "claude": {"auto_answer": False, "context_minutes": "2.5", "model": "  sonnet "},
        "vault": {"top_k": 4.0, "exclude": "NDA/**\n\n Private/** \n"},
        "stt": {"compute_type": "float16"},
    })
    assert out == {
        "claude": {"auto_answer": False, "context_minutes": 2.5, "model": "sonnet"},
        "vault": {"top_k": 4, "exclude": ["NDA/**", "Private/**"]},
        "stt": {"compute_type": "float16"},
    }


@pytest.mark.parametrize("changes,field", [
    ({"claude": {"auto_answer": "yes"}}, "claude.auto_answer"),
    ({"vault": {"top_k": 0}}, "vault.top_k"),
    ({"vault": {"top_k": 2.5}}, "vault.top_k"),
    ({"audio": {"silence_threshold": "loud"}}, "audio.silence_threshold"),
    ({"audio": {"end_silence_sec": float("nan")}}, "audio.end_silence_sec"),
    ({"stt": {"device": "tpu"}}, "stt.device"),
    ({"claude": {"extra_args": ["--dangerously-skip-permissions"]}}, "claude.extra_args"),
    ({"secrets": {"x": 1}}, "secrets.x"),
])
def test_validate_rejects_bad_values(changes, field):
    with pytest.raises(SettingsError) as e:
        validate_changes(changes)
    assert field in e.value.errors


def test_validate_reports_every_bad_field():
    with pytest.raises(SettingsError) as e:
        validate_changes({"vault": {"top_k": -1, "per_note": 99}, "claude": {"thinking": 1}})
    assert set(e.value.errors) == {"vault.top_k", "vault.per_note", "claude.thinking"}


# --------------------------------------------------------------------------- #
# Writing config.toml
# --------------------------------------------------------------------------- #
SAMPLE = '''# my config
[audio]
device = "Stereo Mix"   # the Realtek card
silence_threshold = 0.01

[vault]
path = "C:/notes"   # where my notes live
exclude = [
  "NDA/**",   # contracts
  "Private/**",
]

[claude]
about_me = "I use C# # and F#"   # tricky: '#' inside the string
model = "haiku"
'''


def test_update_keeps_comments_and_other_lines():
    out = update_toml_text(SAMPLE, {"audio": {"device": "loopback"}, "claude": {"model": "sonnet"}})
    assert 'device = "loopback"     # the Realtek card' in out  # comment stays in its column
    assert 'model = "sonnet"' in out
    old_lines, new_lines = SAMPLE.splitlines(), out.splitlines()
    assert len(old_lines) == len(new_lines)
    changed = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    assert [old_lines[i].split("=")[0].strip() for i in changed] == ["device", "model"]
    assert tomllib.loads(out)["audio"]["device"] == "loopback"


def test_update_string_with_hash_and_multiline_array():
    out = update_toml_text(SAMPLE, {"claude": {"about_me": 'Lead of "data" team #1'},
                                    "vault": {"exclude": ["Secret/**"]}})
    data = tomllib.loads(out)
    assert data["claude"]["about_me"] == 'Lead of "data" team #1'
    assert data["vault"]["exclude"] == ["Secret/**"]
    assert "# tricky: '#' inside the string" in out
    assert "contracts" not in out  # the old multi-line array was replaced as a whole
    assert data["vault"]["path"] == "C:/notes"


def test_update_adds_missing_key_and_section():
    out = update_toml_text(SAMPLE, {"audio": {"preroll_sec": 0.5}, "stt": {"language": "zh"}})
    data = tomllib.loads(out)
    assert data["audio"]["preroll_sec"] == 0.5
    assert data["stt"]["language"] == "zh"
    lines = out.splitlines()
    # the new audio key sits inside [audio], before the blank line and [vault]
    assert lines.index("preroll_sec = 0.5") < lines.index("[vault]")


def test_update_values_round_trip():
    vals = {"t": True, "f": False, "i": 7, "x": 0.8, "y": 20.0, "s": 'C:\\Users\\me "x"\n中文', "l": ["a", "b"]}
    out = update_toml_text("", {"sec": vals})
    assert tomllib.loads(out)["sec"] == vals


def test_update_keeps_crlf_line_endings():
    out = update_toml_text(SAMPLE.replace("\n", "\r\n"), {"claude": {"model": "opus"}})
    assert "\r\n" in out and "\n" not in out.replace("\r\n", "")
    assert tomllib.loads(out.replace("\r\n", "\n"))["claude"]["model"] == "opus"


def test_save_creates_from_template(tmp_path):
    template = tmp_path / "config.example.toml"
    template.write_text(SAMPLE, encoding="utf-8")
    target = tmp_path / "config.toml"
    settings.save_changes(target, {"claude": {"model": "sonnet"}}, template=template)
    assert "# my config" in target.read_text(encoding="utf-8")
    assert tomllib.loads(target.read_text(encoding="utf-8"))["claude"]["model"] == "sonnet"
    assert template.read_text(encoding="utf-8") == SAMPLE  # the template itself is untouched


# --------------------------------------------------------------------------- #
# Applying to a running App
# --------------------------------------------------------------------------- #
@pytest.fixture
def app(tmp_path, vault_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[audio]\ndevice = "loopback"\n\n'
        f'[vault]\npath = "{vault_path.as_posix()}"\nexclude = ["NDA/**"]\nembedding_model = ""\n\n'
        '[claude]\nmodel = "haiku"   # fastest\n',
        encoding="utf-8")
    cfg, path = copilot.load_config(str(cfg_file))
    args = argparse.Namespace(dry_run=False, simulate=None, port=None, open=False, verbose=False)
    a = copilot.App(cfg, args, config_path=path)
    a.restarts = []
    a.engine.restart = lambda: a.restarts.append(dict(model=a.claude_cfg.model, vault=str(a.engine.vault.root)))
    a.events = []
    a.hub.publish = lambda ev, data: a.events.append((ev, data))
    a.cfg_file = cfg_file
    return a


def test_live_changes_need_no_restart(app):
    r = app.apply_settings({"claude": {"auto_answer": False, "about_me": "Tech lead", "context_minutes": 5},
                            "vault": {"top_k": 3, "per_note": 1}}, save=False)
    assert r["effects"] == ["live"] and r["saved"] is None
    assert app.engine.cfg.auto_answer is False and app.engine.cfg.about_me == "Tech lead"
    assert app.engine.cfg.context_minutes == 5
    assert (app.vault.top_k, app.vault.per_note) == (3, 1)
    assert app.restarts == []
    assert ("info", {"auto_answer": False, "model": "haiku"}) in app.events
    assert set(r["unsaved"]) == {"claude.auto_answer", "claude.about_me", "claude.context_minutes",
                                 "vault.top_k", "vault.per_note"}


def test_unchanged_values_are_ignored(app):
    r = app.apply_settings({"claude": {"model": "haiku"}, "vault": {"exclude": ["NDA/**"]}})
    assert r["changed"] == [] and app.restarts == [] and r["saved"] is None


def test_claude_model_restarts_claude_and_is_saved(app):
    r = app.apply_settings({"claude": {"model": "sonnet", "effort": "medium"}})
    assert r["effects"] == ["claude"]
    assert app.restarts == [{"model": "sonnet", "vault": app.restarts[0]["vault"]}]
    text = app.cfg_file.read_text(encoding="utf-8")
    assert 'model = "sonnet"  # fastest' in text  # comment kept, same column
    assert tomllib.loads(text)["claude"]["effort"] == "medium"
    assert app.settings_payload()["values"]["claude"]["model"] == "sonnet"


def test_unsaved_changes_are_written_on_the_next_save(app):
    app.apply_settings({"claude": {"about_me": "A"}}, save=False)
    assert "about_me" not in app.cfg_file.read_text(encoding="utf-8")
    r = app.apply_settings({"vault": {"top_k": 2}}, save=True)
    data = tomllib.loads(app.cfg_file.read_text(encoding="utf-8"))
    assert data["claude"]["about_me"] == "A" and data["vault"]["top_k"] == 2
    assert r["unsaved"] == []


def test_vault_change_rebuilds_index_and_restarts_claude(app, vault_path, tmp_path):
    old = app.vault
    other = tmp_path / "notes"
    other.mkdir()
    (other / "a.md").write_text("# Alpha\nThe alpha project ships in May.", encoding="utf-8")
    r = app.apply_settings({"vault": {"path": "notes"}})  # relative to the config file
    assert r["effects"] == ["index"]
    assert app.vault is not old and app.engine.vault is app.vault
    assert str(app.vault.root) == str(other.resolve())
    assert app.restarts and app.restarts[-1]["vault"] == str(other.resolve())
    assert app.settings_payload()["values"]["vault"]["path"] == "notes"  # shown as typed
    assert tomllib.loads(app.cfg_file.read_text(encoding="utf-8"))["vault"]["path"] == "notes"
    assert app.vault_name == "notes"


def test_stt_settings(app):
    r = app.apply_settings({"stt": {"language": "zh", "initial_prompt": "Apollo"}}, save=False)
    assert r["effects"] == ["next"] and not app._stt_reload.is_set()
    assert app.stt_cfg.language == "zh" and app.stt_cfg.initial_prompt == "Apollo"
    r = app.apply_settings({"stt": {"model": "small", "device": "cpu"}}, save=False)
    assert r["effects"] == ["stt"] and app._stt_reload.is_set()


def test_vad_settings_reach_the_pipeline(app):
    seen = []
    app.pipeline = type("P", (), {"update_vad": lambda self, cfg: seen.append(cfg)})()
    app.apply_settings({"audio": {"end_silence_sec": 1.2, "silence_threshold": 0.02}}, save=False)
    assert seen and seen[0].end_silence_sec == 1.2 and seen[0].silence_threshold == 0.02


def test_invalid_change_applies_nothing(app):
    with pytest.raises(SettingsError):
        app.apply_settings({"claude": {"model": "sonnet"}, "vault": {"top_k": 0}})
    assert app.claude_cfg.model == "haiku" and app.restarts == []
    assert "sonnet" not in app.cfg_file.read_text(encoding="utf-8")


def test_example_config_is_never_overwritten(app):
    app.config_path = copilot.HERE / "config.example.toml"
    assert app._save_target() == copilot.HERE / "config.toml"


def test_payload_lists_fields_values_and_devices(app, monkeypatch):
    monkeypatch.setattr(copilot.audio, "device_options", lambda names=None: ["loopback", "Stereo Mix"])
    p = app.settings_payload()
    ids = {f"{f['section']}.{f['key']}" for f in p["fields"]}
    assert {"audio.device", "claude.model", "vault.path", "stt.language"} <= ids
    assert p["values"]["claude"]["effort"] == "low"  # the real default, not blank
    assert p["values"]["audio"]["device"] == "loopback"
    assert p["device_options"] == ["loopback", "Stereo Mix"]
    assert all(f["effect_label"] for f in p["fields"])


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def test_http_config_endpoints(app, monkeypatch):
    monkeypatch.setattr(copilot.audio, "device_options", lambda names=None: ["loopback"])
    srv = copilot.Server(("127.0.0.1", 0), copilot.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        got = json.load(urllib.request.urlopen(base + "/config"))
        assert got["values"]["claude"]["model"] == "haiku"

        def post(body, ctype="application/json"):
            req = urllib.request.Request(base + "/config", data=json.dumps(body).encode(),
                                         headers={"Content-Type": ctype}, method="POST")
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, json.load(r)
            except urllib.error.HTTPError as e:
                return e.code, json.load(e)

        code, body = post({"changes": {"claude": {"auto_answer": False}}, "save": False})
        assert code == 200 and body["changed"] == ["claude.auto_answer"]
        code, body = post({"changes": {"vault": {"top_k": "many"}}})
        assert code == 400 and "vault.top_k" in body["errors"]
        code, _ = post({"changes": {"claude": {"auto_answer": True}}}, ctype="text/plain")
        assert code == 415  # no simple cross-site form posts
        assert app.claude_cfg.auto_answer is False
    finally:
        srv.shutdown()
        srv.server_close()
