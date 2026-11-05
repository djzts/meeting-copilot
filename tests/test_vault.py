import re

from vault import (BM25, Vault, chunk_text, glob_to_regex, iter_markdown_files, parse_note, rrf_fuse,
                   split_sections, strip_frontmatter, tokenize)


# ---- tokenizer ------------------------------------------------------------- #
def test_tokenize_english_lowercase_and_stopwords():
    toks = tokenize("What is the Budget for the Migrations?")
    assert "budget" in toks
    assert "migration" in toks  # simple plural folding
    assert "the" not in toks and "what" not in toks and "is" not in toks


def test_tokenize_chinese_unigrams_and_bigrams():
    toks = tokenize("数据库迁移")
    for t in ["数", "据", "库", "迁", "移", "数据", "据库", "库迁", "迁移"]:
        assert t in toks
    # question words are not useful search terms
    assert "什么" not in tokenize("什么是回滚")
    assert "回滚" in tokenize("什么是回滚")


def test_tokenize_mixed():
    toks = tokenize("Apollo 迁移 timeline")
    assert {"apollo", "迁移", "timeline"} <= set(toks)


# ---- markdown parsing ------------------------------------------------------ #
def test_strip_frontmatter():
    text = "---\ntags: [a]\n---\n# Title\nbody"
    assert strip_frontmatter(text) == "# Title\nbody"
    assert strip_frontmatter("no frontmatter\n---\n") == "no frontmatter\n---\n"


def test_split_sections_ignores_hash_in_code_blocks():
    md = "intro\n# A\none\n```bash\n# not a heading\n```\n## B\ntwo"
    secs = split_sections(md)
    assert [h for h, _ in secs] == ["", "A", "B"]
    assert "# not a heading" in secs[1][1]


def test_chunk_text_respects_size():
    para = ("This is a sentence about the system. " * 20).strip()
    body = "\n\n".join([para] * 6)
    chunks = chunk_text(body, 1200)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") == body.replace("\n", "").replace(" ", "")


def test_parse_note_titles(vault):
    titles = {c.title for c in vault._snap.chunks}
    assert "Apollo Migration › Timeline" in titles
    assert "Apollo Migration › Budget" in titles
    assert "数据库迁移 › 回滚方案" in titles


def test_parse_note_drops_frontmatter():
    chunks = parse_note("---\nowner: X\n---\n# T\nhello", "n")
    assert all("owner" not in text for _, text in chunks)


# ---- discovery / exclusion ------------------------------------------------- #
def test_glob_to_regex():
    assert glob_to_regex("NDA/**").match("NDA/a.md")
    assert glob_to_regex("NDA/**").match("NDA/sub/b.md")
    assert not glob_to_regex("NDA/**").match("Projects/NDA.md")
    assert glob_to_regex("NDA").match("NDA/x.md")
    assert glob_to_regex("**/secret/**").match("a/b/secret/c.md")
    assert glob_to_regex("**/secret/**").match("secret/c.md")
    assert glob_to_regex("*.excalidraw.md").match("deep/dir/x.excalidraw.md")
    assert not glob_to_regex("Projects/*.md").match("Projects/sub/x.md")


def test_iter_skips_dot_dirs_and_excludes(vault_path):
    rels = [rel for _, rel in iter_markdown_files(vault_path, ["NDA/**"])]
    assert "Projects/Apollo Migration.md" in rels
    assert not any(r.startswith(".obsidian") for r in rels)
    assert not any(r.startswith("NDA/") for r in rels)


def test_excluded_content_never_searchable(vault):
    assert all(h.chunk.path != "NDA/Acme Contract.md" for h in vault.search("Acme secret pricing per seat"))
    assert all("hidden" not in h.chunk.path for h in vault.search("zebra unicorn"))


# ---- retrieval --------------------------------------------------------------- #
def test_search_english(vault):
    hits = vault.search("What's the timeline for the Apollo cut-over?")
    assert hits[0].chunk.title == "Apollo Migration › Timeline"


def test_search_budget(vault):
    hits = vault.search("how much budget is left")
    assert hits[0].chunk.title == "Apollo Migration › Budget"


def test_search_chinese(vault):
    hits = vault.search("如果切换失败怎么回滚？")
    assert hits[0].chunk.path == "中文笔记/数据库迁移.md"
    assert hits[0].chunk.heading == "回滚方案"


def test_search_latency_note(vault):
    hits = vault.search("search p95 latency SLO")
    assert hits[0].chunk.path == "Projects/Search Service.md"


def test_previous_utterance_does_not_drown_the_question(vault):
    hits = vault.search("Can you walk me through the main risks before the cut-over?",
                        context="How much budget do we have left for the migration project?")
    assert hits[0].chunk.title == "Apollo Migration › Risks"
    hits = vault.search("How much budget do we have left for the migration project?",
                        context="We had some disk issues on the staging cluster last week, but that is fixed now.")
    assert hits[0].chunk.title == "Apollo Migration › Budget"


def test_context_helps_vague_follow_up(vault):
    hits = vault.search("and how long does it take now?", context="The reconciliation job is slow on PostgreSQL")
    assert hits[0].chunk.title in ("Apollo Migration › Risks", "2026-10-01 Standup › Standup 2026-10-01")


def test_search_per_note_limit_and_top_k(vault):
    hits = vault.search("Apollo migration timeline risks budget reconciliation", top_k=6, per_note=2)
    paths = [h.chunk.path for h in hits]
    assert paths.count("Projects/Apollo Migration.md") <= 2
    assert len(hits) <= 6


def test_search_empty_query(vault):
    assert vault.search("   ") == []


def test_stats(vault):
    st = vault.stats()
    assert st["files"] == 5  # .obsidian and NDA are skipped
    assert st["chunks"] >= 8
    assert st["vectors"] is False


def test_bm25_and_rrf():
    bm = BM25([["apple", "pie"], ["apple"], ["banana"]])
    top = bm.top(["apple", "pie"], 3)
    assert top[0][0] == 0
    fused = rrf_fuse([[0, 1, 2], [2, 0]])
    assert fused[0][0] == 0  # in both lists, near the top


def test_missing_vault_path(tmp_path):
    v = Vault(str(tmp_path / "nope"), cache_dir=str(tmp_path))
    try:
        v.build()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
    assert v.search("anything") == []
