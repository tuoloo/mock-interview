from mockinterview.document_loader import load_text


def test_plain_text(tmp_path):
    p = tmp_path / "resume.txt"
    p.write_text("我是前端工程师", encoding="utf-8")
    assert "前端工程师" in load_text(str(p))


def test_markdown(tmp_path):
    p = tmp_path / "resume.md"
    p.write_text("# 简历\n- React", encoding="utf-8")
    out = load_text(str(p))
    assert "React" in out


def test_unsupported_extension(tmp_path):
    p = tmp_path / "resume.xyz"
    p.write_text("x", encoding="utf-8")
    try:
        load_text(str(p))
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "unsupported" in str(e).lower()
