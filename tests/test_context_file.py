from clm.context_file import parse_back, render_editable, was_edited


def _history():
    return [
        {"role": "assistant", "content": "Let me look.", "reasoning": "think",
         "tool_calls": [{"id": "c1", "name": "bash", "arguments": '{"command": "ls"}'}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nb.txt"},
        {"role": "user", "content": "next op"},
    ]


def test_render_has_headers_and_flattened_tool_call():
    text = render_editable(_history())
    assert "[[CTX_TURN 1 role=assistant]]" in text
    assert "[[CTX_TURN 2 role=tool]]" in text
    assert 'bash {"command": "ls"}' in text
    assert "<reasoning>" in text
    assert "<reasoning>" not in render_editable(_history(), include_reasoning=False)


def test_roundtrip_maps_roles_and_merges():
    out = parse_back(render_editable(_history()))
    # tool + user merge into one user message
    assert [m["role"] for m in out] == ["assistant", "user"]
    assert "a.txt" in out[1]["content"] and "next op" in out[1]["content"]


def test_invented_role_becomes_user_and_preamble_kept():
    text = "orphan note\n[[CTX_TURN 9 role=notes]]\nmy notes\n[[CTX_TURN 3 role=assistant]]\nok\n"
    out = parse_back(text)
    assert out == [
        {"role": "user", "content": "orphan note\n\nmy notes"},
        {"role": "assistant", "content": "ok"},
    ]


def test_empty_turns_dropped_and_empty_file():
    assert parse_back("[[CTX_TURN 1 role=assistant]]\n\n[[CTX_TURN 2 role=tool]]\n  \n") == []
    assert parse_back("") == []


def test_was_edited_ignores_trailing_whitespace():
    text = render_editable(_history())
    assert not was_edited(text, text + "\n\n")
    assert not was_edited(text, text.replace("\n", "\r\n"))
    assert was_edited(text, text.replace("a.txt", "X"))
    assert not was_edited(text, None)
