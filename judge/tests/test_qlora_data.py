import json

import pytest

from judge.build_sft_dataset import JUDGE_SYSTEM_PROMPT
from judge.qlora_data import (
    _encode_and_mask,
    _pad_batch,
    load_jsonl,
    load_prompt_response_pairs,
    normalize_record,
)


def fake_apply_chat_template(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"<{m['role']}>{m['content']}" for m in messages) + "\n<assistant>"


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_normalize_record_messages_schema():
    record = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Clause:\nfoo\n\nRetrieved GDPR Article 5:\nbar"},
            {"role": "assistant", "content": '{"compliance_status": "compliant"}'},
        ]
    }
    prompt, response = normalize_record(record, fake_apply_chat_template)
    assert "system prompt" in prompt
    assert "Clause:\nfoo" in prompt
    assert prompt.endswith("<assistant>")
    assert response == '{"compliance_status": "compliant"}'


def test_normalize_record_instruction_output_schema():
    record = {
        "instruction": "Clause:\nfoo\n\nRetrieved GDPR Article 5:\nbar",
        "output": {"compliance_status": "compliant"},
    }
    prompt, response = normalize_record(record, fake_apply_chat_template)
    assert JUDGE_SYSTEM_PROMPT in prompt
    assert "Clause:\nfoo" in prompt
    # non-string output is serialized to JSON text
    assert json.loads(response) == {"compliance_status": "compliant"}


def test_normalize_record_instruction_input_output_schema():
    record = {
        "instruction": "Judge this clause against the retrieved article.",
        "input": "Clause:\nfoo\n\nArticle:\nbar",
        "output": '{"compliance_status": "partial"}',
    }
    prompt, response = normalize_record(record, fake_apply_chat_template)
    assert "Judge this clause" in prompt
    assert "Clause:\nfoo" in prompt
    assert response == '{"compliance_status": "partial"}'


def test_normalize_record_unrecognized_schema_raises():
    with pytest.raises(ValueError):
        normalize_record({"foo": "bar"}, fake_apply_chat_template)


def test_load_prompt_response_pairs(tmp_path):
    path = tmp_path / "data.jsonl"
    records = [
        {"instruction": "do the thing", "output": '{"compliance_status": "compliant"}'},
        {"instruction": "do another thing", "output": '{"compliance_status": "partial"}'},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    pairs = load_prompt_response_pairs(path, fake_apply_chat_template)
    assert len(pairs) == 2
    assert set(pairs[0]) == {"prompt", "response"}
    assert "do the thing" in pairs[0]["prompt"]


def test_encode_and_mask_masks_prompt_tokens():
    prompt_ids = [1, 2, 3]
    response_ids = [4, 5]
    result = _encode_and_mask(prompt_ids, response_ids, eos_id=99, max_length=100)
    assert result["input_ids"] == [1, 2, 3, 4, 5, 99]
    assert result["labels"] == [-100, -100, -100, 4, 5, 99]


def test_encode_and_mask_no_eos():
    result = _encode_and_mask([1, 2], [3], eos_id=None, max_length=100)
    assert result["input_ids"] == [1, 2, 3]
    assert result["labels"] == [-100, -100, 3]


def test_encode_and_mask_truncates_to_max_length():
    result = _encode_and_mask([1, 2, 3], [4, 5, 6], eos_id=99, max_length=4)
    assert len(result["input_ids"]) == 4
    assert len(result["labels"]) == 4
    assert result["input_ids"] == [1, 2, 3, 4]
    assert result["labels"] == [-100, -100, -100, 4]


def test_pad_batch_pads_to_max_length_and_masks_padding():
    encoded = [
        {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]},
        {"input_ids": [4, 5], "labels": [-100, 5]},
    ]
    batch = _pad_batch(encoded, pad_id=0)
    assert batch["input_ids"] == [[1, 2, 3], [4, 5, 0]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"] == [[-100, -100, 3], [-100, 5, -100]]
