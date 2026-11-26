import json

REQUIRED_FIELDS = ["id", "document_type", "locator", "url", "source_version", "verified_on", "text"]
DOCUMENT_TYPES = ["CTB", "MBFT"]

def load_passages(file_path: str) -> list[dict]:
    passages: list[dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            json_line =json.loads(line)
            field_missing = []
            for field in REQUIRED_FIELDS:
                if field not in json_line.keys():
                    field_missing.append(field)
            if field_missing:
                raise ValueError(f"Missing required fields {field_missing} in passage: {json_line}")
            if json_line["document_type"] not in DOCUMENT_TYPES:
                raise ValueError(f"Invalid document type in passage: {json_line}")
            if json_line["id"] not in [p["id"] for p in passages]:
                passages.append(json_line)
            else:
                raise ValueError(f"Duplicate passage ID found: {json_line['id']}")
    for passage in passages:
        if passage["document_type"] == "MBFT":
            if passage["ctb_passage_id"] not in [p["id"] for p in passages if p["document_type"] == "CTB"]:
                raise ValueError(f"MBFT passage references non-existent CTB passage: {passage['ctb_passage_id']}")
    return passages