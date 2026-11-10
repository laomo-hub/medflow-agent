from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path

from src.core.config import get_settings

BATCH_SIZE = 20
GROUP_SIZE = 5000
COLLECTION_NAME = "knowledge_docs"
EVAL_FILE = Path(__file__).resolve().parents[3] / "data" / "eval" / "medqa_zh.json"


class ExistingDataError(Exception):
    """The target already contains data from this source."""


def _load_dataset(dataset: str):
    from datasets import load_dataset
    if dataset == "cmirb":
        return load_dataset("CMIRB/MedicalRetrieval", "corpus", split="corpus")
    if dataset == "dialogue":
        return load_dataset("BillGPT/Chinese-medical-dialogue-data")
    if dataset == "medqa":
        return load_dataset("fzkuji/MedQA", "med_qa_zh_source")
    raise ValueError("Unknown dataset")


def _texts(dataset: str, data) -> list[str]:
    texts = []
    if dataset == "cmirb":
        for row in data:
            value = row.get("text", "") or row.get("content", "")
            if value and len(value.strip()) > 20:
                texts.append(value.strip()[:2000])
    else:
        for rows in data.values():
            for row in rows:
                value = row.get("input", "") or ""
                if not value:
                    q = row.get("ask", "") or row.get("question", "") or ""
                    a = row.get("answer", "") or ""
                    if len(q) > 5 and len(a) > 10:
                        value = f"问：{q.strip()}\n答：{a.strip()}"
                if value and value.strip():
                    texts.append(value.strip()[:2000])
    return texts


def _medqa_records(data) -> list[dict]:
    records = []
    for split_name, rows in data.items():
        for row in rows:
            question = row.get("question", "")
            if question:
                records.append({"question": question, "choices": row.get("options", row.get("choices", {})),
                                "answer": row.get("answer_idx", "") or row.get("answer", ""),
                                "split": split_name})
    return records


def _save_medqa(records: list[dict], target: Path | None = None) -> None:
    target = target or EVAL_FILE
    # A hard link creates the final path atomically and fails if it already exists.
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".medqa-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(records, out, ensure_ascii=False, indent=2)
        try:
            os.link(temp_name, target)
        except FileExistsError as exc:
            raise ExistingDataError("MedQA output already exists") from exc
    finally:
        os.unlink(temp_name)


def _make_deps():
    from langchain_community.embeddings import DashScopeEmbeddings
    from pymilvus import MilvusClient
    from src.agents.knowledge.doc_ingestion import ensure_knowledge_collection
    settings = get_settings()
    if not settings.DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY is required")
    client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    ensure_knowledge_collection(client)
    model = DashScopeEmbeddings(model=settings.EMBEDDING_MODEL, dashscope_api_key=settings.DASHSCOPE_API_KEY)
    return client, model


def _doc_id(dataset: str, offset: int) -> str:
    source = "cmirb" if dataset == "cmirb" else "med_dialogue"
    return hashlib.md5(f"{source}_{offset}".encode()).hexdigest()[:16]


def _check_absent(client, doc_ids: list[str]) -> None:
    for doc_id in doc_ids:
        found = client.query(collection_name=COLLECTION_NAME, filter=f'doc_id == "{doc_id}"',
                             output_fields=["doc_id"], limit=1, consistency_level="Strong")
        if found:
            raise ExistingDataError("Data for this dataset already exists; no overwrite performed")


async def import_dataset(dataset: str, report, check_owner) -> dict:
    if dataset == "medqa" and EVAL_FILE.exists():
        raise ExistingDataError("MedQA output already exists")
    await report("downloading", 0, 0)
    data = await asyncio.to_thread(_load_dataset, dataset)
    await check_owner()
    if dataset == "medqa":
        records = await asyncio.to_thread(_medqa_records, data)
        if not records:
            raise ValueError("No usable MedQA records")
        await report("saving", 0, len(records))
        await check_owner()
        await asyncio.to_thread(_save_medqa, records)
        await report("saving", len(records), len(records))
        return {"records": len(records), "destination": "data/eval/medqa_zh.json"}

    texts = await asyncio.to_thread(_texts, dataset, data)
    if not texts:
        raise ValueError("No usable dataset records")
    await report("checking", 0, len(texts))
    await check_owner()
    client, embedding = await asyncio.to_thread(_make_deps)
    try:
        offsets = list(range(0, len(texts), GROUP_SIZE))
        await asyncio.to_thread(_check_absent, client, [_doc_id(dataset, i) for i in offsets])
        done = 0
        for offset in offsets:
            doc_id = _doc_id(dataset, offset)
            # Recheck the group before the first insertion; never delete existing data.
            await check_owner()
            await asyncio.to_thread(_check_absent, client, [doc_id])
            group = texts[offset:offset + GROUP_SIZE]
            label = "CMIRB医学检索语料" if dataset == "cmirb" else "中文医患对话"
            category = "医学文献" if dataset == "cmirb" else "医患对话"
            for pos in range(0, len(group), BATCH_SIZE):
                await check_owner()
                batch = group[pos:pos + BATCH_SIZE]
                vectors = await embedding.aembed_documents(batch)
                if len(vectors) != len(batch):
                    raise ValueError("Embedding count does not match input")
                rows = [{"id": f"{doc_id}_{pos + j}", "doc_id": doc_id,
                         "doc_name": f"{label}_{offset // GROUP_SIZE + 1}",
                         "doc_type": "literature", "category": category, "page_number": 0,
                         "chunk_index": pos + j, "text": value[:65000], "embedding": vectors[j]}
                        for j, value in enumerate(batch)]
                await check_owner()
                result = await asyncio.to_thread(client.insert, collection_name=COLLECTION_NAME, data=rows)
                inserted = result.get("insert_count", 0)
                if not isinstance(inserted, int) or not 0 <= inserted <= len(rows):
                    raise ValueError("Invalid Milvus insert count")
                done += inserted
                await report("inserting", done, len(texts))
                if inserted != len(rows):
                    raise ValueError("Incomplete Milvus insert")
        return {"records": done, "destination": COLLECTION_NAME}
    finally:
        await asyncio.to_thread(client.close)