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
CMIRB_RESUME_REVISION = "3844e2255fbe12c43844927e1e4930b3af911667"
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


def _load_cmirb_snapshot():
    from datasets import load_dataset
    return load_dataset("CMIRB/MedicalRetrieval", "corpus", split="corpus", revision=CMIRB_RESUME_REVISION)


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


def _existing_rows(client, ids: list[str], expected: dict[str, tuple[str, int, str]]) -> set[str]:
    """Strong-read expected IDs; reject duplicates or any mismatched existing content."""
    if not ids:
        return set()
    rows = client.get(collection_name=COLLECTION_NAME, ids=ids,
                      output_fields=["id", "doc_id", "chunk_index", "text"],
                      consistency_level="Strong")
    found = set()
    for row in rows:
        ident = row["id"]
        if ident in found or ident not in expected:
            raise ExistingDataError("Duplicate or unexpected existing ID")
        doc_id, pos, text = expected[ident]
        if row.get("doc_id") != doc_id or row.get("chunk_index") != pos or row.get("text") != text:
            raise ExistingDataError("Existing record differs from pinned source")
        found.add(ident)
    return found


def _group_count(client, doc_id: str) -> int:
    result = client.query(collection_name=COLLECTION_NAME, filter=f'doc_id == "{doc_id}"',
                          output_fields=["count(*)"], consistency_level="Strong")
    return int(result[0]["count(*)"])


async def _embed_with_retry(embedding, batch, check_owner):
    for attempt in range(3):
        await check_owner()
        try:
            return await embedding.aembed_documents(batch)
        except (ConnectionError, TimeoutError, OSError):
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)


async def _resume_cmirb(client, embedding, texts, report, check_owner) -> dict:
    expected = {}
    groups = []
    for offset in range(0, len(texts), GROUP_SIZE):
        doc_id = _doc_id("cmirb", offset)
        group = texts[offset:offset + GROUP_SIZE]
        ids = []
        for pos, value in enumerate(group):
            ident = f"{doc_id}_{pos}"
            expected[ident] = (doc_id, pos, value[:65000])
            ids.append(ident)
        groups.append((offset, doc_id, ids))

    present = set()
    # Validate every group before any write, including unknown or duplicated IDs.
    for offset, doc_id, ids in groups:
        group_found = set()
        for start in range(0, len(ids), 500):
            await check_owner()
            found = await asyncio.to_thread(_existing_rows, client, ids[start:start + 500], expected)
            group_found.update(found)
        await check_owner()
        count = await asyncio.to_thread(_group_count, client, doc_id)
        if count != len(group_found):
            raise ExistingDataError("Unexpected or duplicate records in source group")
        present.update(group_found)
        await report("checking", min(offset + len(ids), len(texts)), len(texts))

    skipped = len(present)
    inserted_total = 0
    await report("inserting", skipped, len(texts))
    for offset, doc_id, ids in groups:
        for pos in range(0, len(ids), BATCH_SIZE):
            await check_owner()
            candidates = [ident for ident in ids[pos:pos + BATCH_SIZE] if ident not in present]
            if not candidates:
                continue
            # A second read detects matching or conflicting writes made during preflight.
            found = await asyncio.to_thread(_existing_rows, client, candidates, expected)
            present.update(found)
            skipped += len(found)
            candidates = [ident for ident in candidates if ident not in found]
            if not candidates:
                await report("inserting", skipped + inserted_total, len(texts))
                continue
            vectors = await _embed_with_retry(embedding, [expected[ident][2] for ident in candidates], check_owner)
            if len(vectors) != len(candidates):
                raise ValueError("Embedding count does not match input")
            await check_owner()
            # Recheck immediately before insertion; external writers are not locked by this API.
            found = await asyncio.to_thread(_existing_rows, client, candidates, expected)
            present.update(found)
            skipped += len(found)
            rows = []
            for ident, vector in zip(candidates, vectors):
                if ident in found:
                    continue
                _, index, value = expected[ident]
                rows.append({"id": ident, "doc_id": doc_id, "doc_name": f"CMIRB医学检索语料_{offset // GROUP_SIZE + 1}",
                             "doc_type": "literature", "category": "医学文献", "page_number": 0,
                             "chunk_index": index, "text": value, "embedding": vector})
            if rows:
                await check_owner()
                result = await asyncio.to_thread(client.insert, collection_name=COLLECTION_NAME, data=rows)
                count = result.get("insert_count", 0)
                if not isinstance(count, int) or not 0 <= count <= len(rows):
                    raise ValueError("Invalid Milvus insert count")
                inserted_total += count
                await report("inserting", skipped + inserted_total, len(texts))
                # Never automatically retry an ambiguous write; a new resume rechecks IDs.
                if count != len(rows):
                    raise ValueError("Incomplete Milvus insert")
            else:
                await report("inserting", skipped + inserted_total, len(texts))
    return {"records": skipped + inserted_total, "inserted": inserted_total,
            "skipped": skipped, "destination": COLLECTION_NAME, "revision": CMIRB_RESUME_REVISION}


async def import_dataset(dataset: str, report, check_owner, resume: bool = False, expected_embedding_model: str | None = None) -> dict:
    if dataset == "medqa" and EVAL_FILE.exists():
        raise ExistingDataError("MedQA output already exists")
    await report("downloading", 0, 0)
    if resume and (dataset != "cmirb" or not expected_embedding_model):
        raise ValueError("CMIRB resume requires expected_embedding_model")
    if resume and get_settings().EMBEDDING_MODEL != expected_embedding_model:
        raise ValueError("Configured embedding model does not match expected model")
    data = await asyncio.to_thread(_load_cmirb_snapshot if resume else _load_dataset, *(() if resume else (dataset,)))
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
        if resume:
            return await _resume_cmirb(client, embedding, texts, report, check_owner)
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