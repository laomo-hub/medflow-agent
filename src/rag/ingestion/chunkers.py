from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import DashScopeEmbeddings
from src.rag.config import ChunkingConfig


@dataclass
class Chunk:
    text: str
    metadata: dict


class FixedChunker:
    def __init__(self, config: ChunkingConfig):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " "],
        )

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        docs = self.splitter.create_documents([text])
        return [
            Chunk(text=doc.page_content, metadata={**(metadata or {}), "chunk_index": i})
            for i, doc in enumerate(docs)
        ]


class SemanticChunkerWrapper:
    def __init__(self, embedding_model: DashScopeEmbeddings, breakpoint_threshold: float = 0.3):
        self.chunker = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold,
        )

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        docs = self.chunker.create_documents([text])
        return [
            Chunk(text=doc.page_content, metadata={**(metadata or {}), "chunk_index": i})
            for i, doc in enumerate(docs)
        ]


class ParentChildChunker:
    def __init__(self, config: ChunkingConfig):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.parent_chunk_size, chunk_overlap=128, separators=["\n\n", "\n"],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " "],
        )

    def chunk(self, text: str, metadata: dict = None) -> list[Chunk]:
        parent_docs = self.parent_splitter.create_documents([text])
        chunks = []
        for pi, parent in enumerate(parent_docs):
            child_docs = self.child_splitter.create_documents([parent.page_content])
            for ci, child in enumerate(child_docs):
                chunks.append(Chunk(
                    text=child.page_content,
                    metadata={**(metadata or {}), "parent_index": pi,
                              "parent_text": parent.page_content, "chunk_index": len(chunks)},
                ))
        return chunks


def get_chunker(config: ChunkingConfig, embedding_model: DashScopeEmbeddings = None):
    if config.strategy == "semantic":
        return SemanticChunkerWrapper(embedding_model)
    elif config.strategy == "parent_child":
        return ParentChildChunker(config)
    else:
        return FixedChunker(config)