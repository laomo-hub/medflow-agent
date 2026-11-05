from dataclasses import dataclass, field


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    strategy: str = "fixed"  # "fixed" | "semantic" | "parent_child"
    parent_chunk_size: int = 2048


@dataclass
class RetrievalConfig:
    top_k: int = 20
    rerank_top_k: int = 5
    use_hyde: bool = False
    use_rerank: bool = True
    use_hybrid: bool = False
    similarity_threshold: float = 0.3


@dataclass
class GenerationConfig:
    model: str = "deepseek-chat"
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class RAGConfig:
    collection_name: str = "knowledge_docs"
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1024
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)