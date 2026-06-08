import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

EMBEDDING_MODEL      = "all-MiniLM-L12-v2"
COLLECTION_NAME      = "eng_mental_health_chatbot"
LLM_MODEL            = "openai/gpt-oss-120b"
ANSWER_CHUNK_SIZE    = 500
ANSWER_CHUNK_OVERLAP = 50
VECTOR_DIM           = 384
CONTENT_PAYLOAD_KEY  = "text"

SYSTEM_PROMPT = """\
You are a compassionate mental health support assistant.

User language: {language} ({language_code}) — reply in this language only.
Detected emotion: {emotion_label}
Tone guidance: {emotion_tone}

Guidelines:
- Always acknowledge feelings before giving advice
- Keep response to 3-5 paragraphs
- Never diagnose or prescribe medication
- If the context below is not enough, say so and suggest professional help

Relevant knowledge:
{context}
"""


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{i+1}] {get_doc_text(doc)}" for i, doc in enumerate(docs)
    )


def get_doc_text(doc):
    """Return retrieved text across Qdrant/LangChain payload shapes."""
    text = getattr(doc, "page_content", "") or ""
    metadata = getattr(doc, "metadata", {}) or {}
    return text or metadata.get(CONTENT_PAYLOAD_KEY, "") or metadata.get("page_content", "")


class RAGPipeline:
    def __init__(self, qdrant_url, qdrant_api_key, groq_api_key, collection_name=COLLECTION_NAME):
        self.qdrant_url      = qdrant_url
        self.qdrant_api_key  = qdrant_api_key
        self.groq_api_key    = groq_api_key
        self.collection_name = collection_name

        self.embeddings   = None
        self.retriever    = None
        self.llm          = None

        self._build()

    def _build(self):
        self._load_embeddings()
        self._connect_qdrant()
        self._load_llm()
        logger.info("RAG pipeline ready.")

    def _load_embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name    = EMBEDDING_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
        logger.info(f"Embeddings loaded: {EMBEDDING_MODEL}")

    def _connect_qdrant(self):
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore
        client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        store  = QdrantVectorStore(
            client          = client,
            collection_name = self.collection_name,
            embedding       = self.embeddings,
            content_payload_key = CONTENT_PAYLOAD_KEY,
        )
        self.retriever = store.as_retriever(
            search_type   = "similarity",
            search_kwargs = {"k": 5},
        )
        logger.info("Qdrant connected.")

    def _load_llm(self):
        from langchain_groq import ChatGroq
        self.llm = ChatGroq(
            model       = LLM_MODEL,
            api_key     = self.groq_api_key,
            temperature = 0.7,
            max_tokens  = 1024,
        )
        logger.info(f"LLM loaded: {LLM_MODEL}")

    def ask(self, question, emotion_tone="Be warm.", emotion_label="neutral",
            language="English", language_code="en", show_sources=True):

        if not question or not question.strip():
            return {"answer": "Please share what's on your mind.", "sources": []}

        try:
            docs    = self.retriever.invoke(question)
            context = format_docs(docs)

            system_msg = SYSTEM_PROMPT.format(
                language      = language,
                language_code = language_code,
                emotion_label = emotion_label,
                emotion_tone  = emotion_tone,
                context       = context,
            )

            from langchain_core.messages import SystemMessage, HumanMessage
            response = self.llm.invoke([
                SystemMessage(content=system_msg),
                HumanMessage(content=question),
            ])

            sources = []
            if show_sources:
                sources = []
                for doc in docs:
                    text = get_doc_text(doc)
                    metadata = getattr(doc, "metadata", {}) or {}
                    sources.append({
                        "text":     text,
                        "snippet":  text[:200] + ("..." if len(text) > 200 else ""),
                        "question": metadata.get("question", "")[:100],
                    })

            return {"answer": response.content, "sources": sources}

        except Exception as e:
            logger.error(f"RAG error: {e}", exc_info=True)
            return {
                "answer":  "I'm having trouble connecting right now. Please try again in a moment.",
                "sources": [],
            }

    @classmethod
    def index_dataset(cls, qdrant_url, qdrant_api_key, collection_name=COLLECTION_NAME):
        """
        One-time indexing of the mental health counseling dataset into Qdrant.
        Chunking strategy: each chunk = full patient question + part of therapist answer.
        """
        from datasets import load_dataset
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        logger.info("Loading dataset...")
        ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
        df = ds.to_pandas().dropna(subset=["Context", "Response"]).reset_index(drop=True)
        df["Context"]  = df["Context"].str.strip()
        df["Response"] = df["Response"].str.strip()
        logger.info(f"{len(df):,} conversations loaded.")

        # split only the answer — keep the full question in every chunk
        answer_splitter = RecursiveCharacterTextSplitter(
            chunk_size    = ANSWER_CHUNK_SIZE,
            chunk_overlap = ANSWER_CHUNK_OVERLAP,
            separators    = ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        chunks = []
        for i, row in df.iterrows():
            question = row["Context"]
            for piece in answer_splitter.split_text(row["Response"]):
                chunks.append(Document(
                    page_content = f"Patient: {question}\nTherapist: {piece}",
                    metadata     = {
                        "source":   "mental_health_counseling",
                        "row_id":   i,
                        "question": question[:200],
                    },
                ))

        logger.info(f"{len(chunks):,} chunks created (question + answer part each).")

        embeddings = HuggingFaceEmbeddings(
            model_name    = EMBEDDING_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )

        client   = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        existing = [c.name for c in client.get_collections().collections]

        if collection_name not in existing:
            client.create_collection(
                collection_name = collection_name,
                vectors_config  = VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{collection_name}' created.")

        batch_size = 256
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            QdrantVectorStore.from_documents(
                documents       = batch,
                embedding       = embeddings,
                url             = qdrant_url,
                api_key         = qdrant_api_key,
                collection_name = collection_name,
                content_payload_key = CONTENT_PAYLOAD_KEY,
            )
            logger.info(f"  Uploaded {min(i + batch_size, len(chunks)):,}/{len(chunks):,}")

        logger.info("Indexing complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    from dotenv import load_dotenv

    load_dotenv()


    if len(sys.argv) > 1 and sys.argv[1] == "--index":
        RAGPipeline.index_dataset(
            os.getenv("qdrant_url", ""),
            os.getenv("qdrant_api_key", ""),
        )
    else:
        rag = RAGPipeline(
            os.getenv("qdrant_url", ""),
            os.getenv("qdrant_api_key", ""),
            os.getenv("groq_api_key", ""),
        )
        print(rag.ask("I feel very anxious and cannot sleep", show_sources=False)["answer"])
