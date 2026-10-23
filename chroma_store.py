"""항목과 공식 본문을 별도 로컬 Chroma 컬렉션에 보관한다."""
import hashlib
import json
from functools import lru_cache
import chromadb
from chromadb.config import Settings
import local_embeddings
from config import VECTOR_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from data_loader import corpus_digest, item_documents, chunks


@lru_cache(maxsize=1)
def client():
    return chromadb.PersistentClient(path=str(VECTOR_PATH), settings=Settings(anonymized_telemetry=False))


def signature():
    value = [corpus_digest(), EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, 'token-window-mean-v1',
             [doc.page_content for doc in item_documents()]]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()[:16]


def collections():
    suffix = signature()
    return tuple(client().get_or_create_collection(
        name=f'cs_{kind}_{suffix}', embedding_function=None,
        metadata={'hnsw:space': 'cosine', 'embedding_model': EMBEDDING_MODEL})
        for kind in ('items', 'documents'))


def counts():
    return dict(zip(('items', 'documents'), (collection.count() for collection in collections())))


def require_index():
    expected={'items':len(item_documents()),'documents':len(chunks())}
    if counts()!=expected:
        raise ValueError('검색 자료를 준비 중입니다. 관리자에게 검색 색인 준비를 요청해 주세요.')


def index_documents(progress=lambda value: None):
    groups = (item_documents(), chunks())
    total = sum(map(len, groups))
    done = 0
    for collection, documents in zip(collections(), groups):
        existing = set(collection.get(include=[])['ids'])
        for start in range(0, len(documents), 32):
            batch = documents[start:start+32]
            missing = [doc for doc in batch if doc.metadata['chunk_id'] not in existing]
            if missing:
                collection.upsert(ids=[doc.metadata['chunk_id'] for doc in missing],
                    documents=[doc.page_content for doc in missing],
                    metadatas=[doc.metadata for doc in missing],
                    embeddings=local_embeddings.embed([doc.page_content for doc in missing]))
            done += len(batch)
            progress(done / max(total, 1))
    return counts()


def search(query, *, kind='documents', top_k=6, where=None):
    if kind not in ('items', 'documents'):
        raise ValueError('검색 컬렉션이 올바르지 않습니다.')
    if top_k < 1 or not query.strip():
        return []
    collection = collections()[0 if kind == 'items' else 1]
    return search_collection(collection, query, top_k=top_k, where=where)


def search_collection(collection, query, *, top_k=6, where=None):
    if top_k < 1 or not query.strip():
        return []
    if not collection.count():
        return []
    result = collection.query(query_embeddings=local_embeddings.embed([query]),
        n_results=min(top_k, collection.count()), where=where,
        include=['documents', 'metadatas', 'distances'])
    return [{**metadata, 'text': text, 'semantic_score': round(1-float(distance), 6)}
            for metadata, text, distance in zip(result['metadatas'][0], result['documents'][0], result['distances'][0])]
