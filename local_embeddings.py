"""CPU ONNX 임베딩. 긴 청크는 토큰 창별 벡터를 평균해 뒷부분도 반영한다."""
from functools import lru_cache
import numpy as np
from fastembed import TextEmbedding
from config import EMBEDDING_MODEL, RUNTIME
from langchain_core.embeddings import Embeddings


@lru_cache(maxsize=1)
def model():
    return TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=str(RUNTIME / 'models'),
                         threads=2, providers=['CPUExecutionProvider'])


def windows(text, tokenizer):
    encoded = tokenizer.encode(text)
    spans = []
    for part in [encoded, *encoded.overflowing]:
        offsets = [(start, end) for start, end in part.offsets if end > start]
        if offsets:
            spans.append(text[min(start for start, _ in offsets):max(end for _, end in offsets)])
    return spans or [text]


def embed(texts):
    if not texts:
        return []
    encoder = model()
    groups = [windows(text, encoder.model.tokenizer) for text in texts]
    vectors = iter(encoder.embed([part for group in groups for part in group], batch_size=16))
    result = []
    for group in groups:
        vector = np.mean([next(vectors) for _ in group], axis=0)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        result.append(vector.tolist())
    return result


class LocalEmbeddings(Embeddings):
    """LangChain 컬렉션 검색에서도 같은 로컬 벡터를 사용한다."""
    def embed_documents(self,texts):
        return embed(texts)

    def embed_query(self,text):
        return embed([text])[0]
