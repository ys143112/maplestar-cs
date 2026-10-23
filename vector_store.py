"""Chroma 의미 검색과 단어 점수를 결합해 공식 문서를 찾는다."""
import math
import re
from datetime import date
from config import TOP_K
from data_loader import chunks, item_documents
import chroma_store


def index_count():
    return chroma_store.counts()['documents']


def index_documents(progress=lambda value:None):
    return chroma_store.index_documents(progress)['documents']


def retrieve(query, as_of, top_k=TOP_K):
    if top_k < 1:
        return []
    counts = chroma_store.counts()
    if counts != {'items': len(item_documents()), 'documents': len(chunks())}:
        raise ValueError('검색 색인이 준비되지 않았습니다. 과제 검토 모드 → 검색 색인 준비 또는 '
                         'python scripts/build_vector_index.py를 실행해 주세요.')
    day = int(date.fromisoformat(as_of).strftime('%Y%m%d'))
    available = [(i,d) for i,d in enumerate(chunks()) if d.metadata['published_day'] <= day]
    if not available:
        return []
    # 고객 표현을 정책의 용어로 연결한다. 날짜·처리 가능 여부 같은 답은 추가하지 않는다.
    if re.search(r'잃어버|사라[졌진지]|분실|유실|손실|복구', query):
        query += ' 유실 손실 복구'
        if re.search(r'기한|언제까지|며칠|기간', query):
            query += ' 발생 시점 이내 경과 조사 요청'
    if re.search(r'비밀번호|인증번호|OTP', query, re.IGNORECASE):
        query += ' 개인정보 계정정보'
    tokens = set(re.findall(r'[가-힣A-Za-z0-9]{2,}',query.casefold()))
    texts = {d.metadata['chunk_id']:(d.metadata['title']+' '+d.page_content).casefold() for _,d in available}
    frequency = {t:sum(t in text for text in texts.values()) for t in tokens}
    scores = {mid:sum(math.log(1+len(available)/(frequency[t] or 1)) for t in tokens if t in text)
              for mid,text in texts.items()}
    semantic = {row['chunk_id']: row['semantic_score'] for row in chroma_store.search(
        query, top_k=len(available), where={'published_day': {'$lte': day}})}
    maximum = max(scores.values(),default=0) or 1
    ranked = []
    for i,doc in available:
        mid = doc.metadata['chunk_id']
        similarity = max(0, semantic.get(mid, 0))
        score = 0.7*scores[mid]/maximum + 0.3*similarity
        if score > 0:
            ranked.append({**doc.metadata,'text':doc.page_content,'semantic_score':similarity,
                           'retrieval_score':round(score,6)})
    return sorted(ranked,key=lambda row:(-row['retrieval_score'],row['chunk_id']))[:top_k]
