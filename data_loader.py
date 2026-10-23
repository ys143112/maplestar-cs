"""수집 완료 자료만 읽는다. 이미지 검토 대기·가상 문의는 근거에 넣지 않는다."""
import hashlib
import json
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DATA, CHUNK_SIZE, CHUNK_OVERLAP


@lru_cache(maxsize=1)
def load_documents():
    docs = json.loads(DATA.read_text(encoding='utf-8'))
    if len({d['id'] for d in docs}) != len(docs) or any(d['quality']!='ready' for d in docs):
        raise ValueError('중복되거나 검토 대기 중인 문서가 있습니다.')
    return docs


def corpus_digest():
    return hashlib.sha256(DATA.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_items():
    payload = json.loads((DATA.parent / 'items.json').read_text(encoding='utf-8'))
    items = payload['items']
    if payload['documents_sha256'] != corpus_digest():
        raise ValueError('원문이 변경되었습니다. scripts/build_items.py로 항목 목록을 갱신해 주세요.')
    source_ids = {doc['id'] for doc in load_documents()}
    if len({item['item_id'] for item in items}) != len(items) or {item['source_id'] for item in items} != source_ids:
        raise ValueError('항목 목록과 원문 연결을 확인해 주세요.')
    return items


def item_documents():
    """F2001: 항목당 속성 텍스트 한 건. 정책 본문이나 추측한 요약을 넣지 않는다."""
    result = []
    for item in load_items():
        text = '\n'.join((f"자료명: {item['name']}", f"자료 유형: {item['kind']}",
                          f"수집 분류 태그: {', '.join(item['topics']) or '미분류'}",
                          f"게시일: {item['published_at'] or '미표시'}",
                          f"수정일: {item['updated_at'] or '미표시'}",
                          f"적용 상태: {item['effective_status']}",
                          f"자료 적용 범위: {item.get('scope_note', '메이플스타 공식 자료')}"))
        result.append(Document(page_content=text, metadata=dict(
            chunk_id=f"meta:{item['item_id']}", item_id=item['item_id'],
            source_id=item['source_id'], title=item['name'], url=item['source_url'],
            source_type='item_metadata', source_scope=item.get('source_scope', 'maplestar'), section=item['section'],
            published_at=item['published_at'] or '', updated_at=item['updated_at'] or '')))
    return result


@lru_cache(maxsize=1)
def chunks():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    result = []
    for source in load_documents():
        published = source.get('published_at') or ''
        available = published or source.get('first_verified_at') or ''
        for i, text in enumerate(splitter.split_text(source['text'])):
            metadata = dict(chunk_id=f"{source['id']}:{i}", item_id=source['id'],
                            source_type='external_document', source_scope=source.get('source_scope', 'maplestar'),
                            scope_note=source.get('scope_note', '메이플스타 공식 자료'),
                            source_id=source['id'], title=source['title'],
                            url=source['source_url'], section=source['section'], published_at=published,
                            published_day=int(datetime.fromisoformat(available.replace('Z','+00:00')).astimezone(
                                ZoneInfo('Asia/Seoul')).strftime('%Y%m%d')) if available else 0,
                            updated_at=source.get('updated_at') or '', collected_at=source['collected_at'],
                            effective_status=source.get('effective_status','unknown'))
            result.append(Document(page_content=text, metadata=metadata))
    return result
