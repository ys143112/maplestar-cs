"""과제 필수 항목 탐색/선택 및 실행 증빙을 유저 화면과 분리한다."""
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from data_loader import load_items
from item_search import discover
from llm import error_message
import chroma_store


def choose_item(item_id):
    st.session_state.review_item=item_id


def sidebar(has_key):
    today=datetime.now(ZoneInfo('Asia/Seoul')).date()
    busy=bool(st.session_state.get('pending_inquiry'))
    with st.sidebar:
        mode=st.toggle('과제 검토 모드',key='review_mode',disabled=busy)
        if not mode:
            return False,'policy',today,'bundle'
        st.caption('공식 자료를 탐색·선택하면 자료별 대화와 전체 인용 근거를 확인할 수 있습니다.')
        items=load_items()
        item_map={item['item_id']:item for item in items}
        st.session_state.setdefault('review_item','policy')
        as_of=st.date_input('문의 기준일',value=today,key='review_as_of',disabled=busy)
        provider=st.selectbox('문서 수집 방식',['live','bundle'],
            format_func=lambda value:'공식 사이트 검색·선별·본문 수집' if value=='live' else '공식 문서 묶음 (외부 호출 제한 시)',disabled=busy)
        with st.expander('검색 색인 준비·설정',expanded=False):
            counts=chroma_store.counts()
            st.write(f"메타데이터 {counts['items']}건 · 본문 {counts['documents']}조각")
            st.caption('초기 환경에서는 먼저 색인을 준비해 주세요. 키 없이 CPU로 실행합니다.')
            if st.button('검색 색인 준비',key='review_build_index',disabled=busy):
                progress=st.progress(0.0)
                result=chroma_store.index_documents(progress.progress)
                st.success(f"색인 준비 완료: {result['items']}항목 / {result['documents']}본문")
        with st.form('review_item_search'):
            condition=st.text_input('원하는 자료의 조건',placeholder='예: 아이템 복구 신청 기한을 다룬 정책')
            submitted=st.form_submit_button('자료 찾기',disabled=busy or not has_key)
        if submitted:
            try:
                with st.spinner('조건에 맞는 공식 자료를 찾는 중…'):
                    st.session_state.item_matches=discover(condition,as_of.isoformat())
                if not st.session_state.item_matches:
                    st.info('조건을 확인할 수 있는 자료가 없습니다.')
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(error_message(exc))
        for item in st.session_state.get('item_matches',[]):
            st.button(item['name'],key='choose_'+item['item_id'],on_click=choose_item,args=(item['item_id'],),disabled=busy)
        selected=st.selectbox('질문할 공식 자료',list(item_map),key='review_item',
                              format_func=lambda mid:item_map[mid]['name'],disabled=busy)
        st.caption('자료를 바꿔도 항목별 대화는 현재 세션에 보관됩니다.')
        return True,selected,as_of,provider


def evidence(result):
    from rag import cited_sources
    draft=result.get('draft','')+'\n'+result.get('policy_details','')
    used=cited_sources(draft,result.get('sources',[]))
    if not used:
        st.info('이 답변에 인용된 공식 근거가 없습니다. 조건 확인 질문 또는 확인 불가 안내입니다.')
    else:
        with st.expander(f'실제 인용 근거 {len(used)}개',expanded=True):
            for number,source in used:
                kind='항목 메타데이터' if source.get('source_type')=='item_metadata' else '외부 문서'
                st.markdown(f"**[근거 {number}] {source['title']}**")
                st.caption(kind+' · '+source.get('source_scope','maplestar'))
                st.write(source['text'])
                st.link_button('원문: '+source['title'],source['url'])
    with st.expander('검색·수집 중간 결과',expanded=False):
        st.write('검색어',result.get('search_queries',[]))
        trace=result.get('collection_trace',{})
        st.write('수집 방식',trace.get('provider','확인 질문: 수집 생략'))
        st.write('선택 원문',trace.get('selected_source_ids',[]))
        st.write('제외 원문',trace.get('excluded_source_ids',[]))
        st.write('캐시 적중',trace.get('cache_hit',False))
        st.write('최종 근거 수',len(result.get('sources',[])))
        for source in result.get('candidate_sources',[]):
            st.caption(source.get('chunk_id','')+' · '+source.get('title',''))
