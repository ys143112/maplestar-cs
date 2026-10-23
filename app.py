"""실행: streamlit run app.py --server.port 8503"""
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
try:
    for name in ('GEMINI_API_KEY', 'GEMINI_MODEL','TICKET_TRANSPORT','TICKET_EMAIL_TO',
                 'TICKET_EMAIL_FROM','TICKET_SMTP_HOST','TICKET_SMTP_PORT','TICKET_SMTP_USER',
                 'TICKET_SMTP_PASSWORD','TICKET_WEBHOOK_URL','TICKET_WEBHOOK_TOKEN'):
        if name in st.secrets:
            os.environ.setdefault(name, str(st.secrets[name]))
except FileNotFoundError:
    pass
from rag import build_graph, cited_sources
from llm import error_message
from ticket_ui import render_intake, render_inbox
import chroma_store


@st.cache_resource(show_spinner=False)
def prepare_search_index(index_signature):
    """임시 디스크가 초기화되는 Cloud에서도 현재 자료의 색인을 한 번 준비한다."""
    from data_loader import item_documents, chunks
    expected={'items':len(item_documents()),'documents':len(chunks())}
    if chroma_store.counts()!=expected:
        chroma_store.index_documents()
    chroma_store.require_index()
    return expected

st.set_page_config(page_title='메이플스타 CS',page_icon='🍁',layout='centered',initial_sidebar_state='collapsed')
st.html('''<style>
[data-testid="stMainBlockContainer"] {
    max-width: 1000px;
    padding: 5.5rem clamp(1rem, 4vw, 2rem) calc(16rem + env(safe-area-inset-bottom, 0px));
}
.st-key-intro h3 {
    font-size: clamp(1.2rem, 2.5vw, 1.65rem);
    font-weight: 700;
    line-height: 1.5;
    margin: 0;
    padding: 0 0 0.5rem;
    word-break: keep-all;
    overflow-wrap: anywhere;
}
.st-key-intro p {line-height: 1.7; word-break: keep-all; overflow-wrap: anywhere;}
.st-key-examples button {width: 100%; min-height: 3rem;}
.st-key-examples button p {font-size: 0.9rem; word-break: keep-all;}
.st-key-chat_composer {
    position: fixed;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: calc(100% - clamp(2rem, 8vw, 4rem));
    max-width: 936px;
    padding: 0.75rem 0 calc(0.75rem + env(safe-area-inset-bottom, 0px));
    box-sizing: border-box;
    border-top: 1px solid rgba(128, 128, 128, 0.15);
    background: var(--background-color, white);
    z-index: 100;
}
.st-key-chat_composer [data-testid="stHorizontalBlock"] {flex-wrap: nowrap; align-items: center;}
.st-key-chat_composer [data-testid="stColumn"] {min-width: 0 !important;}
.st-key-chat_composer [data-testid="stColumn"]:first-child {flex: 1 1 0 !important; width: 0 !important;}
.st-key-chat_composer [data-testid="stColumn"]:last-child {flex: 0 0 4.75rem !important; width: 4.75rem !important;}
.st-key-chat_composer [data-testid="stButton"] button {
    min-height: 44px;
    width: 100%;
    white-space: nowrap;
    padding: 0.35rem 0.5rem;
}
.st-key-chat_composer [data-testid="stButton"] button p {font-size: 0.85rem;}
.st-key-chat_composer textarea {max-height: 12rem;}
[data-testid="stChatMessage"] {
    width: calc(100% - 2rem);
    max-width: 88%;
    border-radius: 1rem;
    padding: 1rem 1.25rem;
    border: 1px solid rgba(128, 128, 128, 0.15);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    margin-left: auto;
    margin-right: 0.75rem;
    align-self: flex-end;
    flex-direction: row-reverse;
    background: rgba(59, 130, 246, 0.09);
}
[data-testid="stChatMessageAvatarUser"] {display: none;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    margin-left: 0.75rem;
    margin-right: auto;
    align-self: flex-start;
    flex-direction: row;
    background: var(--secondary-background-color, rgba(128, 128, 128, 0.06));
}
[data-testid="stChatMessageContent"] {min-width: 0; overflow-wrap: anywhere; line-height: 1.7;}
[data-testid="stChatMessageContent"] pre {max-width: 100%; overflow-x: auto;}
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] {padding-top: 5rem;}
    .st-key-chat_composer [data-testid="stChatInput"] {min-height: 4.5rem;}
    .st-key-chat_composer textarea {min-height: 3.5rem; font-size: 1rem;}
    [data-testid="stChatMessage"] {width: 96%; max-width: 96%; padding: 0.75rem; gap: 0.5rem;}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {margin-right: 0.35rem;}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {margin-left: 0.35rem;}
}
</style>''')


def clear_chat():
    st.session_state.messages=[]
    st.session_state.conversation=[]
    st.session_state.pop('pending_inquiry',None)
    st.session_state.pop('separate_inquiry',None)
    st.session_state.pop('continue_current',None)
    reset_ticket_draft()


def reset_ticket_draft():
    for key in list(st.session_state):
        if key.startswith('ticket_') and key!='ticket_demo_inbox':
            del st.session_state[key]


def select_example(inquiry):
    reset_ticket_draft()
    st.session_state.pending_inquiry=inquiry


def submit_inquiry():
    st.session_state.pending_inquiry=st.session_state.inquiry


def resolve_separate(start_new):
    inquiry=st.session_state.separate_inquiry
    if start_new:
        clear_chat()
    else:
        # 선택 안내와 그 입력만 제거하고 기존 문의 맥락은 보존한다.
        st.session_state.messages=st.session_state.messages[:-2]
        st.session_state.pop('separate_inquiry',None)
    st.session_state.continue_current=True
    st.session_state.pending_inquiry=inquiry


def render_message(message):
    with st.chat_message(message['role']):
        if message.get('error'):
            st.error(message['content'])
        else:
            content=message['content']
            if message['role']=='assistant':
                content=re.sub(r'[ \t]*\[(?:출처|근거)\s*\d+(?:\s*[,·~\-]\s*\d+)*\]', '', content)
            st.markdown(content)
        result=message.get('result')
        if result:
            used=cited_sources(result['draft'],result['sources'])
            if used:
                source=next((s for _,s in used if s.get('source_type')!='item_metadata'),used[0][1])
                label='관련 운영정책 보기' if source.get('source_id')=='policy' else '관련 공식 안내 보기'
                st.caption(f"[{label}]({source['url']})")


st.session_state.setdefault('messages',[])
st.session_state.setdefault('conversation',[])

# 스타픽시 제출용: 문의를 기준으로 대화를 이어가고 공식 문서 묶음을 검색한다.
item_id='policy'
source_mode='bundle'
as_of=datetime.now(ZoneInfo('Asia/Seoul')).date()
st.session_state.selected_item_id=item_id
if st.session_state.get('conversation_as_of') != as_of.isoformat():
    st.session_state.conversation=[]
    st.session_state.conversation_as_of=as_of.isoformat()
    if st.session_state.messages:
        st.info('날짜가 바뀌어 이전 문의 조건을 초기화했습니다. 새 문의로 시작해 주세요.')
if not os.getenv('GEMINI_API_KEY','').strip():
    st.info('현재 응대 서비스를 준비 중입니다. 관리자에게 연결 설정을 확인해 주세요.')
inquiry=st.session_state.get('pending_inquiry')
busy=bool(inquiry)
choosing=bool(st.session_state.get('separate_inquiry'))
intro_slot=st.empty()
if not st.session_state.messages and not busy:
    with intro_slot.container(), st.container(key='intro'):
        st.subheader('고객의 환불·아이템 복구 문의를 입력해주세요.',anchor=False)
        st.write('필요한 조건을 확인하고, 문의 정리부터 접수까지 도와드립니다.')
        st.caption('공식 접수 연동을 가정한 모의 접수 데모입니다.')
        st.caption('예시 문의를 선택하시면 바로 답변을 받아보실 수 있습니다.')
        with st.container(key='examples'):
            examples=[
                ('미수령 교환권 환불', '메이플포인트 교환권을 3일 전에 구매했고, 우편함에서 아직 수령하지 않았어요. 환불 기준과 신청 방법을 알려주세요.'),
                ('상점에 판 아이템 복구', '어제 NPC 상점에 아이템을 실수로 팔았고 재구매하지 않았어요. 복구 기준과 신청 방법을 알려주세요.'),
                ('강화 실패 아이템 복구', '정상적으로 강화를 시도하다 실패해서 아이템이 사라졌어요. 이런 경우에도 복구를 신청할 수 있나요?'),
            ]
            for column,(label,question) in zip(st.columns(3,gap='small'),examples):
                with column:
                    st.button(label,on_click=select_example,args=(question,),
                              disabled=not bool(os.getenv('GEMINI_API_KEY','').strip()))
else:
    intro_slot.empty()
for message in st.session_state.messages:
    render_message(message)
if choosing:
    new,keep=st.columns(2)
    with new:
        st.button('새 문의로 시작',on_click=resolve_separate,args=(True,))
    with keep:
        st.button('기존 문의에 이어서 작성',on_click=resolve_separate,args=(False,))
intake_slot=st.empty()
if st.session_state.messages and not busy and not choosing:
    with intake_slot.container():
        try:
            render_intake(st.session_state.messages)
        except ValueError as exc:
            st.error(str(exc))
else:
    intake_slot.empty()
if not busy and not choosing:
    render_inbox(st.session_state.get('ticket_demo_inbox',[]))

# 입력창과 새 문의 버튼을 함께 화면 아래에 고정한다.
submitted=bool(st.session_state.get('ticket_receipt'))
submission_locked=submitted or bool(st.session_state.get('ticket_dispatch_error'))
with st.container(key='chat_composer'):
    entry, action = st.columns([8,1],gap='small',vertical_alignment='center')
    with entry:
        placeholder=('접수가 완료되었습니다. 새 문의를 시작해 주세요' if submitted else
                     '답변을 작성하고 있습니다…' if busy else '고객의 환불·아이템 복구 문의를 입력해 주세요')
        st.chat_input(placeholder,
                      key='inquiry',max_chars=4000,on_submit=submit_inquiry,
                      disabled=busy or choosing or submission_locked or not bool(os.getenv('GEMINI_API_KEY','').strip()))
    with action:
        st.button('새 문의',on_click=clear_chat,type='secondary',disabled=busy,key='new_inquiry')
if inquiry:
    user_message={'role':'user','content':inquiry,'inquiry_flow':'unresolved'}
    st.session_state.messages.append(user_message)
    render_message(user_message)
    try:
        with st.spinner('문의 내용을 확인하고 답변을 작성하고 있습니다…'):
            with st.spinner('첫 실행 시 검색 자료를 준비합니다. 잠시만 기다려 주세요…'):
                prepare_search_index(chroma_store.signature())
            result={'inquiry':inquiry,'as_of':as_of.isoformat(),'source_mode':source_mode,
                    'history':st.session_state.conversation,'item_id':item_id,
                    'continue_current':st.session_state.get('continue_current',False),
                    'customer_statement':st.session_state.get('ticket_form_draft',{}).get('statements',
                        st.session_state.get('ticket_condition_basis',{}).get('statements',''))}
            for event in build_graph().stream(result,stream_mode='updates'):
                for node,update in event.items():
                    result.update(update)
        flow=result.get('inquiry_flow','followup')
        user_message['inquiry_flow']=flow
        answer={'role':'assistant','content':result['draft']}
        if flow=='new_inquiry':
            st.session_state.separate_inquiry=inquiry
        elif flow=='followup':
            # 추가 진술은 새 요약에 반영하되, 이미 입력한 식별자는 다시 요구하지 않는다.
            saved=st.session_state.get('ticket_form_draft',{})
            identity={key:saved[key] for key in ('uid','code','email') if key in saved}
            prior=saved.get('statements') or st.session_state.get('ticket_condition_basis',{}).get('statements')
            statements=(prior+'\n\n추가 설명·정정:\n'+inquiry if prior else
                '\n\n'.join(m['content'] for m in st.session_state.messages
                            if m['role']=='user' and m.get('inquiry_flow','followup')=='followup'))
            if len(statements)>20000:
                statements=statements[-19950:]+'\n(긴 문의의 앞부분 생략: 내용을 확인해 주세요.)'
            reset_ticket_draft()
            st.session_state.ticket_form_draft={**identity,'statements':statements,
                'summary':saved.get('summary') if saved.get('summary_edited') else result.get('summary',''),
                'summary_edited':saved.get('summary_edited',False)}
            st.session_state.ticket_condition_basis={'statements':statements,'analysis':result,'edited':False}
            answer['result']=result
            st.session_state.conversation=(st.session_state.conversation+[
                {'role':'user','content':inquiry},
                {'role':'assistant','content':result['draft']}])[-12:]
    except ValueError as exc:
        answer={'role':'assistant','content':str(exc),'error':True}
    except Exception as exc:
        answer={'role':'assistant','content':error_message(exc),'error':True}
    st.session_state.messages.append(answer)
    st.session_state.pop('pending_inquiry',None)
    st.session_state.pop('continue_current',None)
    st.rerun()
