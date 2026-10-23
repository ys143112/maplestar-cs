"""UID 등 계정 정보는 LLM 요청과 분리하고 최종 동의 후에만 전달한다."""
import json
import streamlit as st
from rag import cited_sources
from ticket_service import receive_demo, make_ticket, new_ticket_id
from intake_extraction import extract_statement
from llm import error_message
from inquiry_rules import FACT_LABELS, REQUIRED_FACTS
from ticket_service import FIELD_LABELS


def render_inbox(receipts):
    """접수 payload를 그대로 읽는 담당자용 보기. 추가 모델 호출은 하지 않는다."""
    if not receipts:
        return
    by_id={receipt['ticket']['ticket_id']:receipt for receipt in reversed(receipts)}
    if st.session_state.get('inbox_selected_ticket') not in by_id:
        st.session_state.inbox_selected_ticket=next(iter(by_id))
    with st.expander('이번 세션의 모의 접수함 · 담당자 검토'):
        st.caption('현재 세션의 데모 내역입니다. 실제 GM 접수함이 아니며 세션 종료 시 사라질 수 있습니다.')
        selected=st.selectbox('확인할 문의',list(by_id),
            format_func=lambda key:by_id[key]['ticket']['summary']+' · '+key[-6:],key='inbox_selected_ticket')
        packet=by_id[selected]['ticket']
        st.markdown('**'+packet['summary']+'**')
        st.caption(f"{packet['ticket_id']} · {packet['category']} / {packet.get('subtype','기타')}")
        st.caption('모의 접수 완료 · 담당자 검토 전')

        st.markdown('**먼저 확인할 사항**')
        missing=list(dict.fromkeys(packet.get('unconfirmed_items',[])))
        if packet.get('policy_review_required'):
            st.warning('문의 본문이 수정되어 정책 적용을 다시 확인해야 합니다. 이전 답변의 정책 근거는 승계하지 않았습니다.')
        if missing:
            st.write('미확인 조건: '+', '.join(missing))
        else:
            st.caption('자동 추출에서 표시한 미확인 조건은 없습니다. 실제 기록과 정책 적용 확인은 별도로 필요합니다.')
        reviews=list(dict.fromkeys(packet.get('ai_review_items',[])))
        for review in reviews:
            st.write('• '+review)
        if not reviews:
            st.caption('추가로 추출된 검토 사항이 없습니다.')

        st.markdown('**고객 진술에서 정리한 조건**')
        st.caption('게임 기록으로 확인한 사실이 아닙니다. 고객이 말한 내용과 모름·미제공을 구분합니다.')
        facts=packet.get('structured_conditions',{})
        rows=[]
        # 상세 상태와 중복되지 않는 대상·요청은 기존 요약 필드에서 함께 보여준다.
        for key in ('item','requested_action'):
            value=packet.get('conditions',{}).get(key,'미확인')
            rows.append({'조건':FIELD_LABELS[key],'내용':value,'구분':'자동 정리 · 원문 대조 필요'})
        if facts:
            required=REQUIRED_FACTS.get(packet.get('scenario','other'),())
            for key,label in FACT_LABELS.items():
                fact=facts.get(key,{})
                status=fact.get('status','not_provided')
                if key not in required and status=='not_provided':
                    continue
                value=fact.get('value','missing')
                rows.append({'조건':label,'내용':{'yes':'예','no':'아니요','unknown':'모름','missing':'미제공'}.get(value,value),
                    '구분':{'customer_reported':'고객 진술 · 원문 대조 필요','unknown':'미확인',
                            'not_provided':'미제공'}.get(status,'원문 대조 필요')})
        else:
            for key in ('occurred_at','incident','received_used','repurchase_trade'):
                rows.append({'조건':FIELD_LABELS[key],'내용':packet.get('conditions',{}).get(key,'미확인'),
                             '구분':'이전 접수 형식 · 원문 대조 필요'})
        st.table(rows)

        st.markdown('**최종 문의 원문**')
        st.text(packet['customer_statements'])
        source='수정한 접수 본문' if packet.get('extraction_source')=='edited_form' else '문의 대화'
        st.caption('조건 추출 기준: '+source+' · 고객 진술이며 기록 검증 전입니다.')

        st.markdown('**관련 정책**')
        policies=packet.get('reference_policies',[])
        for policy in policies:
            st.write(policy.get('title') or '공식 자료')
            url=policy.get('url','')
            if url.startswith('https://'):
                st.link_button('원문 보기',url)
            st.caption('게시: '+(policy.get('published_at') or '미상')+' · 수정: '+(policy.get('updated_at') or '미상')
                       +' · 현재 적용 여부는 담당자 확인 필요')
        if not policies:
            st.caption('연결된 정책 근거가 없습니다. 실제 처리 판단 전에 관련 정책을 확인해 주세요.')

        if st.toggle('계정·연락처 보기',key='inbox_identity_'+selected):
            account=packet.get('account',{})
            st.text('월드 계정 UID: '+account.get('world_uid',''))
            st.text('프로필 코드: '+account.get('world_profile_code',''))
            st.text('답변받을 이메일: '+(packet.get('reply_email') or '미입력'))
            st.caption('계정 존재·본인 소유·이메일 소유는 검증하지 않았습니다.')
        if st.toggle('원본 접수 데이터 보기',key='inbox_json_'+selected):
            st.json(packet,expanded=False)
        st.download_button('담당자 전달 파일 내려받기',json.dumps(packet,ensure_ascii=False,indent=2),
            file_name=selected+'.json',mime='application/json',key='ticket_inbox_'+selected)


def save_intake_field(field,key):
    draft=st.session_state.setdefault('ticket_form_draft',{})
    draft[field]=st.session_state[key]
    if field=='summary':
        draft['summary_edited']=True
    st.session_state.ticket_consent=False


def reset_intake_consent():
    st.session_state.ticket_consent=False


def render_intake(messages):
    latest=next((m for m in reversed(messages) if m.get('result') and m['role']=='assistant'),None)
    if not latest or messages[-1].get('error'):
        return
    result=latest['result']
    st.session_state.setdefault('ticket_id',new_ticket_id())
    # 이전 화면의 확인 단계는 더 이상 사용하지 않는다.
    st.session_state.pop('ticket_preview',None)
    with st.expander('문의 확인 · 모의 접수',expanded=not bool(result.get('clarification_question'))):
        if st.session_state.get('ticket_receipt'):
            receipt=st.session_state.ticket_receipt
            st.success('모의 접수 완료 · 담당자 검토 대기')
            st.code(receipt['ticket']['ticket_id'],language=None)
            st.write('문의 내용과 확인한 조건이 모의 접수함에 저장되었습니다.')
            st.caption('공식 접수 연동을 가정한 데모입니다. 실제 접수·계정 조회·환불·복구 처리는 하지 않습니다.')
            st.caption('다른 문의는 아래 ‘새 문의’로 시작해 주세요. 담당자용 접수 내용은 모의 접수함에서 확인하실 수 있습니다.')
            return
        if st.session_state.get('ticket_dispatch_error'):
            st.warning('접수 결과를 확인하지 못했습니다. 중복 접수를 막기 위해 재접수를 잠갔습니다. 아래 번호로 운영자에게 확인해 주세요.')
            st.code(st.session_state.ticket_id,language=None)
            return
        statements='\n\n'.join(m['content'] for m in messages if m['role']=='user'
                                and m.get('inquiry_flow','followup')=='followup')
        if len(statements)>20000:
            statements=statements[-19950:]+'\n(긴 문의의 앞부분 생략: 내용을 확인해 주세요.)'
        draft=st.session_state.setdefault('ticket_form_draft',
            {'summary':result.get('summary',''),'statements':statements})
        basis=st.session_state.get('ticket_condition_basis',
            {'statements':statements,'analysis':result,'edited':False})
        st.caption('자동으로 정리한 내용을 확인해 주세요. 필요한 부분만 수정하실 수 있습니다.')
        if st.toggle('문의 내용 수정',key='ticket_editing',on_change=reset_intake_consent):
            for field,label in (('summary','문의 제목'),('statements','문의 내용')):
                key='ticket_input_'+field
                st.session_state.setdefault(key,draft.get(field,''))
                if field=='summary':
                    st.text_input(label,key=key,max_chars=1000,on_change=save_intake_field,args=(field,key))
                else:
                    st.text_area(label,key=key,max_chars=20000,height=150,on_change=save_intake_field,args=(field,key))
        else:
            st.markdown('**문의 제목**')
            st.text(draft.get('summary',''))
            st.markdown('**문의 내용**')
            st.text(draft.get('statements',''))
        st.caption('접수할 계정 정보를 입력해 주세요. 데모에서는 가상 정보를 사용해 주세요.')
        for field,label,placeholder,limit in (
            ('uid','메이플스토리 월드 계정 UID','월드 클라이언트 → 설정 → 계정에서 확인',64),
            ('code','메이플스토리 월드 프로필 코드','#으로 시작하는 프로필 코드',64),
            ('email','답변받을 이메일 (선택)','',254)):
            key='ticket_input_'+field
            st.session_state.setdefault(key,draft.get(field,''))
            st.text_input(label,key=key,placeholder=placeholder,max_chars=limit,
                          on_change=save_intake_field,args=(field,key))
        st.caption('프로필 코드: 월드 클라이언트 → 더보기 → 프로필 편집에서 확인하실 수 있습니다.')
        st.caption('공식 접수 연동을 가정한 모의 접수입니다. 비밀번호·인증번호·카드 전체 번호는 입력하지 마세요.')
        consent=st.checkbox('위 내용과 계정 정보를 확인했으며, 현재 세션의 모의 접수함에 저장하는 데 동의합니다.',key='ticket_consent')
        if not st.button('모의 접수하기',disabled=not consent,type='primary'):
            return
        try:
            summary=draft.get('summary','')
            statements=draft.get('statements','')
            uid=draft.get('uid',''); code=draft.get('code',''); email=draft.get('email','')
            # 계정 입력 오류는 모델 호출 전에 안내한다.
            make_ticket(ticket_id=st.session_state.ticket_id,category='기타',subtype='기타',
                summary=summary,statements=statements,fields={},uid=uid,profile_code=code,
                reply_email=email,missing=[],policies=[])
            if statements.strip()!=basis['statements'].strip():
                with st.spinner('수정한 문의 내용을 정리하는 중…'):
                    analysis=extract_statement(statements)
                basis={'statements':statements,'analysis':analysis,'edited':True}
                st.session_state.ticket_condition_basis=basis
            analysis=basis['analysis']; edited=basis['edited']
            policies=[]
            if not edited:
                for _,source in cited_sources(result['draft']+'\n'+result.get('policy_details',''),result.get('sources',[])):
                    policy={key:source.get(key,'') for key in ('source_id','title','url','published_at','updated_at','effective_status')}
                    if policy not in policies:
                        policies.append(policy)
            review=analysis.get('missing_fields',[])+analysis.get('review_items',[])
            if edited:
                review=review+['문의 본문 수정 후 정책 적용·최신성을 담당자가 다시 확인해야 합니다.']
            packet=make_ticket(ticket_id=st.session_state.ticket_id,category=analysis.get('category','기타'),subtype='기타',
                summary=summary,statements=statements,fields=analysis.get('intake_fields',{}),uid=uid,profile_code=code,
                reply_email=email,missing=review,policies=policies,
                scenario=analysis.get('scenario','other'),case_facts=analysis.get('case_facts',{}),
                extraction_source='edited_form' if edited else 'conversation',statements_edited=edited,
                fact_evidence=analysis.get('fact_evidence',{}))
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(error_message(exc))
            return
        # 접수 시작 후에는 재실행으로 중복 저장하지 않는다.
        st.session_state.ticket_dispatch_error=True
        try:
            receipt=receive_demo(packet)
            st.session_state.ticket_receipt=receipt
            inbox=st.session_state.setdefault('ticket_demo_inbox',[])
            inbox.append(receipt)
            st.session_state.ticket_demo_inbox=inbox[-10:]
            st.session_state.pop('ticket_dispatch_error',None)
        except Exception:
            st.rerun()
        st.rerun()
