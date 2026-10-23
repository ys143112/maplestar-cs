"""담당자 전달용 문의와 전송 경계. 기본값은 외부 전송 없는 세션 모의 접수."""
import json
import os
import re
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
from uuid import uuid4
from inquiry_rules import REQUIRED_FACTS, FACT_LABELS

FIELD_LABELS={
    'item':'상품·아이템', 'occurred_at':'구매·손실 시점',
    'incident':'결제·손실 경위', 'received_used':'수령·사용 상태',
    'repurchase_trade':'재구매·거래 여부', 'requested_action':'요청 사항',
}

SUBTYPES={
    'voucher_refund':'교환권 환불','point_goods_refund':'포인트 상품 환불','cash_refund':'카드·현금 반환 문의',
    'npc_sale':'상점 오판매','discard_delete':'삭제·버림','synthesis':'합성·분해',
    'normal_enhancement':'정상 강화 실패','enhancement_unknown':'강화 손실 원인 미확인',
    'completed_trade':'완료 거래','consumed_expired':'사용·소모·기간 만료','system_error':'시스템 오류 의심'}


def make_inquiry_letter(summary,statements,facts=None,evidence=None):
    """최종 고객 원문만 사용한다. 정책 판단·AI 추정·계정 양식은 넣지 않는다."""
    if not summary.strip() or not statements.strip():
        raise ValueError('문의 제목과 내용을 입력해 주세요.')
    from fact_grounding import ground_facts
    grounded,quotes,_=ground_facts(facts or {},evidence or {},statements)
    rows=[]
    for key,value in grounded.items():
        if key in quotes and value not in ('unknown','missing'):
            rows.append(f"- {FACT_LABELS[key]}: "+{'yes':'예','no':'아니요'}.get(value,value))
    conditions='[문의 조건]\n'+'\n'.join(rows)+'\n\n' if rows else ''
    body='안녕하세요. 아래 문의 내용의 확인을 부탁드립니다.\n\n'+conditions+'[문의 경위·요청]\n'+statements.strip()
    return {'title':summary.strip(),'body':body}


def destination():
    mode=os.getenv('TICKET_TRANSPORT','demo').strip().lower()
    if mode=='demo':
        return mode,'현재 브라우저 세션의 모의 접수함 (외부 전송 없음)'
    if mode=='email':
        recipient=os.getenv('TICKET_EMAIL_TO','').strip()
        required=('TICKET_SMTP_HOST','TICKET_SMTP_USER','TICKET_SMTP_PASSWORD','TICKET_EMAIL_FROM')
        if not recipient or any(not os.getenv(key,'').strip() for key in required):
            raise ValueError('운영자의 이메일 전송 설정이 완료되지 않았습니다.')
        if not re.fullmatch(r'[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+',recipient):
            raise ValueError('운영자의 수신 이메일 설정을 확인해 주세요.')
        return mode,recipient
    if mode=='webhook':
        endpoint=os.getenv('TICKET_WEBHOOK_URL','').strip()
        parsed=urlparse(endpoint)
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('접수 API는 인증정보·쿼리가 없는 HTTPS 주소로 설정해야 합니다.')
        if not os.getenv('TICKET_WEBHOOK_TOKEN','').strip():
            raise ValueError('운영자의 접수 API 인증 설정이 완료되지 않았습니다.')
        return mode,endpoint
    raise ValueError('지원하지 않는 접수 전송 설정입니다.')


def make_ticket(*, ticket_id, category, subtype, summary, statements, fields, uid, profile_code, reply_email, missing, policies,
                scenario='other', case_facts=None, extraction_source='conversation', statements_edited=False, fact_evidence=None):
    uid=uid.strip()
    profile_code=profile_code.strip()
    reply_email=reply_email.strip()
    # 공개된 공식 규격으로 단정하지 않는 최소 입력 검사. 존재/본인 확인이 아니다.
    if not uid or len(uid)>64 or not re.fullmatch(r'[0-9]+',uid):
        raise ValueError('월드 계정 UID는 숫자로 입력해 주세요.')
    if not re.fullmatch(r'#[^\s#]{1,63}',profile_code):
        raise ValueError('월드 프로필 코드는 #으로 시작하고 공백 없이 입력해 주세요.')
    if reply_email and not re.fullmatch(r'[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+',reply_email):
        raise ValueError('답변받을 이메일 형식을 확인해 주세요.')
    if not summary.strip() or not statements.strip():
        raise ValueError('문의 요약과 문의 내용을 확인해 주세요.')
    required=['item','occurred_at','incident']
    if category=='결제/환불' and subtype=='환불 요청':
        required.append('received_used')
    if category=='아이템/복구' and subtype in ('상점 오판매','삭제·버림'):
        required.append('repurchase_trade')
    facts=case_facts or {}
    unconfirmed=[FIELD_LABELS[key] for key in required
        if not str(fields.get(key,'')).strip() or re.search(r'모름|모르|미상|미확인|unknown|알 수 없|기억.*(?:안|않|없)',str(fields.get(key,'')))]
    if scenario!='other':
        unconfirmed=[FACT_LABELS[key] for key in REQUIRED_FACTS.get(scenario,())
                     if facts.get(key,'missing') in ('unknown','missing','')]
    return {
        'schema_version':2, 'ticket_id':ticket_id,
        'category':category, 'subtype':SUBTYPES.get(scenario,subtype), 'scenario':scenario,
        'summary':summary.strip(), 'customer_statements':statements.strip(),
        'conditions':{key:str(fields.get(key,'')).strip() or '미확인' for key in FIELD_LABELS},
        'ai_review_items':list(dict.fromkeys(missing)),
        'unconfirmed_items':list(dict.fromkeys(unconfirmed)),
        'structured_conditions':{key:{'label':label,'value':facts.get(key,'missing'),
            'status':'not_provided' if facts.get(key,'missing') in ('missing','') else
                     'unknown' if facts.get(key)=='unknown' else 'customer_reported'}
            for key,label in FACT_LABELS.items()},
        'extraction_source':extraction_source,
        'fact_evidence':fact_evidence or {},
        'statements_edited':statements_edited,
        'policy_review_required':statements_edited,
        'account':{'world_uid':uid,'world_profile_code':profile_code,
                   'existence_verified':False,'ownership_verified':False},
        'reply_email':reply_email or None,
        'reference_policies':policies,
        'review_note':('최종 문의 본문을 기준으로 조건을 자동 추출했습니다. 고객 진술이며 계정·결제·게임 기록으로 검증되지 않았습니다. '
                       '담당자는 원문과 추출 조건을 대조하고 정책 적용 및 실제 처리 여부를 확인해야 합니다.'
                       +(' 본문 수정 후 정책 적합성을 재검색하지 않았으므로 이전 답변의 정책 연결은 제외했습니다.' if statements_edited else '')),
    }


def new_ticket_id():
    return 'DEMO-'+uuid4().hex[:16].upper()


def receive_demo(ticket):
    """공식 접수 연동을 가정한 데모. 전송 설정과 무관하게 세션용 결과만 반환한다."""
    address='현재 브라우저 세션의 모의 접수함 (외부 전송 없음)'
    packet={**ticket,'submitted_at':datetime.now(timezone.utc).isoformat(),
            'consent':{'confirmed':True,'transport':'demo','destination':address}}
    return {'status':'demo_received','ticket':packet,'destination':address}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        return None


def dispatch(ticket, *, expected_destination):
    """최종 확인 화면에서만 호출. 수신처 변경 시 재확인, 자동 재시도 없음."""
    mode,address=destination()
    if (mode,address)!=expected_destination:
        raise ValueError('수신 설정이 변경되었습니다. 접수 내용을 다시 확인해 주세요.')
    packet={**ticket,'submitted_at':datetime.now(timezone.utc).isoformat(),
            'consent':{'confirmed':True,'transport':mode,'destination':address}}
    if mode=='demo':
        return {'status':'demo_received','ticket':packet,'destination':address}
    body=json.dumps(packet,ensure_ascii=False,indent=2)
    if mode=='email':
        message=EmailMessage()
        message['Subject']=f"[문의 접수] {ticket['category']} / {ticket['ticket_id']}"
        message['From']=os.environ['TICKET_EMAIL_FROM']
        message['To']=address
        message['Message-ID']=f"<{ticket['ticket_id']}@portfolio.invalid>"
        if ticket['reply_email']:
            message['Reply-To']=ticket['reply_email']
        message.set_content(body)
        with smtplib.SMTP_SSL(os.environ['TICKET_SMTP_HOST'],
                              int(os.getenv('TICKET_SMTP_PORT','465')),
                              context=ssl.create_default_context(),timeout=20) as smtp:
            smtp.login(os.environ['TICKET_SMTP_USER'],os.environ['TICKET_SMTP_PASSWORD'])
            refused=smtp.send_message(message)
            if refused:
                raise RuntimeError('수신 서버에서 접수를 거절했습니다.')
    else:
        request=Request(address,data=body.encode('utf-8'),method='POST',headers={
            'Content-Type':'application/json; charset=utf-8',
            'Authorization':'Bearer '+os.environ['TICKET_WEBHOOK_TOKEN'],
            'Idempotency-Key':ticket['ticket_id'],
        })
        # 다른 주소로의 인증정보/문의 내용 재전송을 막는다.
        with build_opener(NoRedirect()).open(request,timeout=20) as response:
            if not 200<=response.status<300:
                raise RuntimeError('접수 API에서 요청을 수락하지 않았습니다.')
    return {'status':'transport_accepted','ticket':packet,'destination':address}


