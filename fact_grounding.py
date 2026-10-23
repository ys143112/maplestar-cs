"""추출 상태는 고객 원문의 근거 구절로 제한한다. 불확실한 값은 확정하지 않는다."""
import re
from inquiry_rules import QUESTIONS, FACT_LABELS

BINARY_KEYS=('received','used','mail_present','repurchased','traded','result_present')
EVIDENCE_RULES=(
    'fact_evidence에 case_facts의 각 값에 대한 고객 원문 구절을 그대로 적는다. '
    '형식은 {"loss_at":"오늘 NPC 상점에 팔았어요", "traded":"교환은 안 했어요"}와 같다. '
    '원문에 없는 구절·AI 답변·정책을 증거로 만들지 않는다. '
    'traded=yes는 고객이 실제 교환/거래를 했다고 명시한 경우만 가능하며 타인 습득은 unknown이다. '
    'result_present=yes는 결과물이 지금도 남아 있다고 명시한 경우만 가능하다. '
    '과거 결과물을 얻었다거나 이미 사용했다는 진술만으로 현재 보유=yes를 쓰지 않는다. '
    '일부 사용 후 나머지 보유와 전량 사용을 구별하고 불명확하면 unknown으로 둔다. '
    '시점 등 비이진 값은 원문 표현을 그대로 사용하고 날짜를 임의 환산하지 않는다. ')


def ground_facts(facts,evidence,source):
    if not isinstance(evidence,dict):
        evidence={}
    grounded={};quotes={};warnings=[]
    for key,value in facts.items():
        if key not in QUESTIONS:
            continue
        quote=evidence.get(key,'')
        valid=isinstance(quote,str) and bool(quote.strip()) and quote in source
        accepted=value in ('missing','unknown')
        if value not in ('missing','unknown'):
            accepted=valid and (key in BINARY_KEYS or value in quote)
        if accepted and value=='yes' and key=='traded':
            accepted=bool(re.search(r'(?:거래|교환).{0,16}(?:했|하였|완료|됐|되었)|(?:옥션|경매|거래소).{0,16}(?:판매|팔)',quote)) and not bool(
                re.search(r'(?:거래|교환).{0,12}(?:안\s*했|않|없|못)',quote))
        if accepted and value=='yes' and key=='result_present':
            accepted=bool(re.search(r'남아|남았|남은|남겨|보유|가지고\s*있|갖고\s*있',quote)) and not bool(
                re.search(r'안\s*남|남.{0,6}(?:않|없)|보유.{0,6}(?:않|없)',quote))
        if accepted:
            grounded[key]=value
        else:
            grounded[key]='unknown'
            warnings.append(f'{FACT_LABELS[key]}: 고객 원문에서 확정할 수 없어 미확인으로 보존했습니다.')
        if valid:
            quotes[key]=quote
    return grounded,quotes,warnings
