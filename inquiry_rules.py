"""유형별 고정 질문으로 검색 전 정책 판단과 무관한 되묻기를 차단한다."""

SCENARIOS=('voucher_refund','point_goods_refund','cash_refund','npc_sale','discard_delete',
           'synthesis','normal_enhancement','enhancement_unknown','completed_trade',
           'consumed_expired','system_error','other')
QUESTIONS={
    'purchase_target':'구매하신 대상이 메이플포인트 교환권인가요, 포인트로 구매한 상품인가요? 월드코인 구매 취소라면 말씀해 주세요.',
    'purchased_at':'언제 구매하셨나요? 대략적인 시점도 괜찮습니다.',
    'received':'상품이나 교환권을 우편함에서 수령하셨나요?',
    'used':'교환권이나 상품을 사용하셨나요?',
    'mail_present':'상품이 미수령 상태로 우편함에 남아 있나요?',
    'loss_at':'언제 아이템을 잃으셨나요? 대략적인 시점도 괜찮습니다.',
    'repurchased':'판매한 아이템을 NPC 상점에서 재구매하셨나요?',
    'traded':'이후 해당 아이템의 교환이나 거래가 이루어졌나요?',
    'result_present':'합성·분해 후 얻은 결과물이 현재도 남아 있나요?',
    'incident':'NPC 상점 판매·삭제·분해·강화 등 어떤 상황에서 아이템을 잃으셨나요?',
    'enhancement_cause':'정상적인 강화 실패였나요, 아니면 오류 표시 등 평소와 다른 현상이 있었나요?',
    'error_description':'어디에서 어떤 현상이 발생했나요?'}

REQUIRED_FACTS={
    'voucher_refund':('purchased_at','received','used'),
    'point_goods_refund':('received','mail_present'), 'cash_refund':('purchase_target',),
    'npc_sale':('loss_at','repurchased'), 'discard_delete':('loss_at','traded'),
    'synthesis':('loss_at','result_present'), 'normal_enhancement':(),
    'enhancement_unknown':('enhancement_cause',), 'completed_trade':(),
    'consumed_expired':('incident',), 'system_error':('loss_at','error_description'), 'other':('incident',)}
FACT_LABELS=dict(zip(QUESTIONS,('구매 대상','구매 시점','수령 여부','사용 여부','미수령 우편 보관 여부',
                             '손실 시점','NPC 재구매 여부','후속 교환·거래 여부','합성·분해 결과물 보유',
                             '발생 경위','강화 손실 원인','오류 현상')))


def clarification(scenario, facts, general, category):
    if general:
        return ''
    # 알려진 제한 조건은 나머지 항목을 모두 채우지 않고 근거 검색으로 보낸다.
    if scenario=='voucher_refund' and any(facts.get(k)=='yes' for k in ('received','used')):
        return ''
    if scenario=='point_goods_refund' and (facts.get('received')=='yes' or facts.get('mail_present')=='no'):
        return ''
    if scenario=='npc_sale' and facts.get('repurchased')=='yes':
        return ''
    if scenario=='discard_delete' and facts.get('traded')=='yes':
        return ''
    if scenario=='synthesis' and facts.get('result_present')=='no':
        return ''
    required={
        'voucher_refund':('purchased_at','received','used'),
        'point_goods_refund':('received','mail_present'),
        'cash_refund':(),
        'npc_sale':('loss_at','repurchased'),
        'discard_delete':('loss_at','traded'),
        'synthesis':('loss_at','result_present'),
        'normal_enhancement':(), 'enhancement_unknown':('enhancement_cause',),
        'completed_trade':(), 'consumed_expired':(),
        'system_error':('loss_at','error_description'),
        'other':('purchase_target',) if category=='결제/환불' else ('incident',) if category=='아이템/복구' else ()}
    keys=[key for key in required[scenario] if facts.get(key,'missing')=='missing'
          and not (scenario=='voucher_refund' and key=='used' and facts.get('received')=='no')][:2]
    return '\n\n'.join(QUESTIONS[key] for key in keys)
