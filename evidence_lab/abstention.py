"""Conservative, optional evidence checks; passing is never proof of support.

Adapted from rag-reliability-lab/src/raglab/answering.py detail/conflict checks.
No reference answers, benchmark labels, model output, or retrieval scores enter here.
"""
import re

VERSION = 'evidence-gate-v2'
PROVENANCE = {
    'source': 'rag-reliability-lab/src/raglab/answering.py',
    'source_sha256': '16aa613f5d388ad95238127a0147d250638205b449e6988752791e590f5e389e',
    'adaptation': 'Explicit scopes only; no service-specific phone/URL/header heuristics; no extractive answer fallback.',
}
_NUMBER = re.compile(r'\b\d+(?:[.,]\d+)*\b')
_DERIVED = re.compile(r'\b(?:compare|comparison|difference|increase|decrease|change|growth|ratio|percent|percentage|average|sum|total|between|calculate|count|list)\b', re.I)
_NUMBER_WORD = re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|hundred|thousand|million|billion|none|no)\b', re.I)
_STOP = set('what which when where how is are was were the a an of for in during does do to and number many much'.split())


def check(question, sources):
    """Return named reasons from the exact, context-budgeted source text.

    Comparison/arithmetic bypass applies to quantity/conflict checks; explicitly
    requested period and identifier scopes still require literal evidence.
    """
    q = question.casefold()
    evidence = '\n'.join(s['text'] for s in sources).casefold()
    reasons, details = [], []

    def reject(code, **detail):
        if code not in reasons:
            reasons.append(code)
        details.append({'reason': code, **detail})

    derived = bool(_DERIVED.search(q))
    verification = bool(re.match(r'\s*(?:is|are|was|were|can|could|do|does|did|will)\b', q))
    if not verification:
        years = re.findall(r'\b(?:in|during|for)\s+((?:19|20)\d{2})\b', q)
        missing = sorted(set(years) - set(_NUMBER.findall(evidence)))
        if missing:
            reject('missing_requested_year', values=missing)
        for kind, value in re.findall(r'\b(section|article|table|figure|chapter|rule|policy|form|version)\s+(\d+(?:\.\d+)*)\b', q):
            if not re.search(rf'\b{kind}\s+{re.escape(value)}(?![\w.]|\d)', evidence):
                reject('missing_typed_identifier', kind=kind, value=value)
        # A plainly labelled publication date is not a requested quantity. Keep
        # bare four-digit counts (including "2021 participants") untouched.
        quantity_evidence = re.sub(r'\b(?:published in|publication date[: ]*)\s*(?:19|20)\d{2}\b', '', evidence)
        if not derived and re.search(r'\bhow (?:many|much)\b', q) and not (_NUMBER.search(quantity_evidence) or _NUMBER_WORD.search(quantity_evidence)):
            reject('missing_numeric_detail')
    if not derived:
        terms = set(re.findall(r'[a-z]+', q)) - _STOP
        facts = {}
        for source in sources:
            for sentence in re.split(r'(?<=[.!?])\s+|\n+', source['text'].casefold()):
                numbers = _NUMBER.findall(sentence)
                if len(numbers) != 1:
                    continue
                core = ' '.join(_NUMBER.sub('<number>', sentence).split())
                if len((set(re.findall(r'[a-z]+', core)) - _STOP) & terms) < 2:
                    continue
                for previous in facts.get(core, []):
                    if previous['doc_id'] != source['doc_id'] and previous['value'] != numbers[0]:
                        reject('conflicting_numeric_facts', context=core,
                               sources=[previous['source_id'], source.get('source_id')])
                facts.setdefault(core, []).append({'doc_id': source['doc_id'],
                    'source_id': source.get('source_id'), 'value': numbers[0]})
    return {'version': VERSION, 'abstain': bool(reasons), 'reasons': reasons, 'details': details,
            'support_verified': False}
