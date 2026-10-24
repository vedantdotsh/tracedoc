"""Explicit structural eligibility audit; never repairs benchmark annotations."""
import hashlib
import json
from collections import defaultdict

from .baseline import _literal_list


def audit_samples(samples, page_counts):
    if not isinstance(samples, list):
        raise ValueError('samples must be a list')
    groups = defaultdict(list)
    for row, sample in enumerate(samples):
        try:
            if not isinstance(sample, dict):
                raise ValueError('sample must be an object')
            missing = {'doc_id', 'question', 'answer', 'evidence_pages', 'evidence_sources'} - sample.keys()
            if missing:
                raise ValueError(f'missing keys {sorted(missing)}')
            for key in ('doc_id', 'question', 'answer'):
                if not isinstance(sample[key], str):
                    raise ValueError(f'{key} must be a string')
            _literal_list(sample['evidence_pages'], 'evidence_pages', int)
            _literal_list(sample['evidence_sources'], 'evidence_sources', str)
        except ValueError as exc:
            raise ValueError(f'sample {row}: {exc}') from exc
        groups[(sample['doc_id'], sample['question'])].append(row)
    conflicts = set()
    for rows in groups.values():
        signatures = {json.dumps([samples[r].get('answer'), samples[r].get('evidence_pages')],
                                 sort_keys=True) for r in rows}
        if len(signatures) > 1:
            conflicts.update(rows)
    eligible, quarantine, warnings = [], [], []
    for row, sample in enumerate(samples):
        reasons = []
        pages = _literal_list(sample['evidence_pages'], 'evidence_pages', int)
        sources = _literal_list(sample['evidence_sources'], 'evidence_sources', str)
        doc_id = sample['doc_id']
        negative = sample.get('answer') == 'Not answerable'
        if doc_id not in page_counts:
            reasons.append('missing_document')
        elif any(p < 1 or p > page_counts[doc_id] for p in pages):
            reasons.append('incompatible_physical_page_coordinates')
        if not pages and not negative:
            reasons.append('answer_without_page_evidence')
        if pages and negative:
            reasons.append('negative_with_page_evidence_outside_baseline_contract')
        if row in conflicts:
            reasons.append('conflicting_duplicate')
        if not sources and not negative:
            warnings.append({'source_row': row, 'warning': 'missing_evidence_category'})
        if len(pages) != len(set(pages)):
            warnings.append({'source_row': row, 'warning': 'duplicate_pages_scored_as_set'})
        if reasons:
            quarantine.append({'source_row': row, 'reasons': reasons, 'sample': sample})
        else:
            eligible.append({**sample, 'source_row': row})
    report = {
        'policy': 'physical-page-v1', 'total_rows': len(samples),
        'eligible_rows': len(eligible), 'quarantined_rows': len(quarantine),
        'eligible_answerable': sum(bool(_literal_list(s['evidence_pages'], 'pages', int))
                                  for s in eligible),
        'samples_content_sha256': hashlib.sha256(json.dumps(samples, sort_keys=True,
                                                            ensure_ascii=False).encode()).hexdigest(),
        'note': ('Structural eligibility only, not semantic certification. Quarantine does not '
                 'mean an answer is wrong. Printed page labels and negative questions with '
                 'page references need a separate evaluation policy. Raw samples are unchanged.'),
        'quarantine': quarantine, 'warnings': warnings,
    }
    return eligible, report
