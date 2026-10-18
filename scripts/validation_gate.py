#!/usr/bin/env python3
"""Enforce the preregistered fixed-point validation rule before old regression."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from atomic_io import atomic_json, publish_new_file
from card_common import CLASS_NAMES, read_labels


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_entry(root: Path, row: dict) -> list:
    return [row['image'], sha(root / row['image']), sha(root / row['label']),
            row['group_id'], row.get('session_id', '')]


def inputs_digest(entries: list) -> str:
    return hashlib.sha256(json.dumps(sorted(entries), ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def validation_truth(root: Path) -> tuple[dict, str]:
    with (root / 'split_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['split'] == 'val']
    truth, entries = {}, []
    for row in rows:
        key = Path(row['image']).name
        if key in truth:
            raise ValueError('Duplicate validation image identity')
        truth[key] = Counter(CLASS_NAMES[c] for c, *_ in read_labels(root / row['label']))
        entries.append(input_entry(root, row))
    if not truth:
        raise ValueError('Validation split is empty')
    return truth, inputs_digest(entries)


def integer(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Detection counts must be nonnegative integers')
    return value


def audit_report(report: dict, truth: dict, settings: dict) -> dict:
    """Check aggregates against per-image errors and all ground-truth classes."""
    for key, expected in {'split': 'val', 'input_size': settings['input_size'],
                          **{k: settings[k] for k in ('confidence', 'nms_iou', 'nms_mode', 'match_iou')}}.items():
        if report.get(key) != expected:
            raise ValueError(f'Fixed evaluation setting changed: {key}')
    if report.get('images') != len(truth):
        raise ValueError('Validation image count changed')
    per_class = report.get('per_class', {})
    if set(per_class) != set(CLASS_NAMES):
        raise ValueError('Report must contain exactly the 54 required classes')
    fp, fn, misses = Counter(), Counter(), {}
    for error in report.get('errors', []):
        key = Path(error['image']).name
        if key not in truth or key in misses:
            raise ValueError('Unknown or repeated validation error image')
        missing, extra = error.get('missed_classes', {}), error.get('false_positive_classes', {})
        if not set(missing) <= set(CLASS_NAMES) or not set(extra) <= set(CLASS_NAMES):
            raise ValueError('Unknown error class')
        missing = {c: integer(v) for c, v in missing.items()}
        extra = {c: integer(v) for c, v in extra.items()}
        if any(n > truth[key].get(c, 0) for c, n in missing.items()):
            raise ValueError('Missed-card count exceeds ground truth')
        if sum(missing.values()) != integer(error['missed_cards']) or sum(extra.values()) != integer(error['false_positives']):
            raise ValueError('Error-row aggregate disagrees with class counts')
        if not sum(missing.values()) + sum(extra.values()):
            raise ValueError('An error row cannot describe a perfect image')
        misses[key] = missing
        fn.update(missing); fp.update(extra)
    ground_truth = Counter()
    for counts in truth.values():
        ground_truth.update(counts)
    tp_total = 0
    for cls in CLASS_NAMES:
        counts = per_class[cls]
        tp, f, n = (integer(counts[k]) for k in ('tp', 'fp', 'fn'))
        if tp + n != ground_truth[cls] or n != fn[cls] or f != fp[cls]:
            raise ValueError(f'Per-class report disagrees with truth/errors: {cls}')
        tp_total += tp
    totals = {'true_positive': tp_total, 'false_positive': sum(fp.values()),
              'false_negative': sum(fn.values()), 'exact_images': len(truth) - len(misses)}
    if any(integer(report[k]) != v for k, v in totals.items()):
        raise ValueError('Overall report disagrees with ground truth or error rows')
    return {'counts': totals, 'misses': misses}


def compare(baseline: dict, candidate: dict, truth: dict, settings: dict) -> dict:
    before = audit_report(baseline, truth, settings)
    after = audit_report(candidate, truth, settings)
    worsened = []
    for image in truth:
        old, new = before['misses'].get(image, {}), after['misses'].get(image, {})
        for cls, count in new.items():
            if count > old.get(cls, 0):
                worsened.append({'image': image, 'class': cls,
                                 'baseline_missed': old.get(cls, 0), 'candidate_missed': count})
    a, b = before['counts'], after['counts']
    checks = {'no_image_or_class_miss_increase': not worsened,
              'fewer_false_positives': b['false_positive'] < a['false_positive'],
              'more_exact_images': b['exact_images'] > a['exact_images']}
    return {'accepted_for_old_regression': all(checks.values()), 'checks': checks,
            'baseline': a, 'candidate': b, 'worsened_misses': worsened,
            'new_blind_test_images': 0, 'goal_completed': False}


def apply_gate(config: Path, candidate_report: Path, model: Path,
               verification: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('Decision output already exists; frozen decisions are immutable')
    gate = json.loads(config.read_text(encoding='utf-8'))
    protocol = Path(gate['protocol'])
    baseline_path = Path(gate['baseline_report'])
    if sha(protocol) != gate['protocol_sha256'] or sha(baseline_path) != gate['baseline_report_sha256']:
        raise ValueError('Frozen protocol or baseline report changed')
    planned = json.loads(protocol.read_text(encoding='utf-8'))
    import yaml
    data = Path(planned['data']).resolve()
    dataset = yaml.safe_load(data.read_text(encoding='utf-8'))
    root = Path(dataset.get('path', data.parent))
    root = root.resolve() if root.is_absolute() else (data.parent / root).resolve()
    if sha(root / 'split_manifest.csv') != planned['split_manifest_sha256']:
        raise ValueError('Frozen dataset manifest changed')
    truth, signature = validation_truth(root)
    if len(truth) != planned['selection']['images'] or signature != gate['validation_inputs_sha256']:
        raise ValueError('Frozen validation images or labels changed')
    settings = {'input_size': planned['training']['input_size'], **planned['fixed_inference']}
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    candidate = json.loads(candidate_report.read_text(encoding='utf-8'))
    model_sha = sha(model)
    if candidate.get('model_sha256') != model_sha:
        raise ValueError('Candidate score report belongs to another model')
    if candidate.get('split_manifest_sha256') != planned['split_manifest_sha256'] or candidate.get('evaluation_inputs_sha256') != signature:
        raise ValueError('Candidate scores were not produced on the frozen validation inputs')
    verified = json.loads(verification.read_text(encoding='utf-8'))
    if (verified.get('onnx_sha256') != model_sha or verified.get('class_names') != CLASS_NAMES
            or verified.get('export_input_size') != settings['input_size'] or not verified.get('raw_parity')):
        raise ValueError('Candidate ONNX lacks matching export/parity evidence')
    decision = compare(baseline, candidate, truth, settings)
    expected = planned['selection']['baseline']
    if decision['baseline'] != dict(true_positive=expected['tp'], false_positive=expected['fp'],
                                    false_negative=expected['fn'], exact_images=expected['exact']):
        raise ValueError('Pinned baseline does not match the preregistered counts')
    decision.update(created_at=datetime.now(timezone.utc).isoformat(),
                    gate_config_sha256=sha(config), protocol_sha256=sha(protocol),
                    baseline_report_sha256=sha(baseline_path),
                    candidate_report_sha256=sha(candidate_report), onnx_sha256=model_sha,
                    verification_sha256=sha(verification), evaluation_inputs_sha256=signature)
    decision['frozen_candidate_for_regression'] = (
        {'model': str(model), 'onnx_sha256': model_sha, **settings,
         'status': 'selected_for_old_regression_not_formal_release',
         'new_blind_test_images': 0, 'replaces_formal_v1': False}
        if decision['accepted_for_old_regression'] else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.validation-decision-', dir=output.parent) as folder:
        staged = Path(folder) / 'decision.json'
        atomic_json(staged, decision)
        publish_new_file(staged, output)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'candidate-report', 'model', 'verification', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        decision = apply_gate(args.config, args.candidate_report, args.model, args.verification, args.output)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'Validation decision stopped: {error}\n')


if __name__ == '__main__':
    main()
