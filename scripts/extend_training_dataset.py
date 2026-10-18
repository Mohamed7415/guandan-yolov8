#!/usr/bin/env python3
"""Append reviewed development games to train while preserving every frozen split.

The explicit plan pins the base dataset, converted manifest and every new image,
label and reviewed annotation. Existing validation/test samples never move.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from atomic_io import atomic_json
from card_common import CLASS_NAMES, manifest_rows
from convert_annotations import convert_one
from revise_dataset import annotation_digest, sha
from validate_dataset import validate


def inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Frozen image/label paths must stay relative to the dataset')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Dataset path escapes its root')
    return resolved


def extend(source: Path, manifest: Path, plan_path: Path, output: Path, *, full: bool = True) -> dict:
    source, manifest, output = source.resolve(), manifest.resolve(), output.resolve()
    if output.exists():
        raise ValueError('Extension output already exists; use a new version')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('reviewed') is not True or plan.get('purpose') != 'development_train_only' or not plan.get('evidence'):
        raise ValueError('An explicit reviewed development_train_only plan and evidence are required')
    base = validate(source / 'data.yaml', full=full)
    if base['errors']:
        raise ValueError('Base dataset failed validation: ' + '; '.join(base['errors'][:8]))
    if base['dataset_fingerprint'] != plan.get('base_dataset_fingerprint'):
        raise ValueError('Base dataset fingerprint changed')
    if sha(source / 'split_manifest.csv') != plan.get('base_manifest_sha256') or sha(manifest) != plan.get('new_manifest_sha256'):
        raise ValueError('A pinned manifest changed')
    with (source / 'split_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        old = list(csv.DictReader(stream))
    additions = manifest_rows(manifest)
    pins = plan.get('images', [])
    by_name = {p['name']: p for p in pins}
    if len(by_name) != len(pins) or set(by_name) != {Path(r['image']).name for r in additions} or len(additions) != len(pins):
        raise ValueError('Plan must pin every new image exactly once')
    old_units = {r.get(k, '').strip() for r in old for k in ('group_id', 'session_id')} - {''}
    old_sources = {r.get('source_image') for r in old}
    converted = []
    for row in additions:
        if not row.get('session_id', '').strip():
            raise ValueError('New development images need a conservative session_id')
        if row['group_id'] in old_units or row['session_id'] in old_units or row['image'] in old_sources:
            raise ValueError('New development group/session/source overlaps frozen data')
        image, label = Path(row['image']), Path(row['label'])
        annotation = Path(row.get('source_annotation', ''))
        if not annotation.is_absolute():
            annotation = (manifest.parent / annotation).resolve()
        pin = by_name[image.name]
        if sha(image) != pin['image_sha256'] or sha(label) != pin['label_sha256'] or annotation_digest(annotation) != pin['annotation_semantic_sha256']:
            raise ValueError('New review evidence changed: ' + image.name)
        if json.loads(annotation.read_text(encoding='utf-8')).get('checked') is not True:
            raise ValueError('New source annotation is not marked reviewed')
        _, _, _, lines, _ = convert_one({**row, 'label': str(annotation)}, manifest.parent, {n:i for i,n in enumerate(CLASS_NAMES)})
        if lines != label.read_text(encoding='utf-8-sig').splitlines():
            raise ValueError('Converted labels disagree with reviewed annotation: ' + image.name)
        converted.append((row, annotation))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', dir=output.parent))
    try:
        prepared = []
        for row in old:
            row = dict(row)
            for key in ('image', 'label'):
                src = inside(source, row[key]); dest = inside(staging, row[key]);dest.parent.mkdir(parents=True, exist_ok=True)
                # Never mutate hardlinked images; labels are private copies.
                if key == 'image':
                    try: os.link(src, dest)
                    except OSError: shutil.copy2(src, dest)
                else: shutil.copy2(src, dest)
            annotation_rel = row.get('source_annotation')
            if annotation_rel:
                annotation = inside(source, annotation_rel)
                doc = json.loads(annotation.read_text(encoding='utf-8'))
                target = inside(staging, annotation_rel)
                doc['imagePath'] = os.path.relpath(staging / row['image'], target.parent)
                doc['imageData'] = None
                atomic_json(target, doc)
            prepared.append(row)
        for row, annotation in converted:
            identifier = hashlib.sha256((row['group_id'] + '\0' + row['image']).encode()).hexdigest()[:16]
            stem = 'card_' + identifier
            image_rel = 'images/train/' + stem + Path(row['image']).suffix.lower()
            label_rel = 'labels/train/' + stem + '.txt'
            annotation_rel = 'annotations/' + stem + '.json'
            for rel, origin in ((image_rel, row['image']), (label_rel, row['label'])):
                target = staging / rel
                if target.exists(): raise ValueError('New sample collides with frozen data')
                target.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(origin, target)
            doc = json.loads(annotation.read_text(encoding='utf-8'))
            doc['imagePath'] = os.path.relpath(staging / image_rel, (staging / annotation_rel).parent)
            doc['imageData'] = None;atomic_json(staging / annotation_rel, doc)
            prepared.append(dict(image=image_rel, label=label_rel, group_id=row['group_id'], session_id=row['session_id'], scene=row['scene'], reviewed='true', split='train', source_image=row['image'], source_label=row['label'], source_annotation=annotation_rel))
        fields = list(dict.fromkeys(k for r in prepared for k in r))
        with (staging / 'split_manifest.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader();writer.writerows(prepared)
        import yaml
        config = yaml.safe_load((source / 'data.yaml').read_text(encoding='utf-8'));config['path'] = str(output)
        atomic_json(staging / 'data.yaml', config)
        checked = validate(staging / 'data.yaml', dataset_root=staging, full=full)
        if checked['errors']:
            raise ValueError('Extended dataset failed validation: ' + '; '.join(checked['errors'][:8]))
        # Byte-for-byte comparisons include train as well as held-out labels.
        for row in old:
            for key in ('image', 'label'):
                if sha(source / row[key]) != sha(staging / row[key]):
                    raise ValueError('Frozen input changed during extension')
        atomic_json(staging / 'extension.json', {'purpose':'development_train_only', 'source_dataset':str(source), 'source_fingerprint':base['dataset_fingerprint'], 'plan_sha256':sha(plan_path), 'added_images':len(additions), 'base_images':len(old), 'all_base_image_and_label_bytes_preserved':True, 'all_base_split_assignments_preserved':True, 'new_blind_test_images':0})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report = validate(output / 'data.yaml', full=full)
    atomic_json(output / 'dataset_report.json', report)
    return {'output':str(output), 'images':report['images'], 'valid':report['valid'], 'warnings':report['warnings'], 'dataset_fingerprint':report['dataset_fingerprint']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','manifest','plan','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    try: print(json.dumps(extend(a.source,a.manifest,a.plan,a.output),ensure_ascii=False,indent=2))
    except (ValueError,OSError,KeyError,TypeError) as error:p.exit(2,f'Dataset extension stopped: {error}\n')

if __name__ == '__main__':main()
