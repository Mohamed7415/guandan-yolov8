"""A development extension cannot move held-out games or bypass reviewed labels."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from test_dataset_revision import fixture
from extend_training_dataset import extend
from revise_dataset import annotation_digest, sha
from validate_dataset import validate


def setup(folder, *, group='new_game', duplicate=False):
    root=Path(folder);source=root/'base';source.mkdir();old=fixture(source)
    new=root/'new';new.mkdir();image=new/'native.png';label=new/'native.txt';annotation=new/'native.json'
    if duplicate:image.write_bytes((source/old[1]['image']).read_bytes())
    else:Image.new('RGB',(16,16),(255,50,180)).save(image)
    label.write_text('0 0.25000000 0.25000000 0.37500000 0.37500000\n')
    annotation.write_text(json.dumps(dict(imageWidth=16,imageHeight=16,imagePath=str(image),checked=True,flags={},shapes=[dict(label='3♠',shape_type='rectangle',points=[[1,1],[7,7]],flags={})])))
    row=dict(image=str(image),label=str(label),group_id=group,session_id=group,scene='sorted',reviewed='true',source_annotation=str(annotation))
    manifest=new/'manifest.csv'
    with manifest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    plan=root/'plan.json';plan.write_text(json.dumps(dict(reviewed=True,purpose='development_train_only',evidence='reviewed native full frame',base_dataset_fingerprint=validate(source/'data.yaml')['dataset_fingerprint'],base_manifest_sha256=sha(source/'split_manifest.csv'),new_manifest_sha256=sha(manifest),images=[dict(name=image.name,image_sha256=sha(image),label_sha256=sha(label),annotation_semantic_sha256=annotation_digest(annotation))])))
    return source,manifest,plan,root/'extended',annotation,label,old


class ExtensionChecks(unittest.TestCase):
    def test_all_frozen_bytes_and_splits_preserved_new_samples_only_train(self):
        with tempfile.TemporaryDirectory() as folder:
            source,manifest,plan,out,annotation,label,old=setup(folder)
            originals={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
            report=extend(source,manifest,plan,out,full=False)
            self.assertTrue(report['valid']);self.assertEqual(report['images'],dict(train=2,val=1,test=1))
            with (out/'split_manifest.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([{k:r[k] for k in old[0]} for r in rows[:3]],old)
            self.assertEqual(rows[3]['split'],'train')
            for row in old:
                for k in ('image','label'):self.assertEqual((source/row[k]).read_bytes(),(out/row[k]).read_bytes())
            for p,b in originals.items():self.assertEqual(p.read_bytes(),b)
            self.assertEqual(report['dataset_fingerprint'],validate(out/'data.yaml')['dataset_fingerprint'])
            with self.assertRaisesRegex(ValueError,'already exists'):extend(source,manifest,plan,out,full=False)

    def test_old_session_cannot_be_appended_to_training(self):
        with tempfile.TemporaryDirectory() as folder:
            source,manifest,plan,out,*_=setup(folder,group='val')
            with self.assertRaisesRegex(ValueError,'overlaps frozen'):extend(source,manifest,plan,out,full=False)
            self.assertFalse(out.exists())

    def test_pixel_duplicate_of_validation_is_rejected_and_staging_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            source,manifest,plan,out,*_=setup(folder,duplicate=True)
            with self.assertRaisesRegex(ValueError,'Duplicate image pixels across'):extend(source,manifest,plan,out,full=False)
            self.assertFalse(out.exists());self.assertFalse(list(Path(folder).glob('.extended.*')))

    def test_new_label_or_base_label_drift_invalidates_plan(self):
        for change_base in (False,True):
            with self.subTest(base=change_base),tempfile.TemporaryDirectory() as folder:
                source,manifest,plan,out,annotation,label,old=setup(folder)
                path=source/old[1]['label'] if change_base else label
                path.write_text('1 0.25000000 0.25000000 0.37500000 0.37500000\n')
                with self.assertRaisesRegex(ValueError,'fingerprint changed|evidence changed'):extend(source,manifest,plan,out,full=False)
                self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
