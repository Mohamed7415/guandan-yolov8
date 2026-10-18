"""A lower false-positive total cannot hide missing cards or substituted inputs."""
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from card_common import CLASS_NAMES
from validation_gate import apply_gate, audit_report, compare, sha, validation_truth

SETTINGS=dict(input_size=960,confidence=.25,nms_iou=.45,nms_mode='class_aware',match_iou=.5,max_detections=300)
TRUTH={'a.png':Counter({'3♠':1,'3♥':1}),'b.png':Counter({'4♠':1}),'c.png':Counter({'SJ':1})}


def score(missing=None, extra=None):
    missing,extra=missing or {},extra or {};truth=Counter()
    for counts in TRUTH.values():truth.update(counts)
    per_class={c:dict(tp=truth[c],fp=0,fn=0) for c in CLASS_NAMES};errors=[]
    for image in TRUTH:
        n,f=missing.get(image,{}),extra.get(image,{})
        for cls,count in n.items():per_class[cls]['tp']-=count;per_class[cls]['fn']+=count
        for cls,count in f.items():per_class[cls]['fp']+=count
        if n or f:errors.append(dict(image='images/val/'+image,missed_cards=sum(n.values()),false_positives=sum(f.values()),missed_classes=n,false_positive_classes=f))
    return dict(split='val',images=3,**{k:v for k,v in SETTINGS.items() if k!='max_detections'},
                true_positive=sum(c['tp'] for c in per_class.values()),false_positive=sum(c['fp'] for c in per_class.values()),
                false_negative=sum(c['fn'] for c in per_class.values()),exact_images=3-len(errors),per_class=per_class,errors=errors)


def baseline():return score({'a.png':{'3♥':1}},{'b.png':{'4♠':1},'c.png':{'BJ':1}})


class ValidationRuleChecks(unittest.TestCase):
    def test_recovered_cards_and_fewer_false_positives_are_accepted(self):
        decision=compare(baseline(),score(),TRUTH,SETTINGS)
        self.assertTrue(decision['accepted_for_old_regression']);self.assertFalse(decision['goal_completed'])
        self.assertEqual(decision['new_blind_test_images'],0)

    def test_tie_is_not_misreported_as_improvement(self):
        self.assertFalse(compare(baseline(),baseline(),TRUTH,SETTINGS)['accepted_for_old_regression'])

    def test_same_total_misses_moved_to_another_image_are_rejected(self):
        candidate=score({'b.png':{'4♠':1}})
        decision=compare(baseline(),candidate,TRUTH,SETTINGS)
        self.assertEqual(decision['baseline']['false_negative'],decision['candidate']['false_negative'])
        self.assertFalse(decision['accepted_for_old_regression'])
        self.assertEqual(decision['worsened_misses'][0]['image'],'b.png')

    def test_same_total_misses_swapped_between_classes_are_rejected(self):
        decision=compare(baseline(),score({'a.png':{'3♠':1}}),TRUTH,SETTINGS)
        self.assertFalse(decision['accepted_for_old_regression'])
        self.assertEqual(decision['worsened_misses'][0]['class'],'3♠')

    def test_zero_output_cannot_pass_by_having_no_false_positives(self):
        decision=compare(baseline(),score({k:dict(v) for k,v in TRUTH.items()}),TRUTH,SETTINGS)
        self.assertEqual(decision['candidate']['false_positive'],0)
        self.assertFalse(decision['accepted_for_old_regression'])

    def test_changed_settings_and_inconsistent_reports_are_refused(self):
        cases=[]
        for key,value in [('confidence',.3),('nms_mode','agnostic'),('images',2),('exact_images',2)]:
            r=score();r[key]=value;cases.append(r)
        r=baseline();r['errors'][0]['image']='unknown.png';cases.append(r)
        r=score();r['per_class']['3♠']['tp']=0;cases.append(r)
        r=score();del r['per_class']['SJ'];cases.append(r)
        for report in cases:
            with self.subTest(report=report),self.assertRaises(ValueError):audit_report(report,TRUTH,SETTINGS)


class ProvenanceChecks(unittest.TestCase):
    def fixture(self,folder):
        root=Path(folder);data=root/'data';(data/'images/val').mkdir(parents=True);(data/'labels/val').mkdir(parents=True)
        rows=[]
        for index,(name,counts) in enumerate(TRUTH.items()):
            Image.new('RGB',(64,64),(40+index*50,100,80)).save(data/'images/val'/name)
            label='labels/val/'+Path(name).stem+'.txt';(data/label).write_text(''.join(f'{CLASS_NAMES.index(cls)} .5 .5 .2 .2\n' for cls in counts))
            rows.append(dict(image='images/val/'+name,label=label,split='val',group_id='game'+str(index),session_id='game'+str(index)))
        with (data/'split_manifest.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (data/'data.yaml').write_text(json.dumps({'path':str(data)}))
        _,signature=validation_truth(data);b=baseline();br=root/'baseline.json';br.write_text(json.dumps(b))
        protocol=root/'protocol.json';protocol.write_text(json.dumps(dict(data=str(data/'data.yaml'),split_manifest_sha256=sha(data/'split_manifest.csv'),training={'input_size':960},fixed_inference={k:v for k,v in SETTINGS.items() if k!='input_size'},selection={'images':3,'baseline':dict(tp=3,fp=2,fn=1,exact=0)})))
        gate=root/'gate.json';gate.write_text(json.dumps(dict(protocol=str(protocol),protocol_sha256=sha(protocol),baseline_report=str(br),baseline_report_sha256=sha(br),validation_inputs_sha256=signature)))
        model=root/'candidate.onnx';model.write_bytes(b'locally_trusted_model_hash_fixture_not_an_inference_test')
        r=score();r.update(model_sha256=sha(model),split_manifest_sha256=sha(data/'split_manifest.csv'),evaluation_inputs_sha256=signature)
        candidate=root/'candidate.json';candidate.write_text(json.dumps(r))
        verified=root/'verified.json';verified.write_text(json.dumps(dict(onnx_sha256=sha(model),class_names=CLASS_NAMES,export_input_size=960,raw_parity=[{'verified_by_exporter':True}])))
        return gate,candidate,model,verified,root/'decision.json',data

    def test_decision_is_frozen_and_rejects_a_substituted_model(self):
        with tempfile.TemporaryDirectory() as folder:
            gate,candidate,model,verified,out,_=self.fixture(folder)
            result=apply_gate(gate,candidate,model,verified,out)
            self.assertTrue(result['accepted_for_old_regression']);self.assertEqual(result['frozen_candidate_for_regression']['onnx_sha256'],sha(model))
            with self.assertRaisesRegex(ValueError,'immutable'):apply_gate(gate,candidate,model,verified,out)
            model.write_bytes(b'other_model')
            with self.assertRaisesRegex(ValueError,'another model'):apply_gate(gate,candidate,model,verified,Path(folder)/'other.json')

    def test_label_or_pixel_drift_invalidates_the_frozen_inputs(self):
        for pixels in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                gate,candidate,model,verified,out,data=self.fixture(folder)
                if pixels:Image.new('RGB',(64,64),'white').save(data/'images/val/a.png')
                else:(data/'labels/val/a.txt').write_text('0 .4 .4 .2 .2\n')
                with self.assertRaisesRegex(ValueError,'images or labels changed'):apply_gate(gate,candidate,model,verified,out)
                self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
