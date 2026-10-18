import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from infer_onnx import preprocess,decode,candidate_count,model_input_size
class InferenceBoundaryTests(unittest.TestCase):
 def output(self):return np.zeros((1,58,8400),dtype=np.float32)
 def test_empty_background(self):self.assertEqual(decode(self.output(),(1000,1620,3),640/1620,(0,122)),[])
 def test_exact_confidence_boundary_does_not_crash(self):
  out=self.output();out[0,:4,0]=[100,100,20,20];out[0,4,0]=.25
  self.assertEqual(decode(out,(640,640,3),1,(0,0),.25),[])
 def test_overlapping_different_cards_survive_duplicate_is_removed(self):
  out=self.output()
  for i,(c,score) in enumerate([(0,.9),(0,.8),(1,.85)]):out[0,:4,i]=[200,200,100,100];out[0,4+c,i]=score
  result=decode(out,(640,640,3),1,(0,0));self.assertEqual([r['class_id'] for r in result],[0,1])
 def test_letterbox_restores_original_coordinates(self):
  image=np.zeros((1000,1600,3),dtype=np.uint8);tensor,ratio,pad=preprocess(image);self.assertEqual(tensor.shape,(1,3,640,640));self.assertEqual(pad,(0,120))
  out=self.output();out[0,:4,0]=[200,260,80,120];out[0,4,0]=.9
  self.assertEqual(decode(out,image.shape,ratio,pad)[0]['xyxy'],[400.0,200.0,600.0,500.0])
 def test_explicit_agnostic_nms_removes_cross_class_duplicate_only(self):
  out=self.output()
  for i,(c,score,x) in enumerate([(0,.9,200),(1,.85,200),(2,.8,400)]):
   out[0,:4,i]=[x,200,100,100];out[0,4+c,i]=score
  self.assertEqual([r['class_id'] for r in decode(out,(640,640,3),1,(0,0),agnostic_nms=True)],[0,2])
  self.assertEqual([r['class_id'] for r in decode(out,(640,640,3),1,(0,0))],[0,1,2])
 def test_agnostic_nms_is_explicitly_lossy_for_overlapping_objects(self):
  out=self.output()
  for i,(c,score) in enumerate([(0,.9),(1,.85)]):out[0,:4,i]=[200,200,100,100];out[0,4+c,i]=score
  self.assertEqual(len(decode(out,(640,640,3),1,(0,0))),2)
  self.assertEqual(len(decode(out,(640,640,3),1,(0,0),agnostic_nms=True)),1)
  self.assertEqual(decode(self.output(),(640,640,3),1,(0,0),agnostic_nms=True),[])
 def test_nonfinite_output_rejected(self):
  out=self.output();out[0,0,0]=np.nan
  with self.assertRaises(ValueError):decode(out,(640,640,3),1,(0,0))
 def test_960_uses_18900_predictions_and_restores_coordinates(self):
  image=np.zeros((1000,1600,3),dtype=np.uint8);tensor,ratio,pad=preprocess(image,960)
  self.assertEqual(tensor.shape,(1,3,960,960));self.assertEqual(candidate_count(960),18900)
  out=np.zeros((1,58,18900),dtype=np.float32);out[0,:4,0]=[300,390,120,180];out[0,4,0]=.9
  self.assertEqual(decode(out,image.shape,ratio,pad,input_size=960)[0]['xyxy'],[400.0,200.0,600.0,500.0])
  with self.assertRaises(ValueError):decode(out,image.shape,ratio,pad)
 def test_dynamic_or_mismatched_models_are_rejected(self):
  from types import SimpleNamespace
  class Session:
   def get_inputs(self):return [SimpleNamespace(shape=[1,3,960,960],type='tensor(float)')]
   def get_outputs(self):return [SimpleNamespace(shape=[1,58,8400],type='tensor(float)')]
  with self.assertRaises(ValueError):model_input_size(Session())
  with self.assertRaises(ValueError):candidate_count('dynamic')
if __name__=='__main__':unittest.main()
