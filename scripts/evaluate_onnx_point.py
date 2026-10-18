#!/usr/bin/env python3
"""Report fixed-threshold card detection counts on a frozen split (IoU >= .5)."""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from card_common import CLASS_NAMES
from infer_onnx import preprocess, decode, model_input_size


def match(predicted, labels, width, height):
    gt = []
    for line in labels.read_text().splitlines():
        c,x,y,w,h = map(float,line.split())
        gt.append((int(c),[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]))
    pairs=[]
    for pi,p in enumerate(predicted):
        a=np.array(p['xyxy'])
        for gi,(c,box) in enumerate(gt):
            if c != p['class_id']:continue
            b=np.array(box);intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            iou=intersection/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection+1e-9)
            if iou>=.5:pairs.append((iou,pi,gi))
    usedp,usedg=set(),set()
    for _,pi,gi in sorted(pairs,reverse=True):
        if pi not in usedp and gi not in usedg:usedp.add(pi);usedg.add(gi)
    tp=np.zeros(54,dtype=int);fp=tp.copy();fn=tp.copy()
    for i,p in enumerate(predicted):(tp if i in usedp else fp)[p['class_id']]+=1
    for i,(c,_) in enumerate(gt):
        if i not in usedg:fn[c]+=1
    return tp,fp,fn


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/prepared_full2000'))
    p.add_argument('--model',type=Path,default=Path('artifacts/card_model.onnx'))
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 0 < a.confidence < 1:p.error('confidence must be between 0 and 1')
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(a.model),sess_options=options,providers=['CPUExecutionProvider'])
    input_size=model_input_size(session)
    rows=[r for r in csv.DictReader((a.data/'split_manifest.csv').open()) if r['split']==a.split]
    tp=np.zeros(54,dtype=int);fp=tp.copy();fn=tp.copy();errors=[];durations=[];perfect=0
    for row in rows:
        path=a.data/row['image'];im=cv2.imread(str(path));tensor,ratio,padding=preprocess(im,input_size)
        start=time.perf_counter();out=session.run(None,{session.get_inputs()[0].name:tensor})[0];durations.append(time.perf_counter()-start)
        pred=decode(out,im.shape,ratio,padding,a.confidence,.45,input_size)
        t,f,n=match(pred,a.data/row['label'],im.shape[1],im.shape[0]);tp+=t;fp+=f;fn+=n
        if not f.sum() and not n.sum():perfect+=1
        else:errors.append(dict(image=str(path.resolve()),source_image=row['source_image'],false_positives=int(f.sum()),missed_cards=int(n.sum()),missed_classes={CLASS_NAMES[i]:int(v) for i,v in enumerate(n) if v},false_positive_classes={CLASS_NAMES[i]:int(v) for i,v in enumerate(f) if v}))
    report=dict(split=a.split,images=len(rows),input_size=input_size,confidence=a.confidence,nms_iou=.45,match_iou=.5,
                true_positive=int(tp.sum()),false_positive=int(fp.sum()),false_negative=int(fn.sum()),
                precision=float(tp.sum()/max(1,tp.sum()+fp.sum())),recall=float(tp.sum()/max(1,tp.sum()+fn.sum())),
                exact_images=perfect,exact_image_fraction=perfect/len(rows),
                per_class={c:dict(tp=int(tp[i]),fp=int(fp[i]),fn=int(fn[i])) for i,c in enumerate(CLASS_NAMES)},
                mean_inference_ms=float(np.mean(durations)*1000),median_inference_ms=float(np.median(durations)*1000),
                timing_note='ONNX session.run only, 4 CPU threads; excludes file loading/preprocessing/NMS; first run included.',
                errors=sorted(errors,key=lambda r:r['false_positives']+r['missed_cards'],reverse=True))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['per_class','errors']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
