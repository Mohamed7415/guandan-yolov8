#!/usr/bin/env python3
"""Run the exported card detector using only NumPy, OpenCV and ONNX Runtime."""
import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def candidate_count(input_size):
    if not isinstance(input_size, int) or input_size < 32 or input_size % 32:
        raise ValueError("Expected a static square input size divisible by 32")
    return sum((input_size // stride) ** 2 for stride in (8, 16, 32))


def model_input_size(session):
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected one image input and one detection output")
    shape = inputs[0].shape
    if len(shape) != 4 or shape[:2] != [1, 3] or shape[2] != shape[3] or inputs[0].type != "tensor(float)":
        raise ValueError("Expected static FP32 RGB [1,3,S,S] input")
    size = shape[2]
    if outputs[0].shape != [1, 58, candidate_count(size)] or outputs[0].type != "tensor(float)":
        raise ValueError("Expected a standard 54-class YOLOv8 output for this input size")
    return size


def preprocess(image, input_size=640):
    candidate_count(input_size)
    h, w = image.shape[:2]
    ratio = min(input_size / h, input_size / w)
    resized = (round(w * ratio), round(h * ratio))
    dx, dy = (input_size - resized[0]) / 2, (input_size - resized[1]) / 2
    left, top = round(dx - 0.1), round(dy - 0.1)
    canvas = cv2.copyMakeBorder(cv2.resize(image, resized), top, round(dy + .1),
                                left, round(dx + .1), cv2.BORDER_CONSTANT, value=(114,114,114))
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2,0,1)[None], dtype=np.float32) / 255
    return tensor, ratio, (left, top)


def decode(output, shape, ratio, padding, confidence=.25, iou=.45, input_size=640):
    if output.shape != (1,58,candidate_count(input_size)) or not np.isfinite(output).all():
        raise ValueError(f"Invalid model output: {output.shape}")
    predictions = output[0].T
    classes = predictions[:,4:].argmax(axis=1)
    scores = predictions[np.arange(len(predictions)),classes+4]
    keep = scores >= confidence
    predictions, classes, scores = predictions[keep,:4], classes[keep], scores[keep]
    boxes = predictions.copy()
    boxes[:,:2] -= boxes[:,2:] / 2
    selected = []
    # Class-aware NMS keeps visually overlapping cards with different classes.
    for c in np.unique(classes):
        subset = np.where(classes == c)[0]
        indices = cv2.dnn.NMSBoxes(boxes[subset].tolist(),scores[subset].tolist(),confidence,iou)
        selected.extend(subset[np.asarray(indices, dtype=np.int64).reshape(-1)].tolist())
    selected.sort(key=lambda i:float(scores[i]),reverse=True)
    h,w = shape[:2]
    results = []
    for i in selected[:300]:
        x,y,bw,bh = boxes[i]
        x1,x2 = np.clip((np.array([x,x+bw])-padding[0])/ratio,0,w)
        y1,y2 = np.clip((np.array([y,y+bh])-padding[1])/ratio,0,h)
        if x2 > x1 and y2 > y1:
            results.append(dict(class_id=int(classes[i]),confidence=round(float(scores[i]),6),
                                xyxy=[round(float(v),2) for v in (x1,y1,x2,y2)]))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image",type=Path)
    p.add_argument("--model",type=Path,default=Path("artifacts/card_model.onnx"))
    p.add_argument("--output",type=Path,default=Path("reports/inference"))
    p.add_argument("--confidence",type=float,default=.25)
    p.add_argument("--iou",type=float,default=.45)
    a = p.parse_args()
    if not (0 < a.confidence < 1 and 0 < a.iou < 1): p.error("Thresholds must be between 0 and 1")
    original = cv2.imread(str(a.image))
    if original is None: p.error("Cannot decode input image")
    session = ort.InferenceSession(str(a.model),providers=["CPUExecutionProvider"])
    try: input_size = model_input_size(session)
    except ValueError as error: p.error(str(error))
    names = ast.literal_eval(session.get_modelmeta().custom_metadata_map["names"])
    tensor,ratio,padding = preprocess(original,input_size)
    output = session.run(None,{session.get_inputs()[0].name:tensor})[0]
    detections = decode(output,original.shape,ratio,padding,a.confidence,a.iou,input_size)
    for d in detections: d["label"] = names[d["class_id"]]
    a.output.mkdir(parents=True,exist_ok=True)
    record = dict(image=str(a.image.resolve()),model=str(a.model.resolve()),confidence=a.confidence,
                  iou=a.iou,input_size=input_size,detections=detections)
    (a.output/(a.image.stem+".json")).write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    from PIL import Image,ImageDraw,ImageFont
    preview = Image.fromarray(cv2.cvtColor(original,cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(preview)
    font_paths = [Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),
                  Path.home()/'Library/Application Support/Ultralytics/Arial.Unicode.ttf']
    font = next((ImageFont.truetype(str(f),22) for f in font_paths if f.exists()),ImageFont.load_default())
    for d in detections:
        box=d['xyxy'];draw.rectangle(box,outline='#00ff00',width=2)
        draw.text((box[0],max(0,box[1]-23)),f"{d['label']} {d['confidence']:.2f}",font=font,fill='blue',stroke_width=1,stroke_fill='white')
    preview.save(a.output/(a.image.stem+"_prediction.jpg"),quality=95)
    print(json.dumps(dict(objects=len(detections),output=str(a.output.resolve())),ensure_ascii=False))


if __name__ == "__main__": main()
