#!/usr/bin/env python3
"""Draft X-AnyLabeling boxes from verified game glyphs; never approves labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def ink_mask(rgb):
    a = rgb.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    black = (a.max(2) < 125) & (a.max(2) - a.min(2) < 75)
    red = (r > 70) & (r > g * 1.4) & (r > b * 1.4)
    yellow = (r > 130) & (g > 80) & (b < 110) & (r > g * 1.1)
    return (black | red | yellow).astype(np.uint8)


def components(mask, low_h=15, high_h=85):
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [tuple(map(int, s[:4])) for s in stats[1:]
            if low_h <= s[3] <= high_h and 5 <= s[2] <= 100 and s[4] > 80]


def normalize(mask):
    ys, xs = np.where(mask)
    if not len(xs): return np.zeros((64, 64), np.uint8)
    trim = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = trim.shape
    scale = min(58 / h, 58 / w)
    small = cv2.resize(trim, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_NEAREST)
    result = np.zeros((64, 64), np.uint8)
    y, x = (64-small.shape[0])//2, (64-small.shape[1])//2
    result[y:y+small.shape[0], x:x+small.shape[1]] = small
    return result


def crop_mask(mask, box):
    x, y, w, h = box
    return mask[y:y+h, x:x+w]


def merge_tens(boxes, mask=None, rank_bank=None):
    used, result = set(), []
    # A zero can start a pixel above its leading one; process the thin one
    # first rather than treating connected-component scan order as text order.
    order=sorted(range(len(boxes)),key=lambda i:(boxes[i][2]/boxes[i][3]>=.48,boxes[i][1],boxes[i][0]))
    for i in order:
        x,y,w,h=boxes[i]
        if i in used: continue
        if w / h < .48:
            options = [(j, b) for j,b in enumerate(boxes) if j != i and j not in used
                       and -.15*h < b[0]-(x+w) < .2*h and abs(b[1]-y) < .12*h
                       and .75 < b[3]/h < 1.25 and .48 < b[2]/b[3] < .85]
            if mask is not None:
                # A card's thin left border can resemble a leading 1. Only a
                # closed zero may complete 10; J/K/2 beside a border must stay
                # independent glyphs rather than becoming an unreadable blob.
                def closed_zero(box):
                    if rank_bank is not None:
                        _,distance,margin=classify(crop_mask(mask,box),rank_bank)
                        if distance<.25 and margin>.04:
                            return False
                    contours, hierarchy = cv2.findContours(crop_mask(mask,box),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
                    return hierarchy is not None and any(
                        hierarchy[0][k][3]>=0 and cv2.contourArea(c)>.04*box[2]*box[3]
                        for k,c in enumerate(contours))
                options=[(j,b) for j,b in options if closed_zero(b)]
            if options:
                j,(xx,yy,ww,hh) = min(options, key=lambda z: abs(z[1][0]-(x+w)))
                result.append((min(x,xx),min(y,yy),max(x+w,xx+ww)-min(x,xx),max(y+h,yy+hh)-min(y,yy)))
                used.update([i,j]); continue
        result.append((x,y,w,h)); used.add(i)
    return result


def bank_from_reviewed():
    cached = ROOT / 'configs/card_glyph_bank.npz'
    if cached.exists():
        # Checked-in visual templates make proposal/regression tools usable
        # without the historical candidate folders. No pickle data is loaded.
        with np.load(cached, allow_pickle=False) as source:
            banks = {kind: list(zip(source[kind + '_names'].tolist(), source[kind + '_masks']))
                     for kind in ('rank', 'suit')}
            banks['joker'] = [(source[f'joker_{i}'], bool(own))
                              for i, own in enumerate(source['joker_own'])]
        return banks
    rgb = np.asarray(Image.open(ROOT/'data/raw/tutorial001_frame0002.jpg').convert('RGB'))
    doc = json.loads((ROOT/'data/annotations/tutorial001_frame0002.json').read_text())
    banks = {'rank':[], 'suit':[]}
    for item in doc['shapes']:
        label = item['label']
        if label in ('SJ','BJ') or label in ('4♥','3♥','2♥'): continue
        x1=round(min(p[0] for p in item['points'])); y1=round(min(p[1] for p in item['points']))
        visible_height=round(max(p[1] for p in item['points'])-y1)
        patch=rgb[y1:y1+min(61,visible_height-3),x1:x1+100]
        mask=ink_mask(patch)
        mask[:2]=0;mask[-2:]=0;mask[:,:2]=0
        rank_mask=mask[:,:50].copy();suit_mask=mask[:,50:100].copy()
        for kind,candidate,name in [('rank',rank_mask,label[:-1]),('suit',suit_mask,label[-1])]:
            valid=components(candidate,25,61)
            if not valid:continue
            if kind=='rank' and name=='10':
                clean=np.zeros_like(candidate)
                for box in valid:
                    x,y,w,h=box;clean[y:y+h,x:x+w]|=candidate[y:y+h,x:x+w]
            else:
                clean=crop_mask(candidate,max(valid,key=lambda b:b[2]*b[3]))
            banks[kind].append((name,normalize(clean)))
    # Visually identified A-spade at the start of this current mini-program round.
    folder=ROOT/'data/candidates/live_20261017_01'
    first=sorted(folder.glob('*.png'))[5]
    image=np.asarray(Image.open(first).convert('RGB'))
    mask=ink_mask(image)
    # These glyphs are independently read from the unobscured starting hand.
    # Store separate rank/suit shapes so geometry never invents a card label.
    rank_boxes={'A':(308,609,49,58),'K':(407,753,47,59),
        'Q':(504,752,44,61),'J':(604,681,35,59),'10':(696,681,51,59),
        '9':(800,681,35,59),'8':(898,681,36,59),'7':(999,753,33,59),
        '6':(1095,609,35,58),'5':(1194,753,33,59),'4':(1290,681,37,59),
        '3':(1388,536,36,59),'2':(211,753,34,59)}
    suit_boxes={'♠':(1438,546,40,44),'♣':(1144,618,40,44),
                '♦':(1144,689,40,47),'♥':(359,765,40,39)}
    banks['suit']=[]
    for label,box in rank_boxes.items():
        banks['rank'].append((label,normalize(crop_mask(mask,box))))
    for label,box in suit_boxes.items():
        banks['suit'].append((label,normalize(crop_mask(mask,box))))
    # Current JOKER artwork differs from the tutorial; use its visible JO letters.
    banks['joker'] = [(mask[753:810,113:139].copy(), True)]
    played=sorted(folder.glob('*.png'))[63]
    played_mask=ink_mask(np.asarray(Image.open(played).convert('RGB')))
    banks['joker'].append((played_mask[238:286,764:785].copy(), False))
    red_own=ink_mask(np.asarray(Image.open(sorted(folder.glob('*.png'))[43]).convert('RGB')))
    banks['joker'].append((red_own[753:810,113:139].copy(), True))
    banks['suit'].append(('♠',normalize(red_own[361:390,1283:1309])))
    wild=ink_mask(np.asarray(Image.open(sorted(folder.glob('*.png'))[35]).convert('RGB')))
    banks['suit'].append(('♥',normalize(wild[361:388,392:420])))
    return banks


def classify(mask, bank):
    sample=normalize(mask)
    scores={}
    for label,template in bank:
        union=np.count_nonzero(sample|template)
        distance=1-np.count_nonzero(sample&template)/max(1,union)
        scores[label]=min(scores.get(label,1),distance)
    ordered=sorted(scores.items(),key=lambda v:v[1])
    if not ordered:return ('?',1,0)
    return ordered[0][0],ordered[0][1],(ordered[1][1]-ordered[0][1] if len(ordered)>1 else 1)


PADDING={'A':7,'K':8,'Q':7,'J':9,'10':3,'9':9,'8':9,'7':12,'6':10,'5':11,'4':9,'3':9,'2':8}


def propose(image_path, bank):
    rgb=np.asarray(Image.open(image_path).convert('RGB'))
    height,width=rgb.shape[:2]
    if (width,height)!=(1620,1000):raise ValueError('Draft glyph matching currently expects native 1620x1000 captures')
    mask=ink_mask(rgb)
    mask[:230]=0;mask[813:]=0;mask[:,:50]=0;mask[:,1525:]=0
    boxes=merge_tens(components(mask,20,80),mask,bank['rank'])
    glyphs=[]
    for box in boxes:
        x,y,w,h=box
        patch=rgb[max(0,y-5):min(height,y+h+5),max(0,x-5):min(width,x+w+5)].astype(np.int16)
        neutral=(patch.min(2)>145)&(patch.max(2)-patch.min(2)<65)
        if neutral.mean()<.12:continue
        rank=classify(crop_mask(mask,box),bank['rank'])
        suit=classify(crop_mask(mask,box),bank['suit'])
        glyphs.append(dict(box=box,rank=rank,suit=suit))
    cards=[]
    for glyph in glyphs:
        rank,dist,margin=glyph['rank'];x,y,w,h=glyph['box']
        if dist>.37 or margin<.035 or h<35:continue
        pairs=[]
        for other in glyphs:
            suit,sd,sm=other['suit'];sx,sy,sw,sh=other['box']
            if sd>.33 or sm<.03 or other is glyph:continue
            horizontal=(-8<=sx-(x+w)<=32 and abs(sy-y)<.45*h and h>=47 and .45<sh/h<1.05)
            vertical=(sy>=y+h*.8 and sy<y+h*1.9 and abs((sx+sw/2)-(x+w/2))<h*.6 and h<=50)
            if horizontal or vertical:pairs.append((sd+dist,other,horizontal))
        if not pairs:continue
        score,other,own=min(pairs,key=lambda v:v[0])
        label=rank+other['suit'][0]
        if own:
            cx=x-PADDING[rank];cy=y-(6 if rank=='Q' else 8)
            bw,bh=132,174
        else:
            cx=x-5;cy=y-5;bw,bh=103,137
        cards.append(dict(label=label,x=cx,y=cy,w=bw,h=bh,own=own,rank_distance=dist,suit_distance=other['suit'][1]))
    # Vertical JOKER word: JO stays visible even on stacked cards. Match shape,
    # then distinguish small/big joker using the actual letter color.
    for template,own in bank['joker']:
        result=cv2.matchTemplate(mask.astype(np.float32),template.astype(np.float32),cv2.TM_CCOEFF_NORMED)
        for _ in range(8):
            _,best,_,point=cv2.minMaxLoc(result)
            if best<.84:break
            xx,yy=point;th,tw=template.shape
            result[max(0,yy-th//2):yy+th//2+1,max(0,xx-tw//2):xx+tw//2+1]=0
            if yy<230 or yy>790 or xx<50 or xx>1520:continue
            actual=rgb[yy:yy+th,xx:xx+tw].astype(np.int16)
            active=template.astype(bool)
            red=((actual[:,:,0]>actual[:,:,1]*1.4)&(actual[:,:,0]>actual[:,:,2]*1.4))[active].mean()
            label='BJ' if red>.4 else 'SJ'
            card_own=own or yy>520
            cards.append(dict(label=label,x=xx-(8 if card_own else 7),y=yy-(8 if card_own else 7),w=132 if card_own else 103,
                              h=174 if card_own else 137,own=card_own,rank_distance=1-best,suit_distance=0))
    # Remove same-corner duplicates and trim to visible strips in hand stacks.
    unique=[]
    for card in sorted(cards,key=lambda c:c['rank_distance']+c['suit_distance']):
        if any(abs(card['x']-p['x'])<15 and abs(card['y']-p['y'])<15 for p in unique):continue
        unique.append(card)
    for card in unique:
        if card['own']:
            lower=[p for p in unique if p['own'] and abs(p['x']-card['x'])<16 and p['y']>card['y']+20]
            if lower:card['h']=min(card['h'],min(p['y'] for p in lower)-card['y'])
            right=[p for p in unique if p['own'] and card['x']+20<p['x']<card['x']+card['w']
                   and p['y']<=card['y']+10 and p['y']+p['h']>card['y']]
            if right:card['w']=min(card['w'],min(p['x'] for p in right)-card['x'])
        else:
            right=[p for p in unique if not p['own'] and card['x']+15<p['x']<card['x']+card['w'] and abs(p['y']-card['y'])<15]
            if right:card['w']=min(p['x'] for p in right)-card['x']
    return rgb,sorted(unique,key=lambda c:(c['y'],c['x']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images',type=Path,nargs='+')
    parser.add_argument('--skip-existing',action='store_true',help='Leave existing proposal JSON and previews unchanged')
    parser.add_argument('--output',type=Path,default=ROOT/'data/proposals/live_20261017_01')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    bank=bank_from_reviewed()
    print('Verified glyph template labels:',{k:sorted({x[0] for x in v}) for k,v in bank.items() if k!='joker'})
    for image_path in args.images:
        if args.skip_existing and (args.output/(image_path.stem+'.json')).exists(): continue
        image_path=image_path.resolve();rgb,cards=propose(image_path,bank)
        shapes=[]
        overlay=Image.fromarray(rgb);draw=ImageDraw.Draw(overlay)
        for c in cards:
            x,y,w,h=c['x'],c['y'],c['w'],c['h']
            shapes.append(dict(label=c['label'],points=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]],
                group_id=None,description='DRAFT: requires visual review',shape_type='rectangle',flags={'occluded':h<174 or w<132},
                score=round(1-max(c['rank_distance'],c['suit_distance']),4)))
            draw.rectangle((x,y,x+w,y+h),outline='lime',width=2)
            draw.text((x+1,y+2),c['label'],fill='blue')
        import os
        doc=dict(version='4.1.3',flags={},shapes=shapes,imagePath=os.path.relpath(image_path,args.output.resolve()),
                 imageData=None,imageHeight=rgb.shape[0],imageWidth=rgb.shape[1],checked=False,
                 description='Unreviewed template-based proposals, not training labels')
        (args.output/(image_path.stem+'.json')).write_text(json.dumps(doc,ensure_ascii=False,indent=2))
        overlay.save(args.output/(image_path.stem+'_preview.jpg'))
        print(image_path.name,len(cards),'draft targets')


if __name__=='__main__':main()
