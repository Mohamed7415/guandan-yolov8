#!/usr/bin/env python3
"""Render full-frame annotation proposals for visual review; never approves data."""
import argparse,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('proposals',type=Path);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int,default=100000)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf',20)
 small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf',18)
 docs=[f for f in sorted(a.proposals.glob('*.json')) if a.start<=int(f.stem.rsplit('_',1)[-1])<=a.end]
 for offset in range(0,len(docs),8):
  batch=docs[offset:offset+8];page=Image.new('RGB',(1620,2120),'#eee')
  for i,f in enumerate(batch):
   doc=json.loads(f.read_text());source=(f.parent/doc['imagePath']).resolve()
   im=Image.open(source).convert('RGB');draw=ImageDraw.Draw(im)
   for j,s in enumerate(doc['shapes']):
    x1=min(p[0] for p in s['points']);x2=max(p[0] for p in s['points'])
    y1=min(p[1] for p in s['points']);y2=max(p[1] for p in s['points'])
    draw.rectangle((x1,y1,x2,y2),outline='#00ff00',width=2)
    text=f'{j}:{s["label"]}'
    draw.text((x1+1,y1+2),text,font=font,fill='blue',stroke_width=1,stroke_fill='white')
   col,row=i%2,i//2;x,y=col*810,row*530
   d=ImageDraw.Draw(page);d.text((x+5,y+3),f'{f.stem.rsplit("_",1)[-1]} / {len(doc["shapes"])} boxes',font=small,fill='black')
   page.paste(im.resize((810,500)),(x,y+30))
  name=f'review_{batch[0].stem.rsplit("_",1)[-1]}_{batch[-1].stem.rsplit("_",1)[-1]}.jpg'
  page.save(a.output/name,quality=95)
  print(a.output/name)
if __name__=='__main__':main()
