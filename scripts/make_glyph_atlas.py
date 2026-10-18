#!/usr/bin/env python3
"""Show low-confidence class glyphs as native crops for visual review."""
import argparse,json
from pathlib import Path
from collections import defaultdict
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('proposals',type=Path);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int,default=100000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 bank=defaultdict(list)
 for f in sorted(a.proposals.glob('*.json')):
  index=int(f.stem.rsplit('_',1)[-1])
  if not a.start<=index<=a.end: continue
  doc=json.loads(f.read_text())
  for j,s in enumerate(doc['shapes']):bank[s['label']].append((s.get('score',1),index,j,f,s,doc['imagePath']))
 labels=Path('configs/classes.txt').read_text().splitlines();font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf',22)
 index_rows=[]
 for offset in range(0,len(labels),6):
  page=Image.new('RGB',(1380,960),'#eee');draw=ImageDraw.Draw(page)
  for row,label in enumerate(labels[offset:offset+6]):
   draw.text((4,row*160+5),label+'  n='+str(len(bank[label])),font=font,fill='black')
   # Different image positions and sizes before repeating the same rendering.
   chosen=[];seen=set()
   for item in sorted(bank[label]):
    _,index,j,f,s,path=item;ps=s['points'];x1=int(min(p[0] for p in ps));y1=int(min(p[1] for p in ps));x2=int(max(p[0] for p in ps));y2=int(max(p[1] for p in ps))
    key=(x1//20,y1//20,(y2-y1)//20)
    if key in seen:continue
    seen.add(key);chosen.append(item)
    if len(chosen)==9:break
   for col,item in enumerate(chosen):
    score,index,j,f,s,path=item;ps=s['points'];x1=int(min(p[0] for p in ps));y1=int(min(p[1] for p in ps));x2=int(max(p[0] for p in ps));y2=int(max(p[1] for p in ps))
    im=Image.open((f.parent/path).resolve()).convert('RGB');crop=im.crop((x1,y1,min(x2,x1+132),min(y2,y1+115)))
    crop.thumbnail((132,115));x=150+col*136;y=row*160+32;page.paste(crop,(x,y))
    draw.text((x,row*160+3),f'{index}:{j}',font=font,fill='black')
    index_rows.append({'class':label,'image_index':index,'shape_index':j,'score':score})
  page.save(a.output/f'glyph_{offset//6+1:02d}.jpg',quality=97)
 (a.output/'index.json').write_text(json.dumps(index_rows,ensure_ascii=False,indent=2))
 print('Atlas pages:',(len(labels)+5)//6)
if __name__=='__main__':main()
