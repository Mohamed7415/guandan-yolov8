#!/usr/bin/env python3
"""Package native images and corrected labels into portable release ZIP assets."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
from card_common import CLASS_NAMES

def csv_text(rows):
 s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=rows[0]);w.writeheader();w.writerows(rows);return s.getvalue()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('data/prepared_full2000_corrected'));p.add_argument('--annotations',type=Path,default=Path('data/xanylabeling_full2000/annotations'));p.add_argument('--output',type=Path,default=Path('releases/v1.0.0'));p.add_argument('--part-mib',type=int,default=700);a=p.parse_args()
 if a.part_mib<1:p.error('--part-mib must be positive')
 a.output.mkdir(parents=True,exist_ok=True)
 if any(a.output.glob('dataset-*.zip')):p.error('Dataset archives already exist; use a new output directory')
 source=list(csv.DictReader((a.data/'split_manifest.csv').open()));records=[];annotations=[];labels=[]
 for r in source:
  image=Path(r['source_image']);stem=image.stem;new_image=f"images/{r['split']}/{image.name}";new_label=f"labels/{r['split']}/{stem}.txt"
  doc=json.loads((a.annotations/f'{stem}.json').read_text());doc['imagePath']='../'+new_image;doc['imageData']=None
  # Review descriptions contain no private account/credential material. Remove
  # local absolute filesystem prefixes from descriptive provenance if present.
  project=str(Path.cwd().resolve())+'/'
  def portable(value):
   if isinstance(value,str):return value.replace(project,'')
   if isinstance(value,list):return [portable(x) for x in value]
   if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
   return value
  doc=portable(doc);annotations.append((f'annotations/{stem}.json',json.dumps(doc,ensure_ascii=False,indent=2)+'\n'))
  label=(a.data/r['label']).read_text();labels.append((new_label,label))
  records.append(dict(image=new_image,label=new_label,group_id=r['group_id'],session_id=r['session_id'],scene=r['scene'],reviewed='true',split=r['split'],source_image=new_image,source_label=new_label))
 manifest=[dict(image=r['image'],label='annotations/'+Path(r['image']).stem+'.json',group_id=r['group_id'],scene=r['scene'],reviewed='true',session_id=r['session_id']) for r in records]
 config=dict(train='images/train',val='images/val',test='images/test',nc=54,names=CLASS_NAMES)
 with zipfile.ZipFile(a.output/'dataset-metadata.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name,text in labels+annotations:z.writestr(name,text)
  z.writestr('data.yaml',json.dumps(config,ensure_ascii=False,indent=2)+'\n');z.writestr('split_manifest.csv',csv_text(records));z.writestr('manifest.csv',csv_text(manifest));z.writestr('yolo_manifest.csv',csv_text([{k:r[k] for k in ['image','label','group_id','scene','reviewed','session_id']} for r in records]));z.writestr('classes.txt','\n'.join(CLASS_NAMES)+'\n')
  z.writestr('README.md','# Native 2000-image dataset\n\nExtract dataset-metadata.zip and EVERY dataset-images-NNN.zip into the SAME directory. Run the project validator on this directory/data.yaml. Image bytes and group assignments are preserved; corrected test labels add 13 J-club targets and fix 10 heart/diamond classes. The original training checkpoint used the original frozen labels. This portable revision has 38122 boxes; no retraining or test-threshold tuning followed the correction.\n')
 current=None;part=0;size=0;image_hashes=[]
 try:
  for i,(r,original) in enumerate(zip(records,source)):
   image=Path(original['source_image']);n=image.stat().st_size
   if current is None or size+n>a.part_mib*1024**2:
    if current is not None:current.close()
    part+=1;size=0;current=zipfile.ZipFile(a.output/f'dataset-images-{part:03}.zip','w',zipfile.ZIP_STORED)
    print(f'Image archive {part}',flush=True)
   current.write(image,r['image']);size+=n;image_hashes.append(dict(image=r['image'],sha256=sha(image)))
 finally:
  if current is not None:current.close()
 inventory=dict(images=len(records),boxes=sum(len(t.splitlines()) for _,t in labels),image_parts=part,split_images={s:sum(r['split']==s for r in records) for s in ['train','val','test']},image_bytes_unchanged=True,split_groups_unchanged=True,image_hashes=image_hashes,assets=[])
 for path in sorted(a.output.glob('dataset-*.zip')):
  with zipfile.ZipFile(path) as z:assert z.testzip() is None
  inventory['assets'].append(dict(name=path.name,bytes=path.stat().st_size,sha256=sha(path)))
 (a.output/'dataset-inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in inventory.items() if k not in ['image_hashes','assets']},indent=2),flush=True)
if __name__=='__main__':main()
