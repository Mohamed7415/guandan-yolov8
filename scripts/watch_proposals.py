#!/usr/bin/env python3
"""Make unreviewed proposals only for committed captures, and drain on exit."""
import argparse,csv,json,subprocess,sys,time
from pathlib import Path

def pending_images(run: Path, output: Path) -> list[Path]:
 manifest=run/'captures.csv'
 if not manifest.exists():return []
 with manifest.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
 result=[]
 for row in rows:
  name=row['filename']
  if Path(name).name!=name:raise ValueError('Capture filename must be relative to this run')
  path=run/name
  if not path.is_file():raise ValueError(f'Committed capture is missing: {path}')
  if not (output/(path.stem+'.json')).exists():result.append(path)
 return result

def watch(run: Path, output: Path) -> None:
 while True:
  missing=pending_images(run,output)
  if missing:
   subprocess.run([sys.executable,str(Path(__file__).with_name('propose_card_annotations.py')),*map(str,missing),'--skip-existing','--output',str(output)],check=True)
  status=json.loads((run/'run.json').read_text())['status']
  if status!='running':
   # Drain any last frame committed between the first scan and terminal status.
   if pending_images(run,output):continue
   return
  time.sleep(10)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 watch(a.run,a.output)

if __name__=='__main__':main()
