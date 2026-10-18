#!/usr/bin/env python3
"""Make only unreviewed proposal files while native capture remains active."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
while True:
 missing=[x for x in sorted(a.run.glob('*.png')) if not (a.output/(x.stem+'.json')).exists()]
 if missing:
  subprocess.run([sys.executable,str(Path(__file__).with_name('propose_card_annotations.py')),*map(str,missing),'--skip-existing','--output',str(a.output)],check=True)
 run=json.loads((a.run/'run.json').read_text())
 if run.get('status')!='running':break
 time.sleep(10)
