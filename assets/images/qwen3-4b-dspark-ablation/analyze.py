from pathlib import Path
import json, hashlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root=Path('/Users/xpzhao/Desktop/omniinfer-dev/tensorboard-logs')
out=Path('/tmp/dspark-report')
runs={};inventory=[]; seen=set()
for f in sorted(root.rglob('*tfevents*')):
 sha=hashlib.sha256(f.read_bytes()).hexdigest()
 if sha in seen: continue
 seen.add(sha)
 ea=EventAccumulator(str(f),size_guidance={'scalars':0});ea.Reload()
 tags=ea.Tags()['scalars']; name=f.parent.parent.name if f.parent.name=='runs' else f.parent.name
 inventory.append(dict(run=name,file=str(f.relative_to(root)),sha256=sha,tags=tags))
 if not tags:continue
 runs[name]={t:{x.step:float(x.value) for x in ea.Scalars(t)} for t in tags}
window=list(range(2420,2611,10))
summary={}
for name,tags in runs.items():
 d={t:float(np.mean([v[s] for s in window])) for t,v in tags.items() if all(s in v for s in window)}
 d['at2610']={t:v[2610] for t,v in tags.items() if 2610 in v}
 if 'train/elapsed_sec' in tags:
  e=tags['train/elapsed_sec'];d['seconds_per_step_steady']=(e[2610]-e[110])/(2610-110)
 summary[name]=d
out.joinpath('summary.json').write_text(json.dumps(summary,indent=2))
out.joinpath('inventory.json').write_text(json.dumps(inventory,indent=2))
out.joinpath('scalars.json').write_text(json.dumps(runs))
for n,d in summary.items():
 print(n, ' '.join(f'{t.split("/")[-1]}={d[t]:.6f}' for t in ['train/loss','train/ce_loss','train/l1_loss','train/acc','train/tau_probabilistic','train/tau_loss','seconds_per_step_steady'] if t in d))
 print('positions',[round(d.get(f'train/accept_rate@{i}',0),5) for i in range(7)])
# Match cross-framework trajectories at identical logged steps, independently of names.
a=runs['dspark_block7_qwen3_4b_baseline'];b=runs['qwen3-4b-deepspec-baseline']
for t in ['train/loss','train/ce_loss','train/l1_loss','train/lr']:
 ss=sorted(set(a[t])&set(b[t]));ss=[s for s in ss if s<=2610]
 diff=np.array([a[t][s]-b[t][s] for s in ss]);print('ALIGN',t,'MAE',abs(diff).mean(),'MAX',abs(diff).max(),'TAIL',summary['dspark_block7_qwen3_4b_baseline'][t]-summary['qwen3-4b-deepspec-baseline'][t])
