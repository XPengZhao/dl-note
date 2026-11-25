import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/tmp/dspark-report'); r=json.loads((p/'scalars.json').read_text());s=json.loads((p/'summary.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
base='dspark_block7_qwen3_4b_baseline'
labels={base:'GQA baseline (DeepSpec)','qwen3-4b-gqa-tau':'+ tau loss','qwen3-4b-mla':'MLA','qwen3-4b-mla-swa1024':'MLA + SWA1024','qwen3-4b-mla-swa512':'MLA + SWA512','qwen3-4b-mla-swa128':'MLA + SWA128','qwen3-4b-gqa-context-only':'Context-only','qwen3-4b-gqa-all-mask':'All-mask','qwen3-4b-gqa-aux-3layers':'3 aux layers'}
colors={n:plt.get_cmap('tab10')(i) for i,n in enumerate(labels)}
fig,axs=plt.subplots(1,2,figsize=(12,4.4))
for group,ax in zip([[base,'qwen3-4b-gqa-tau','qwen3-4b-gqa-context-only','qwen3-4b-gqa-all-mask','qwen3-4b-gqa-aux-3layers'],[base,'qwen3-4b-mla','qwen3-4b-mla-swa1024','qwen3-4b-mla-swa512','qwen3-4b-mla-swa128']],axs):
 for n in group:
  vals=r[n]['train/tau_probabilistic']; xs=sorted(int(x) for x in vals if int(x)<=2610); y=np.array([vals[str(x)] for x in xs]); smooth=np.convolve(y,np.ones(10)/10,'valid')
  ax.plot(xs[9:],smooth,label=labels[n],color=colors[n])
 ax.set(xlabel='Optimizer step',ylabel='Training probabilistic tau');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Qwen3-4B DSpark: 10-point trailing means (100 steps)');fig.tight_layout();fig.savefig(p/'tau-curves.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.4))
for group,ax in zip([['qwen3-4b-gqa-tau','qwen3-4b-gqa-context-only','qwen3-4b-gqa-all-mask','qwen3-4b-gqa-aux-3layers'],['qwen3-4b-mla','qwen3-4b-mla-swa1024','qwen3-4b-mla-swa512','qwen3-4b-mla-swa128']],axs):
 for n in group:
  y=[100*(s[n][f'train/accept_rate@{i}']-s[base][f'train/accept_rate@{i}']) for i in range(7)]
  ax.plot(range(1,8),y,'o-',label=labels[n],color=colors[n],ms=4)
 ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Draft prediction position (1-based)',ylabel='Overlap difference vs baseline (pp)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Position-level overlap: mean over steps 2420-2610');fig.tight_layout();fig.savefig(p/'position-deltas.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.4))
for n,label,col in [(base,'DeepSpec baseline','black'),('qwen3-4b-deepspec-baseline','SpecForge baseline','tab:blue'),('qwen3-4b-gqa-tau','SpecForge + tau loss','tab:orange')]:
 for ax,tag in zip(axs,['train/ce_loss','train/l1_loss']):
  v=r[n][tag];x=sorted(int(k) for k in v if 100<=int(k)<=2610);y=[v[str(k)] for k in x];ax.plot(x,y,label=label,color=col,alpha=.8,lw=1)
for ax,tag in zip(axs,['CE loss','Distribution L1 loss']):ax.set(xlabel='Optimizer step',ylabel=tag);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Component losses (raw logged values; first 100 steps excluded)');fig.tight_layout();fig.savefig(p/'component-losses.png');plt.close(fig)
