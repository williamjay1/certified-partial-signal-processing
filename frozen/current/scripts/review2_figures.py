"""Scientific figures from frozen outputs; no fitting or model selection."""
from pathlib import Path
import sys, os, json, csv, hashlib
ROOT=Path(r'D:\MLWork\ASC_CertifiedProcessing_20261002')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache'/'matplotlib')
for folder in (r'D:\ptenv\Lib\site-packages',r'D:\MLWork\ASC_SelectiveReplanning_20260930\env_packages'):
    sys.path.insert(0,folder)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
OUT=ROOT/'manuscript'/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
BLUE='#2864A0'; RED='#B34135'; GREEN='#37806D'; GRAY='#676767'
sources=[]; figures=[]
def read(p):
    p=ROOT/p;sources.append(p);return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    p=ROOT/p;sources.append(p)
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def save(fig,name):
    for ext in ('svg','pdf','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=900,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}_preview.png',dpi=130,bbox_inches='tight',facecolor='white')
    figures.append(name);plt.close(fig)
data=read(Path('results/final_analysis/har_locked_v1/heldout_analysis.json'))
lat=rows(Path('results/review2_latency_analysis/saved_output_v1/paired_call_data.csv'))
prof=read(Path('results/review2_latency_analysis/component_profile_v1/component_profile.json'))
def get(name,delta=None):
    return next(x for x in data['pooled'] if x['name']==name and x['delta']==delta)

# 1. Actual online boundary and decision semantics.
fig,ax=plt.subplots(figsize=(10.6,3.8));ax.set_xlim(0,11);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,h,txt,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',edgecolor=color,facecolor='white',linewidth=1.3))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=9)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':GRAY,'lw':1.3})
box(.2,2.35,1.65,1.0,'Resident\nsignal window\nBlocks uncomputed')
box(2.25,2.35,1.8,1.0,'Execute next batch\nstatistics / FFT\n+ frozen scaling')
box(4.45,2.35,1.8,1.0,'Update leaf masks\nscore enclosure\nprobability intervals')
box(6.65,2.35,1.95,1.0,'Dual lower distances\nsimplex projection\nemitted-output bound')
box(9.05,2.35,1.65,1.0,'Bound ≤ tolerance?\nYes: return p',GREEN)
for a,b in (((1.85,2.85),(2.25,2.85)),((4.05,2.85),(4.45,2.85)),((6.25,2.85),(6.65,2.85)),((8.6,2.85),(9.05,2.85))):arrow(a,b)
box(6.6,.55,4.05,.8,'No: continue\nAll used blocks known → return exact q',RED)
arrow((9.87,2.35),(9.87,1.35));arrow((6.6,.95),(3.15,.95));arrow((3.15,.95),(3.15,2.35))
ax.text(.25,.2,'Complete timer: signal operators + enclosure + output/certificate + control + exact fallback',fontsize=10)
save(fig,'figure1_workflow')

# 2. Fidelity-time curve; normalization uses each locked full baseline.
tol=[0,.01,.025,.05,.1]
har=[get('tight_minimax',d) for d in tol]
hx=[x['timed_cohort_risk_excess'] for x in har];hy=[1-x['time_reduction'] for x in har]
hyd=read(Path('results/hydraulic_external_confirmation/20261002T061642Z_3208a13f/executed_results.json'))
hydrows=[next(x for x in hyd['all_methods'] if x['method']=='tight_minimax' and x['delta']==d) for d in tol]
hydfull=next(x for x in hyd['all_methods'] if x['method']=='compiled_full')['complete_ms']['mean']
hydx=[x['mean_reference_loss_excess'] for x in hydrows]
hydy=[x['complete_ms']['mean']/hydfull for x in hydrows]
fig,axes=plt.subplots(1,2,figsize=(9.0,3.6))
for ax,x,y,title in zip(axes,(hx,hydx),(hy,hydy),('a  Smartphone HAR: same 900 windows','b  Hydraulic: 435 cycles')):
    ax.plot(x,y,'o-',color=BLUE,label='Certified minimax')
    ax.scatter([0],[1],color=GRAY,marker='s',label='Compiled full',zorder=5)
    for d,xx,yy in zip(tol,x,y):ax.annotate(f'δ={d:g}',(xx,yy),xytext=(5,-13 if d==.01 else 5),textcoords='offset points',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':.15})
    ax.axhline(1,color=GRAY,ls=':',lw=1)
    ax.set(title=title,xlabel='Observed-label mean half-Brier increase',ylabel='Complete mean time / compiled full')
    ax.set_xlim(-.001,max(x)*1.32);ax.grid(alpha=.18)
    ax.xaxis.set_major_locator(MaxNLocator(4))
axes[0].legend(frameon=False,fontsize=8,loc='lower left');fig.tight_layout();save(fig,'figure2_fidelity_time')

# 3. Explicit different denominators: timed saving, all-test subject fidelity.
sub=data['subject_primary'];ids=[x['subject'] for x in sub];pos=np.arange(len(ids))
fig,axes=plt.subplots(2,1,figsize=(7.2,5.0),sharex=True)
axes[0].bar(pos,[100*x['mean_time_reduction'] for x in sub],color=BLUE,width=.65)
axes[0].axhline(0,color=GRAY,lw=.8);axes[0].set(ylabel='Mean time reduction (%)',title='a  Same 100 timed windows per subject; three systems averaged')
axes[1].plot(pos,[x['full_risk_excess'] for x in sub],'o-',color=GREEN)
axes[1].set(ylabel='Mean half-Brier increase',xlabel='Held-out subject',title='b  All released test windows for each subject; three systems averaged')
axes[1].set_xticks(pos,ids);axes[1].set_ylim(bottom=0)
for ax in axes:ax.grid(axis='y',alpha=.18)
fig.tight_layout();save(fig,'figure3_subject_variation')

# 4. Survival plot uses all 2700 executions/system, including 72-ms call.
fig,axes=plt.subplots(1,3,figsize=(10.3,3.4),sharey=True)
for ax,seed in zip(axes,(17,29,43)):
    subset=[x for x in lat if int(x['seed'])==seed]
    for key,label,color in (('full_ms','Compiled full',GRAY),('partial_ms','Certified partial',BLUE)):
        a=np.sort([float(x[key]) for x in subset]);surv=(len(a)-np.arange(len(a)))/len(a)
        ax.step(a,surv,where='pre',label=label,color=color,lw=1.2)
        ax.plot(a[-1],surv[-1],'.',color=color,markersize=3)
    for yy,tag in ((.5,'P50'),(.05,'P95'),(.01,'P99')):
        ax.axhline(yy,color=RED if yy==.01 else GRAY,ls=':',lw=.7,alpha=.8)
        ax.text(95,yy*1.12,tag,ha='right',va='bottom',fontsize=7,color=GRAY)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.4,100);ax.set_ylim(1/3000,1.1)
    ax.set(title=f'System {seed}',xlabel='Complete execution time (ms)');ax.grid(alpha=.15)
axes[0].set_ylabel('Fraction of calls at or above time');axes[0].legend(frameon=False,fontsize=8,loc='lower left')
fig.tight_layout();save(fig,'figure4_latency_tails')

# 5. Separate instrumented diagnostic; nonoverlapping spans plus signed residual.
keys=['feature_ms','leaf_enclosure_ms','probability_interval_ms','output_certificate_ms','exact_fallback_ms','control_ms','state_setup_ms','unassigned_instrumentation_residual_ms']
labels=['Signal operators + scaling','Leaf update + score bounds','Probability intervals','Output + certificate','Exact prediction at terminal','Control','State setup','Signed residual']
colors=[BLUE,'#8BA9C9',GREEN,'#AAD1B9',RED,'#E1B89A',GRAY,'#BCBCBC']
prows=[next(x for x in prof['aggregate'] if x['seed']==seed and x['method']==m) for seed in (17,29,43) for m in ('instrumented_compiled_full','instrumented_tight_primary')]
fig,ax=plt.subplots(figsize=(9.4,4.1));left=np.zeros(6);yy=np.arange(6)
for k,l,c in zip(keys,labels,colors):
    vals=np.array([x['component_mean_ms'][k] for x in prows]);assert np.all(vals>=0),'Signed negative residual needs diverging display'
    ax.barh(yy,vals,left=left,color=c,label=l,height=.7);left+=vals
assert np.allclose(left,[x['complete_ms']['mean'] for x in prows],atol=2e-12)
ax.set_yticks(yy,[f'{s} — {m}' for s in (17,29,43) for m in ('full','partial')]);ax.invert_yaxis()
ax.set(xlabel='Instrumented mean time (ms)',title='36 windows × three systems × three repetitions; diagnostic only')
ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False,fontsize=8);ax.grid(axis='x',alpha=.15)
fig.tight_layout();save(fig,'figure5_component_cost')

# 6. Stopping controls: faster mean time is not the same guarantee.
methods=[('tight_minimax',.025,'Simplex minimax'),('coord_minimax',.025,'Coordinate minimax'),('midpoint_tight',.025,'Certified midpoint'),('uniform_A_tight',.025,'Uniform checkpoints'),('score_separation',None,'Class preservation'),('fixed_prefix',None,'Fixed prefix'),('entropy',None,'Entropy')]
print('Available control keys:',[(x['name'],x['delta']) for x in data['pooled']])
# Actual recorded keys are mapped explicitly; a missing key fails rather than inventing data.
aliases={'coord_minimax':'coordinate_minimax','midpoint_tight':'tight_midpoint','uniform_A_tight':'uniform_A_tight_minimax','score_separation':'score_separation_argmax','fixed_prefix':'fixed_prefix_mean_completion','entropy':'entropy_mean_completion'}
valid={(x['name'],x['delta']):x for x in data['pooled']}
records=[]
for n,d,l in methods:
    if (n,d) not in valid:n=aliases.get(n,n)
    if (n,d) not in valid:raise KeyError((n,d))
    records.append((valid[n,d],l))
fig,axes=plt.subplots(1,2,figsize=(9.8,4.2),sharey=True);yy=np.arange(len(records))
axes[0].barh(yy,[x['complete_mean_ms'] for x,l in records],color=[BLUE]*4+[GRAY]*3)
axes[1].barh(yy,[100*x['delta025_every_label_violations']/8841 for x,l in records],color=[BLUE]*4+[RED]*3)
axes[0].set_yticks(yy,[l for x,l in records]);axes[0].invert_yaxis()
axes[0].set(title='a  Same 900 windows',xlabel='Complete mean time (ms)')
axes[1].set(title='b  All 2947 windows × three systems',xlabel='Every-label tolerance exceedances (%)')
for ax in axes:ax.grid(axis='x',alpha=.15)
fig.tight_layout();save(fig,'figure6_stopping_controls')
manifest={'stage':'FIGURES_FROM_EXECUTED_OUTPUTS','figures':figures,'raster_dpi':900,'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dict.fromkeys(sources)],'note':'Primary and diagnostic scopes separated in captions. No inferred samples or new fits.'}
(OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Saved',len(figures),'figures to',OUT)
