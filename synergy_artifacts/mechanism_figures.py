"""Publication-style vector diagnostics from verified aggregate evidence only."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

COLORS=['#246B9B','#CF762D','#5B8B68','#8568A6']


def build(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.titleweight':'bold'})
    figs=[]
    ch=pd.read_csv(root/'prioritization/changed_selections.csv')
    p=ch[(ch.base=='N')&(ch.alternative=='C1')&(ch.budget==.05)]
    fig,ax=plt.subplots(1,2,figsize=(13.3333,7.5));fig.subplots_adjust(left=.08,right=.97,bottom=.19,top=.80,wspace=.29)
    for role,color in [('entered',COLORS[0]),('exited',COLORS[1])]:
        d=p[p.selection==role]
        ax[0].scatter(d.native_signed_margin,d.score_change,s=28,alpha=.8,color=color,label=f'{role.capitalize()} (n={len(d)})')
        x=np.sort(d.native_margin.to_numpy());ax[1].step(np.log10(1+x),np.arange(1,len(x)+1)/len(x),where='post',color=color,label=role.capitalize())
    ax[0].axvline(0,color='grey',lw=.8);ax[0].axhline(0,color='grey',lw=.8)
    ax[0].set(xlabel='Native score minus native cutoff (points)',ylabel='Correction to pair-cell score (points)',title='Entrants can move far from the native cutoff')
    ticks=np.array([0,.5,1,5,10,30]);ax[1].set_xticks(np.log10(1+ticks),[str(x) for x in ticks])
    ax[1].set(xlabel='Absolute native margin to cutoff (points)',ylabel='Cumulative fraction',ylim=(0,1.03),title='Pooling entries and exits hides the asymmetry')
    for a in ax:a.legend(frameon=False);a.grid(alpha=.16)
    fig.suptitle('Loewe prioritization: 67 of 1,137 top-5% selections change',fontsize=19,y=.94)
    fig.text(.08,.065,'C1: recover the original 14,197 admissible fallback roots; identical responses, fits and scoring convention.\nMedian native margin: entrants 3.60 points; exits 0.17 points. Retrospective ranking; biological utility is untested.',fontsize=11)
    figs.append(('selection_margins',fig))
    fig,ax=plt.subplots(1,2,figsize=(13.3333,7.5));fig.subplots_adjust(left=.08,right=.94,bottom=.19,top=.80,wspace=.5)
    b=pd.read_csv(root/'prioritization/budget_stability.csv')
    for j,(name,label) in enumerate([('C1','Original 14,197-root correction'),('I','All 15,620 certified fallback roots')]):
        d=b[b.alternative==name].sort_values('budget');x=d.budget.to_numpy()*100+j*.15
        ax[0].errorbar(x,d.replacement_fraction*100,yerr=np.vstack([(d.replacement_fraction-d.pair_ci_lo)*100,(d.pair_ci_hi-d.replacement_fraction)*100]),marker='o',capsize=3,color=COLORS[j],label=label)
    ax[0].axhline(5,color='grey',lw=.8,ls='--');ax[0].set(xlabel='Screening budget (%)',ylabel='Top-list turnover (%)',title='Effect persists across screening budgets')
    ax[0].set_xticks([1,2,5,10,20])
    ax[0].legend(frameon=False,fontsize=9);ax[0].grid(alpha=.16)
    n=pd.read_csv(root/'native_starts/summary.csv')
    groups=['accepted:boundary','accepted:interior','fallback:boundary','fallback:interior','accepted:undefined_opposite_or_flat']
    starts=['lower_point_one','lower_five','midpoint'];matrix=np.array([[n[(n.stratum==g)&(n.start_rule==s)].acceptance_changes.iloc[0]/50*100 for s in starts] for g in groups])
    im=ax[1].imshow(matrix,cmap='Blues',vmin=0,vmax=70,aspect='auto')
    for i in range(len(groups)):
        for j in range(3):ax[1].text(j,i,f'{matrix[i,j]:.0f}%',ha='center',va='center',color='white' if matrix[i,j]>40 else '#202020')
    ax[1].set_xticks(range(3),['Lower + 0.1','Lower + 5','Midpoint'],rotation=15)
    ax[1].set_yticks(range(len(groups)),['Accepted boundary','Accepted interior','Fallback boundary','Fallback interior','Accepted undefined'])
    ax[1].set_title('Acceptance changes when only Newton start changes')
    fig.suptitle('Independent certificates and native-source interventions',fontsize=19,y=.94)
    fig.text(.08,.065,'Left: whole-drug-pair bootstrap 95% intervals; threshold unchanged. Right: 50 status-stratified cases per row.\nOriginal native path reproduced on all 400 sampled cases. These strata do not estimate population incidence.',fontsize=11)
    figs.append(('numerical_mechanism',fig))
    fig,ax=plt.subplots(1,2,figsize=(13.3333,7.5));fig.subplots_adjust(left=.08,right=.97,bottom=.20,top=.80,wspace=.37)
    d=pd.read_csv(root/'information/error_attribution.csv');arms=['no_drug_identity','no_cell_identity','no_dose']
    positive=np.zeros(3);negative=np.zeros(3)
    for component,color,label in zip(['additive_panel','additive_dose','pair_contrast','dose_contrast'],COLORS,['Additive panel','Additive dose','Pair-cell contrast','Dose-shape contrast']):
        v=d[d.component==component].set_index('mask').loc[arms,'delta_mse'].to_numpy();bottom=np.where(v>=0,positive,negative)
        ax[0].bar(range(3),v,bottom=bottom,color=color,label=label);positive+=np.maximum(v,0);negative+=np.minimum(v,0)
    ax[0].axhline(0,color='grey',lw=.8);ax[0].set_xticks(range(3),['Drug identity','Cell identity','Dose']);ax[0].set(xlabel='Removed direct input',ylabel='Change in response MSE (points²)',title='Most identity cost lies in additive components')
    ax[0].legend(frameon=False,fontsize=9)
    r=pd.read_csv(root/'information/model_reference_residuals.csv');j=0
    for model,label,color in [('neural_F2','Selected neural',COLORS[0]),('gp_F2','GP',COLORS[1]),('cf_selected','Selected CF',COLORS[2])]:
        for contrast,mark,offset in [('panel_reference_residual','o',0),('dose_reference_residual','s',1)]:
            row=r[(r.model==model)&(r.contrast==contrast)].iloc[0]
            ax[1].errorbar(row.r2,j,xerr=[[row.r2-row.drug_low],[row.drug_high-row.r2]],fmt=mark,color=color,capsize=3)
            j+=1
    ax[1].set_yticks(range(6),['Neural: pair-cell','Neural: dose shape','GP: pair-cell','GP: dose shape','CF: pair-cell','CF: dose shape']);ax[1].invert_yaxis();ax[1].axvline(0,color='grey',lw=.8)
    ax[1].set(xlabel='Reference-augmented contrast R²',title='Some dose-shape skill survives reference removal');ax[1].grid(axis='x',alpha=.16)
    fig.suptitle('Response information is different from residual contrast prediction',fontsize=18,y=.94)
    fig.text(.08,.055,'Nested evaluation removes projected training-pair Bliss and HSA spans within cell/fold. Right: delete-one-drug 95% intervals.\nThese spans explain 33.6% of original pair-cell contrast energy and 63.5% of dose-shape contrast energy. No causal synergy claim.',fontsize=10.5)
    figs.append(('information_residuals',fig))
    fig,ax=plt.subplots(1,2,figsize=(13.3333,7.5));fig.subplots_adjust(left=.08,right=.97,bottom=.20,top=.80,wspace=.32)
    e=pd.read_csv(root/'screens/errors.csv');names=['ONEIL','ALMANAC','NCATS_ES','NCATS_DIPG'];x=np.arange(4)
    for j,(m,label,color) in enumerate([('f2','F2 combined',COLORS[0]),('bliss','Training-pair Bliss',COLORS[1])]):
        v=e[e.model==m].set_index('screen').loc[names,'normalized_mse_train_var'];ax[0].bar(x+(j-.5)*.3,v,width=.3,label=label,color=color)
    ax[0].set_xticks(x,["O'Neil",'ALMANAC','NCATS ES','NCATS DIPG'],rotation=15);ax[0].set(ylabel='MSE / training-fold target variance',title='Within-screen, matched seed 2026');ax[0].legend(frameon=False,fontsize=9)
    s=pd.read_csv(root/'screen_support/sensitivities.csv');s=s[s.method=='stable_support_pair_bootstrap']
    for i,row in enumerate(s.itertuples()):
        ax[1].errorbar(row.absolute_gap_reduction*100,i,xerr=[[(row.absolute_gap_reduction-row.reduction_low)*100],[(row.reduction_high-row.absolute_gap_reduction)*100]],fmt='o',color=COLORS[i],capsize=4)
    ax[1].set_yticks(range(3),['ALMANAC','NCATS ES','NCATS DIPG']);ax[1].axvline(0,color='grey',lw=.8);ax[1].invert_yaxis()
    ax[1].set(xlabel='Reduction in absolute gain gap (percentage points)',title='Input standardization does not shrink the gap');ax[1].grid(axis='x',alpha=.16)
    fig.suptitle('Independent-screen heterogeneity remains after the audit',fontsize=19,y=.94)
    fig.text(.08,.055,'Right: exploratory common strata with ≥10 pairs in both screens; pair-bootstrap 95% intervals, 2,000/2,000 valid draws.\nAt least 99.58% of O’Neil mass retained. Negative values mean a wider gain gap. NCATS confirmation remains failed.',fontsize=10.5)
    figs.append(('screen_standardization',fig))
    with PdfPages(out/'mechanism_analysis.pdf') as pdf:
        for name,fig in figs:
            fig.savefig(out/f'{name}.pdf');fig.savefig(out/f'{name}.png',dpi=180);pdf.savefig(fig);plt.close(fig)
    files={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in out.iterdir() if p.suffix in ['.png','.pdf']}
    inputs={p.relative_to(root).as_posix():hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in root.rglob('*') if p.suffix in ['.csv','.json']}
    (out/'manifest.json').write_text(json.dumps(dict(inputs=inputs,artifacts=files),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',default='results/mechanism_analysis')
    p.add_argument('--output',default='reports/figures/mechanism_analysis');a=p.parse_args();build(a.results,a.output)
