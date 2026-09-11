from pathlib import Path
import csv, json, textwrap
import cv2, numpy as np
from PIL import Image
import matplotlib as mpl; mpl.use('Agg'); mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial'})
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path('/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main'); OUT=ROOT/'outputs/paper_figs/openworldsam_tccsam_full_model_response'
IDS=['refcocog_val_umd_69231_p001_07a73e714a','refcocog_val_umd_562474_p000_aa13191e8e','refcocog_val_umd_325545_p001_4e3dda118f','refcocog_val_umd_581042_p000_ba67ba21f6']
def mask(z,n):
    for k in (n+'_mask_feature_resolution',n+'_mask_64'):
        if k in z.files:return np.asarray(z[k]).squeeze()>0
    raise KeyError(n)
def row_for(sid):
    p=ROOT/'outputs/paper_figs/openworldsam_tccsam_contrastive_feature/candidate_prediction_screening.csv'
    return next(r for r in csv.DictReader(p.open()) if r['sample_id']==sid)
def load(sid):
    p=OUT/f'{sid}_full_model_responses.npz'
    if sid.startswith('refcocog_val_umd_69231'):
        p=OUT/f'{sid}_full_model_responses.npz'
    with np.load(p,allow_pickle=False) as z:
        return {'sid':sid,'open':z['open_full_model_response'],'tcc':z['tcc_full_model_response'],'diff':z['response_difference'],'target':mask(z,'target'),'distractor':mask(z,'distractor'),'image':np.asarray(Image.open(row_for(sid)['raw_path']).convert('RGB')),'expr':row_for(sid)['expression']}
def draw(rows, suffix, contours_on=True):
    dmax=float(np.percentile(np.abs(np.concatenate([r['diff'].ravel() for r in rows])),99)); norm=TwoSlopeNorm(vmin=-dmax,vcenter=0,vmax=dmax)
    fig,ax=plt.subplots(4,4,figsize=(11,12),squeeze=False)
    for i,r in enumerate(rows):
        lo=float(min(r['open'].min(),r['tcc'].min())); hi=float(max(r['open'].max(),r['tcc'].max()))
        for j,(arr,title) in enumerate([(r['image'],'Image & Expression'),(r['open'],'OpenWorldSAM Response'),(r['tcc'],'TCC-SAM Response'),(r['diff'],'Response Difference')]):
            a=ax[i,j]
            if j==0:a.imshow(arr,interpolation='bicubic')
            elif j==3:a.imshow(arr,cmap='coolwarm',norm=norm,interpolation='bicubic')
            else:a.imshow(arr,cmap='magma',vmin=lo,vmax=hi,interpolation='bicubic')
            if contours_on:
                a.contour(r['target'].astype(float),[.5],colors='white',linewidths=.45); a.contour(r['distractor'].astype(float),[.5],colors='#d0d0d0',linewidths=.40,linestyles='--')
            a.axis('off')
            if i==0:a.set_title(title,fontsize=10,fontweight='bold')
            if j==0:a.text(.5,-.05,'\n'.join(textwrap.wrap(r['expr'],38)[:2]),transform=a.transAxes,ha='center',va='top',fontsize=7)
    fig.tight_layout(); base=OUT/f'openworldsam_tccsam_full_model_old_tvm_style{suffix}'; fig.savefig(str(base)+'.png',dpi=600,bbox_inches='tight'); fig.savefig(str(base)+'.pdf',bbox_inches='tight'); plt.close(fig)
def main():
    rows=[load(s) for s in IDS]; draw(rows,''); draw(rows,'_no_contours',False)
    out=[]
    for r in rows:
        t=r['target']; d=r['distractor']; full_abs=float(np.mean(np.abs(r['diff']))); td=float(r['diff'][t].mean()); dd=float(r['diff'][d].mean())
        old=ROOT/'outputs/paper_figs/openworldsam_tccsam_old_tvm_style/offline_responses'/f'{r["sid"]}.npz'; old_abs=float(np.mean(np.abs(np.load(old)['response_difference']))) if old.exists() else float('nan')
        out.append({'sample_id':r['sid'],'expression':r['expr'],'full_model_mean_abs_difference':full_abs,'block20_mean_abs_difference':old_abs,'target_difference':td,'distractor_difference':dd,'target_distractor_improvement':td-dd,'group_encoder_weight':1/3,'group_neck_fusion_weight':1/3,'group_decoder_weight':1/3})
    with (OUT/'full_model_response_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    report=['# Full-model old TVM-style response report','',f'- All four samples completed: **{len(rows)==4}**.','- Formal response: target cosine minus distractor cosine from saved spatial features.','- OpenWorldSAM and TCC-SAM use shared per-stage robust normalization parameters.','- Encoder, Neck/Fusion, and Decoder groups use fixed weights 1/3, 1/3, 1/3; group members are averaged before group fusion.','- Response panels use magma and bicubic interpolation; differences use coolwarm with a global symmetric zero-centered range.','- Target contours are white solid (0.45 pt); distractor contours are light-gray dashed (0.40 pt).','- No LayerCAM, mask probability, binary prediction mask, training, full evaluation, model, checkpoint, or fusion-weight changes were used.','', '## Samples and cached stages','']
    for r in rows:
        m=next(x for x in out if x['sample_id']==r['sid']); infos=[]
        for method in ('openworldsam','tcc_sam'):
            meta=json.loads((OUT/'raw_features'/r['sid']/method/'feature_metadata.json').read_text())
            infos.append(f"{method} prediction IoU={meta.get('reference_iou')}; selected decoder row={meta.get('selected_candidate_index_for_audit_only')}")
        report += [f"### {r['sid']}",f"Expression: {r['expr']}",*infos,f"Full mean absolute difference: {m['full_model_mean_abs_difference']:.6f}; block-20 comparison: {m['block20_mean_abs_difference']:.6f}.",f"Target difference: {m['target_difference']:.6f}; distractor difference: {m['distractor_difference']:.6f}; target-distractor improvement: {m['target_distractor_improvement']:.6f}.",'Captured per method: encoder_stage1–4, neck_scale0–7, decoder_spatial.','']
    report += ['## Interpretation','', 'The full-model comparison is reported numerically against the pre-existing block-20 offline response where available; no claim is made that the method-level difference is caused by a single TVM component.','', 'The work did not train, run full-dataset evaluation, or modify the model/checkpoints.']
    (OUT/'full_model_old_tvm_style_report.md').write_text('\n'.join(report)+'\n')
if __name__=='__main__':main()
