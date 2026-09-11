from pathlib import Path
import argparse, csv, json
import cv2, numpy as np
from PIL import Image
import matplotlib as mpl; mpl.use('Agg')
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial'})
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path('/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main')
OUT=ROOT/'outputs/paper_figs/openworldsam_tccsam_full_model_response'; EPS=1e-6

def hwc(x):
    x=np.asarray(x)
    return x if x.ndim==3 and x.shape[-1] in (96,144,288,576) else np.moveaxis(x,0,-1)
def mask_field(z, name):
    for k in (name+'_mask_feature_resolution', name+'_mask_64'):
        if k in z.files: return np.asarray(z[k]).squeeze()>0, k
    raise KeyError((name,z.files))
def response(f,t,d):
    f=hwc(f).astype('float32'); h,w,_=f.shape
    t=cv2.resize(t.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
    d=cv2.resize(d.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
    fn=f/(np.linalg.norm(f,axis=-1,keepdims=True)+EPS)
    def proto(m):
        p=fn[m].mean(0); return p/(np.linalg.norm(p)+EPS)
    return np.einsum('hwc,c->hw',fn,proto(t))-np.einsum('hwc,c->hw',fn,proto(d))
def group(raw, names, t, d):
    oo=[]; tt=[]; norm={}
    for n in names:
        a=response(np.load(raw/'openworldsam'/f'{n}.npy'),t,d); b=response(np.load(raw/'tcc_sam'/f'{n}.npy'),t,d)
        j=np.r_[a.ravel(),b.ravel()]; c=float(np.median(j)); s=float(np.percentile(np.abs(j-c),95)+EPS)
        norm[n]={'center':c,'scale_p95':s}
        oo.append(cv2.resize(np.clip((a-c)/s,-3,3),(64,64),interpolation=cv2.INTER_LINEAR))
        tt.append(cv2.resize(np.clip((b-c)/s,-3,3),(64,64),interpolation=cv2.INTER_LINEAR))
    return np.mean(oo,0),np.mean(tt,0),norm
def contours(ax,t,d):
    ax.contour(t.astype(float),[.5],colors='white',linewidths=.45)
    ax.contour(d.astype(float),[.5],colors='#d0d0d0',linewidths=.40,linestyles='--')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sample-id',required=True); a=ap.parse_args(); sid=a.sample_id
    raw=OUT/'raw_features'/sid; rec=ROOT/'outputs/paper_figs/openworldsam_tccsam_contrastive_feature/recovered_responses'/f'{sid}.npz'
    with np.load(rec,allow_pickle=False) as z:
        t,tk=mask_field(z,'target'); d,dk=mask_field(z,'distractor')
    enc=[f'encoder_stage{i}' for i in range(1,5)]; neck=sorted(p.stem for p in (raw/'openworldsam').glob('neck_scale*.npy')); dec=['decoder_spatial']
    oe,te,en=group(raw,enc,t,d); on,tn,nn=group(raw,neck,t,d); od,td,dn=group(raw,dec,t,d)
    of=(oe+on+od)/3; tf=(te+tn+td)/3; diff=tf-of; tm=cv2.resize(t.astype('uint8'),(64,64),interpolation=cv2.INTER_NEAREST).astype(bool); dm=cv2.resize(d.astype('uint8'),(64,64),interpolation=cv2.INTER_NEAREST).astype(bool)
    np.savez_compressed(OUT/f'{sid}_full_model_responses.npz',open_encoder_response=oe,tcc_encoder_response=te,open_neck_fusion_response=on,tcc_neck_fusion_response=tn,open_decoder_response=od,tcc_decoder_response=td,open_full_model_response=of,tcc_full_model_response=tf,response_difference=diff,target_mask_64=tm,distractor_mask_64=dm,normalization=json.dumps({'encoder':en,'neck':nn,'decoder':dn}),group_weights=json.dumps({'encoder':1/3,'neck_fusion':1/3,'decoder':1/3}))
    screening=ROOT/'outputs/paper_figs/openworldsam_tccsam_contrastive_feature/candidate_prediction_screening.csv'; row=next(r for r in csv.DictReader(screening.open()) if r['sample_id']==sid); image=np.asarray(Image.open(row['raw_path']).convert('RGB')); expr=row['expression']
    vmax=float(np.percentile(np.abs(np.r_[of.ravel(),tf.ravel()]),99)); dmax=float(np.percentile(np.abs(diff),99)); norm=TwoSlopeNorm(vmin=-dmax,vcenter=0,vmax=dmax)
    fig,ax=plt.subplots(1,4,figsize=(12,3.35)); panels=[(image,'Image & Expression',None),(of,'OpenWorldSAM Response','magma'),(tf,'TCC-SAM Response','magma'),(diff,'Response Difference','coolwarm')]
    for x,(arr,title,cm) in zip(ax,panels):
        if cm=='coolwarm': x.imshow(arr,cmap=cm,norm=norm,interpolation='bicubic')
        elif cm: x.imshow(arr,cmap=cm,vmin=-vmax,vmax=vmax,interpolation='bicubic')
        else: x.imshow(arr,interpolation='bicubic')
        contours(x,tm,dm); x.set_title(title,fontsize=9,fontweight='bold'); x.axis('off')
    ax[0].text(.5,-.12,expr,transform=ax[0].transAxes,ha='center',va='top',fontsize=7,wrap=True); fig.tight_layout(); fig.savefig(OUT/f'{sid}_full_model_preview.png',dpi=600,bbox_inches='tight'); fig.savefig(OUT/f'{sid}_full_model_preview.pdf',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(8,5)); vals=[oe,on,od,te,tn,td]; titles=['Open encoder','Open neck/fusion','Open decoder','TCC encoder','TCC neck/fusion','TCC decoder']
    for q,x,title in zip(ax.flat,vals,titles): q.imshow(x,cmap='magma',interpolation='bicubic'); contours(q,tm,dm); q.set_title(title,fontsize=9); q.axis('off')
    fig.tight_layout(); fig.savefig(OUT/f'{sid}_stagewise_diagnostic.png',dpi=600,bbox_inches='tight'); plt.close(fig)
    metrics={'sample_id':sid,'target_mask_key':tk,'distractor_mask_key':dk,'group_weights':{'encoder':1/3,'neck_fusion':1/3,'decoder':1/3},'openworldsam_target_mean':float(of[tm].mean()),'openworldsam_distractor_mean':float(of[dm].mean()),'tcc_sam_target_mean':float(tf[tm].mean()),'tcc_sam_distractor_mean':float(tf[dm].mean())}
    metrics['openworldsam_target_distractor_gap']=metrics['openworldsam_target_mean']-metrics['openworldsam_distractor_mean']; metrics['tcc_sam_target_distractor_gap']=metrics['tcc_sam_target_mean']-metrics['tcc_sam_distractor_mean']; metrics['selectivity_gain']=metrics['tcc_sam_target_distractor_gap']-metrics['openworldsam_target_distractor_gap']; metrics['normalization']={'encoder':en,'neck':nn,'decoder':dn}
    (OUT/f'{sid}_full_model_response_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
