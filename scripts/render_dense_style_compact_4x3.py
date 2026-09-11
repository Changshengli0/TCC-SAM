from pathlib import Path
import csv, json
import cv2, numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('Agg'); mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial'})
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec

ROOT=Path('/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main'); OUT=ROOT/'outputs/paper_figs/openworldsam_tccsam_full_model_response'
IDS=['refcocog_val_umd_69231_p001_07a73e714a','refcocog_val_umd_325545_p001_4e3dda118f']

def row(sid):
 p=ROOT/'outputs/paper_figs/openworldsam_tccsam_contrastive_feature/candidate_prediction_screening.csv'; return next(r for r in csv.DictReader(p.open()) if r['sample_id']==sid)
def mask(path): return np.asarray(Image.open(path).convert('L'))>127
def crop_box(t,d,shape,context=.20):
 h,w=shape; u=t|d; ys,xs=np.where(u); x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
 bw,bh=(x1-x0)*(1+context),(y1-y0)*(1+context); cw=max(bw,bh*4/3); ch=cw*3/4
 if cw>w: cw=w; ch=cw*3/4
 if ch>h: ch=h; cw=ch*4/3
 cx,cy=(x0+x1)/2,(y0+y1)/2; xa,ya=cx-cw/2,cy-ch/2
 xa=max(0,min(xa,w-cw)); ya=max(0,min(ya,h-ch)); xb,yb=xa+cw,ya+ch
 return [int(round(xa)),int(round(ya)),int(round(xb)),int(round(yb))]
def restore_response(r,shape):
 h,w=shape; side=max(h,w); up=cv2.resize(r.astype('float32'),(side,side),interpolation=cv2.INTER_LINEAR)
 px=(side-w)//2; py=(side-h)//2
 return up[py:py+h,px:px+w]
def load(sid):
 q=row(sid); im=Image.open(q['raw_path']).convert('RGB'); t=mask(q['target_mask_path']); d=mask(ROOT/'outputs/paper_figs/openworldsam_tccsam_contrastive_feature/regions'/f'{sid}_distractor_annotation.png'); z=np.load(OUT/f'{sid}_full_model_responses.npz',allow_pickle=False); box=crop_box(t,d,im.size[::-1]); x0,y0,x1,y1=box
 def cut(a): return a[y0:y1,x0:x1]
 return {'sid':sid,'image':np.asarray(im)[y0:y1,x0:x1],'target':cut(t),'distractor':cut(d),'oe':cut(restore_response(z['open_encoder_response'],im.size[::-1])),'od':cut(restore_response(z['open_decoder_response'],im.size[::-1])),'te':cut(restore_response(z['tcc_encoder_response'],im.size[::-1])),'td':cut(restore_response(z['tcc_decoder_response'],im.size[::-1])),'box':box,'original_size':list(im.size)}
def contour(ax,s):
 ax.contour(s['target'].astype(float),[.5],colors='white',linewidths=.40,alpha=.85); ax.contour(s['distractor'].astype(float),[.5],colors='#d0d0d0',linewidths=.35,linestyles='--',alpha=.75)
def render(samples,filename,pdf=False,interp='nearest'):
 ev=np.concatenate([np.r_[s['oe'].ravel(),s['te'].ravel()] for s in samples]); dv=np.concatenate([np.r_[s['od'].ravel(),s['td'].ravel()] for s in samples]); ep=np.percentile(ev,[2,98]); dp=np.percentile(dv,[2,98]); ne=PowerNorm(.65,vmin=ep[0],vmax=ep[1],clip=True); nd=PowerNorm(.65,vmin=dp[0],vmax=dp[1],clip=True)
 fig=plt.figure(figsize=(9,3.35),facecolor='white'); gs=GridSpec(2,5,figure=fig,wspace=.008,hspace=.012,left=.015,right=.945,top=.875,bottom=.025); titles=['(a) Input & GT','(b) Open Encoder','(c) Open Decoder','(d) TCC Encoder','(e) TCC Decoder']; keys=['image','oe','od','te','td']
 for i,s in enumerate(samples):
  for j,k in enumerate(keys):
   ax=fig.add_subplot(gs[i,j]); ax.set_aspect('equal'); ax.set_axis_off();
   if k=='image': ax.imshow(s[k],interpolation='bilinear')
   else: ax.imshow(s[k],cmap='inferno',norm=ne if k in ('oe','te') else nd,interpolation=interp)
   contour(ax,s)
   if i==0: ax.set_title(titles[j],fontsize=8.0,fontweight='semibold',pad=2)
 aa=fig.add_axes([.958,.105,.012,.755]); aa.imshow(np.linspace(0,1,256).reshape(-1,1),cmap='inferno',origin='lower',aspect='auto'); aa.annotate('',xy=(.5,1.015),xytext=(.5,-.015),arrowprops={'arrowstyle':'-|>','color':'white','lw':.5}); aa.text(.5,1.055,'High',ha='center',va='bottom',fontsize=7.5); aa.text(.5,-.055,'Low',ha='center',va='top',fontsize=7.5); aa.set_axis_off()
 base=OUT/filename; fig.savefig(str(base)+'.png',dpi=600,facecolor='white');
 if pdf: fig.savefig(str(base)+'.pdf',facecolor='white')
 plt.close(fig); return ep,dp
def main():
 s=[load(x) for x in IDS]; ep,dp=render(s,'openworldsam_tccsam_dense_style_2x5_compact_4x3_right_arrow',True,'nearest');
 audit={'display_aspect_ratio':'4/3','samples':IDS,'crop_boxes':{x['sid']:x['box'] for x in s},'original_sizes':{x['sid']:x['original_size'] for x in s},'encoder_vmin':float(ep[0]),'encoder_vmax':float(ep[1]),'decoder_vmin':float(dp[0]),'decoder_vmax':float(dp[1]),'cmap':'inferno','gamma':.65,'padding':False,'stretching':False,'response_restored_to_original_coordinates':True}
 (OUT/'dense_style_2x5_compact_4x3_audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
