from pathlib import Path
import csv, json
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('Agg'); mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial'})
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

ROOT=Path('/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main'); OUT=ROOT/'outputs/paper_figs/openworldsam_tccsam_full_model_response'
IDS=['refcocog_val_umd_69231_p001_07a73e714a','refcocog_val_umd_325545_p001_4e3dda118f']
def row(sid):
 p=ROOT/'outputs/paper_figs/openworldsam_tccsam_contrastive_feature/candidate_prediction_screening.csv'; return next(r for r in csv.DictReader(p.open()) if r['sample_id']==sid)
def load(sid):
 q=row(sid); im=Image.open(q['raw_path']).convert('RGB'); t=np.asarray(Image.open(q['target_mask_path']).convert('L'))>127; d=np.asarray(Image.open(ROOT/'outputs/paper_figs/openworldsam_tccsam_contrastive_feature/regions'/f'{sid}_distractor_annotation.png').convert('L'))>127
 audit=json.loads((OUT/'dense_style_2x5_compact_4x3_audit.json').read_text()); x0,y0,x1,y1=audit['crop_boxes'][sid]; z=np.load(OUT/f'{sid}_full_model_responses.npz',allow_pickle=False)
 return {'image':np.asarray(im)[y0:y1,x0:x1],'target':t[y0:y1,x0:x1],'distractor':d[y0:y1,x0:x1],'oe':z['open_encoder_response'],'od':z['open_decoder_response'],'te':z['tcc_encoder_response'],'td':z['tcc_decoder_response']}
def main():
 s=[load(x) for x in IDS]; scale=json.loads((OUT/'dense_2x5_color_scale.json').read_text()); ep=[scale['power_nearest']['encoder_vmin'],scale['power_nearest']['encoder_vmax']]; dp=[scale['power_nearest']['decoder_vmin'],scale['power_nearest']['decoder_vmax']]
 # Exact linear robust ranges used by the original favorite linear feature panels.
 ev=np.concatenate([np.r_[x['oe'].ravel(),x['te'].ravel()] for x in s]); dv=np.concatenate([np.r_[x['od'].ravel(),x['td'].ravel()] for x in s]); ep=np.percentile(ev,[2,98]); dp=np.percentile(dv,[2,98]); ne=Normalize(ep[0],ep[1],clip=True); nd=Normalize(dp[0],dp[1],clip=True)
 fig=plt.figure(figsize=(9,3.35),facecolor='white'); gs=GridSpec(2,5,figure=fig,wspace=.008,hspace=.012,left=.015,right=.945,top=.875,bottom=.025); titles=['(a) Input & GT','(b) Encoder Response','(c) Decoder Response','(d) Encoder Response','(e) Decoder Response']; keys=['image','oe','od','te','td']
 for i,x in enumerate(s):
  for j,k in enumerate(keys):
   ax=fig.add_subplot(gs[i,j]); ax.set_axis_off();
   if k=='image':
    ax.imshow(x[k],interpolation='bilinear',aspect='auto'); ax.contour(x['target'].astype(float),[.5],colors='white',linewidths=.40,alpha=.85); ax.contour(x['distractor'].astype(float),[.5],colors='#d0d0d0',linewidths=.35,linestyles='--',alpha=.75)
   else: ax.imshow(x[k],cmap='inferno',norm=ne if k in ('oe','te') else nd,interpolation='nearest',aspect='auto')
   if i==0: ax.set_title(titles[j],fontsize=8.0,fontweight='semibold',pad=2)
 fig.text(.387,.965,'TCC-SAM w/o TVM',ha='center',va='center',fontsize=8.2,fontweight='semibold'); fig.text(.759,.965,'TCC-SAM w/ TVM',ha='center',va='center',fontsize=8.2,fontweight='semibold')
 aa=fig.add_axes([.958,.105,.012,.755]); aa.imshow(np.linspace(0,1,256).reshape(-1,1),cmap='inferno',origin='lower',aspect='auto'); aa.add_patch(FancyArrowPatch((.5,.04),(.5,.96),transform=aa.transAxes,arrowstyle='-|>',mutation_scale=9,color='white',linewidth=1.0,zorder=5,clip_on=False)); aa.set_axis_off(); fig.text(.964,.882,'High',ha='center',va='bottom',fontsize=7.5,color='black'); fig.text(.964,.083,'Low',ha='center',va='top',fontsize=7.5,color='black')
 base=OUT/'openworldsam_tccsam_dense_style_2x5_compact_linear_input_zoom_right_arrow_caption_fixed'; fig.savefig(str(base)+'.png',dpi=600,facecolor='white'); fig.savefig(str(base)+'.pdf',facecolor='white'); plt.close(fig); print({'encoder_range':[float(ep[0]),float(ep[1])],'decoder_range':[float(dp[0]),float(dp[1])],'feature_data_unchanged':True,'arrow':'right','caption_fixed':True})
if __name__=='__main__': main()
