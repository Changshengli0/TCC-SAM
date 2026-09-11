from pathlib import Path
import csv, json
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial'})
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.gridspec import GridSpec

ROOT=Path('/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main')
OUT=ROOT/'outputs/paper_figs/openworldsam_tccsam_full_model_response'
IDS=['refcocog_val_umd_69231_p001_07a73e714a','refcocog_val_umd_325545_p001_4e3dda118f']

def info(sid):
    p=ROOT/'outputs/paper_figs/openworldsam_tccsam_contrastive_feature/candidate_prediction_screening.csv'
    return next(r for r in csv.DictReader(p.open()) if r['sample_id']==sid)

def square_image(im, target, fill=(0,0,0)):
    w,h=im.size; scale=min(target/w,target/h); nw,nh=round(w*scale),round(h*scale)
    im=im.resize((nw,nh),Image.Resampling.BILINEAR); out=Image.new('RGB',(target,target),fill)
    out.paste(im,((target-nw)//2,(target-nh)//2)); return np.asarray(out)

def square_mask(m, source_hw, target=64):
    h,w=source_hw; scale=min(target/w,target/h); nw,nh=round(w*scale),round(h*scale)
    x=np.asarray(m).astype(np.uint8); x=np.asarray(Image.fromarray(x).resize((nw,nh),Image.Resampling.NEAREST))
    out=np.zeros((target,target),bool); out[(target-nh)//2:(target-nh)//2+nh,(target-nw)//2:(target-nw)//2+nw]=x>0; return out

def load(sid):
    row=info(sid); im=Image.open(row['raw_path']).convert('RGB'); z=np.load(OUT/f'{sid}_full_model_responses.npz',allow_pickle=False)
    return {'sid':sid,'image':square_image(im,640),'target':square_mask(z['target_mask_64'],im.size[::-1]),'distractor':square_mask(z['distractor_mask_64'],im.size[::-1]),'oe':z['open_encoder_response'],'od':z['open_decoder_response'],'te':z['tcc_encoder_response'],'td':z['tcc_decoder_response']}

def contour(ax,s,alpha):
    ax.contour(s['target'].astype(float),[.5],colors='white',linewidths=.28,alpha=alpha)
    ax.contour(s['distractor'].astype(float),[.5],colors='#d0d0d0',linewidths=.25,linestyles='--',alpha=alpha*.85)

def arrow(fig, bottom, top):
    ax=fig.add_axes([0.044,0.075,0.012,0.845]); g=np.linspace(0,1,256).reshape(-1,1); ax.imshow(g,cmap='inferno',origin='lower',aspect='auto'); ax.annotate('',xy=(.5,1.02),xytext=(.5,-.02),arrowprops={'arrowstyle':'-|>','color':'white','lw':.5}); ax.text(.5,1.06,'High',ha='center',va='bottom',fontsize=6.5); ax.text(.5,-.06,'Low',ha='center',va='top',fontsize=6.5); ax.set_axis_off()

def render(samples, mode, interpolation, filename, add_pdf=False, contours=True):
    ev=np.concatenate([np.r_[s['oe'].ravel(),s['te'].ravel()] for s in samples]); dv=np.concatenate([np.r_[s['od'].ravel(),s['td'].ravel()] for s in samples])
    ep=np.percentile(ev,[2,98]); dp=np.percentile(dv,[2,98]);
    norm_e=PowerNorm(.65,vmin=ep[0],vmax=ep[1],clip=True) if mode=='power' else Normalize(ep[0],ep[1],clip=True)
    norm_d=PowerNorm(.65,vmin=dp[0],vmax=dp[1],clip=True) if mode=='power' else Normalize(dp[0],dp[1],clip=True)
    fig=plt.figure(figsize=(8.6,3.75),facecolor='white'); gs=GridSpec(2,5,figure=fig,wspace=.010,hspace=.012,left=.070,right=.995,top=.875,bottom=.025)
    titles=['(a) Input & GT','(b) Open Encoder','(c) Open Decoder','(d) TCC Encoder','(e) TCC Decoder']; keys=['image','oe','od','te','td']
    for i,s in enumerate(samples):
        for j,key in enumerate(keys):
            ax=fig.add_subplot(gs[i,j]); ax.set_aspect('equal'); ax.set_axis_off()
            if key=='image': ax.imshow(s[key],interpolation='nearest'); ax.text(.015,.985,'(a)' if i==0 else '(b)',transform=ax.transAxes,ha='left',va='top',fontsize=8.2,fontweight='semibold',color='white')
            else: ax.imshow(s[key],cmap='inferno',norm=norm_e if key in ('oe','te') else norm_d,interpolation=interpolation)
            if contours: contour(ax,s,.65 if key=='image' else .62)
            if i==0: ax.set_title(titles[j],fontsize=8.3,fontweight='semibold',pad=2)
    arrow(fig,float(ep[0]),float(ep[1])); base=OUT/filename; fig.savefig(str(base)+'.png',dpi=600,facecolor='white');
    if add_pdf: fig.savefig(str(base)+'.pdf',facecolor='white')
    plt.close(fig); return ep,dp

def main():
    s=[load(x) for x in IDS]; scales={}
    ep,dp=render(s,'linear','nearest','openworldsam_tccsam_dense_2x5_linear',False,True); scales['linear_nearest']={'encoder_vmin':float(ep[0]),'encoder_vmax':float(ep[1]),'decoder_vmin':float(dp[0]),'decoder_vmax':float(dp[1])}
    ep,dp=render(s,'power','nearest','openworldsam_tccsam_dense_style_2x5_compact_nearest',False,True); scales['power_nearest']={'encoder_vmin':float(ep[0]),'encoder_vmax':float(ep[1]),'decoder_vmin':float(dp[0]),'decoder_vmax':float(dp[1])}
    render(s,'power','nearest','openworldsam_tccsam_dense_2x5_power',False,True)
    render(s,'linear','nearest','openworldsam_tccsam_dense_style_2x5_compact_linear',False,True)
    render(s,'power','nearest','openworldsam_tccsam_dense_style_2x5_compact_power',False,True)
    render(s,'power','bilinear','openworldsam_tccsam_dense_style_2x5_compact_bilinear',False,True)
    render(s,'power','nearest','openworldsam_tccsam_dense_style_2x5_compact',True,True)
    render(s,'power','nearest','openworldsam_tccsam_dense_style_2x5_compact_no_contours',False,False)
    scales.update({'encoder_percentiles':[2,98],'decoder_percentiles':[2,98],'gamma':.65,'colormap':'inferno','interpolation_outputs':['nearest','bilinear'],'samples':IDS,'main_layout':'2x5','main_grid_only':True})
    (OUT/'dense_2x5_color_scale.json').write_text(json.dumps(scales,indent=2)+'\n')
    print(json.dumps(scales,indent=2))
if __name__=='__main__':main()
