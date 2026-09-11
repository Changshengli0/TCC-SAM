from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg'); mpl.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'Arial'})
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from render_dense_style_compact_4x3 import load, IDS, OUT

def main():
    samples=[load(s) for s in IDS]
    ev=np.concatenate([np.r_[s['oe'].ravel(),s['te'].ravel()] for s in samples]); dv=np.concatenate([np.r_[s['od'].ravel(),s['td'].ravel()] for s in samples])
    ep=np.percentile(ev,[2,98]); dp=np.percentile(dv,[2,98]); ne=Normalize(ep[0],ep[1],clip=True); nd=Normalize(dp[0],dp[1],clip=True)
    fig=plt.figure(figsize=(9,3.35),facecolor='white'); gs=GridSpec(2,5,figure=fig,wspace=.008,hspace=.012,left=.015,right=.945,top=.875,bottom=.025)
    titles=['(a) Input & GT','(b) Open Encoder','(c) Open Decoder','(d) TCC Encoder','(e) TCC Decoder']; keys=['image','oe','od','te','td']
    for i,s in enumerate(samples):
        for j,k in enumerate(keys):
            ax=fig.add_subplot(gs[i,j]); ax.set_aspect('equal'); ax.set_axis_off()
            if k=='image': ax.imshow(s[k],interpolation='bilinear')
            else: ax.imshow(s[k],cmap='inferno',norm=ne if k in ('oe','te') else nd,interpolation='nearest')
            if i==0: ax.set_title(titles[j],fontsize=8.0,fontweight='semibold',pad=2)
    aa=fig.add_axes([.958,.105,.012,.755]); aa.imshow(np.linspace(0,1,256).reshape(-1,1),cmap='inferno',origin='lower',aspect='auto'); aa.annotate('',xy=(.5,1.015),xytext=(.5,-.015),arrowprops={'arrowstyle':'-|>','color':'white','lw':.5}); aa.text(.5,1.055,'High',ha='center',va='bottom',fontsize=7.5); aa.text(.5,-.055,'Low',ha='center',va='top',fontsize=7.5); aa.set_axis_off()
    base=OUT/'openworldsam_tccsam_dense_style_2x5_compact_linear_corrected'; fig.savefig(str(base)+'.png',dpi=600,facecolor='white'); fig.savefig(str(base)+'.pdf',facecolor='white'); plt.close(fig)
    print({'encoder_range':[float(ep[0]),float(ep[1])],'decoder_range':[float(dp[0]),float(dp[1])],'contours':False,'padding':False,'interpolation':'nearest'})
if __name__=='__main__': main()
