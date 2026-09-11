from pathlib import Path
import csv
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial'})
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

ROOT = Path('/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main')
OUT = ROOT / 'outputs/paper_figs/openworldsam_tccsam_full_model_response'
IDS = [
    'refcocog_val_umd_69231_p001_07a73e714a',
    'refcocog_val_umd_325545_p001_4e3dda118f',
]

def screening_row(sid):
    p = ROOT / 'outputs/paper_figs/openworldsam_tccsam_contrastive_feature/candidate_prediction_screening.csv'
    return next(r for r in csv.DictReader(p.open()) if r['sample_id'] == sid)

def load_sample(sid):
    p = OUT / f'{sid}_full_model_responses.npz'
    with np.load(p, allow_pickle=False) as z:
        return {
            'sid': sid,
            'image': np.asarray(Image.open(screening_row(sid)['raw_path']).convert('RGB')),
            'target': z['target_mask_64'].astype(bool),
            'distractor': z['distractor_mask_64'].astype(bool),
            'open_encoder': z['open_encoder_response'],
            'open_decoder': z['open_decoder_response'],
            'tcc_encoder': z['tcc_encoder_response'],
            'tcc_decoder': z['tcc_decoder_response'],
        }

def add_contours(ax, sample):
    ax.contour(sample['target'].astype(float), [0.5], colors='white', linewidths=0.40)
    ax.contour(sample['distractor'].astype(float), [0.5], colors='#d0d0d0', linewidths=0.35, linestyles='--')

def render(with_contours=True):
    samples = [load_sample(s) for s in IDS]
    enc_values = np.concatenate([np.r_[s['open_encoder'].ravel(), s['tcc_encoder'].ravel()] for s in samples])
    dec_values = np.concatenate([np.r_[s['open_decoder'].ravel(), s['tcc_decoder'].ravel()] for s in samples])
    enc_vmin, enc_vmax = np.percentile(enc_values, [1, 99])
    dec_vmin, dec_vmax = np.percentile(dec_values, [1, 99])

    fig = plt.figure(figsize=(9.4, 3.95), facecolor='white')
    gs = GridSpec(2, 6, figure=fig, width_ratios=[0.055, 1, 1, 1, 1, 1],
                  wspace=0.035, hspace=0.055, left=0.035, right=0.995,
                  top=0.875, bottom=0.055)
    arrow_ax = fig.add_subplot(gs[:, 0])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    arrow_ax.imshow(grad, cmap='magma', aspect='auto', origin='lower', extent=(0, 1, 0, 1))
    arrow_ax.annotate('', xy=(0.5, 0.98), xytext=(0.5, 0.02),
                      arrowprops={'arrowstyle': '-|>', 'color': 'white', 'lw': 0.75})
    arrow_ax.text(0.5, 1.015, 'High\nresponse', ha='center', va='bottom', fontsize=6.5)
    arrow_ax.text(0.5, -0.015, 'Low\nresponse', ha='center', va='top', fontsize=6.5)
    arrow_ax.set_axis_off()

    titles = [('Input & GT',), ('OpenWorldSAM', 'Encoder'), ('OpenWorldSAM', 'Decoder'),
              ('TCC-SAM', 'Encoder'), ('TCC-SAM', 'Decoder')]
    keys = [('image', None), ('open_encoder', (enc_vmin, enc_vmax)),
            ('open_decoder', (dec_vmin, dec_vmax)), ('tcc_encoder', (enc_vmin, enc_vmax)),
            ('tcc_decoder', (dec_vmin, dec_vmax))]
    for i, sample in enumerate(samples):
        for j, (key, lim) in enumerate(keys, start=1):
            ax = fig.add_subplot(gs[i, j])
            if key == 'image':
                ax.imshow(sample[key], interpolation='bicubic')
                if with_contours:
                    add_contours(ax, sample)
            else:
                ax.imshow(sample[key], cmap='magma', vmin=lim[0], vmax=lim[1], interpolation='bicubic')
            if i == 0:
                ax.set_title('\n'.join(titles[j - 1]), fontsize=9, fontweight='semibold', pad=5)
            ax.set_axis_off()
    suffix = '' if with_contours else '_no_contours'
    base = OUT / f'openworldsam_tccsam_dense_style_2x5_preview{suffix}'
    fig.savefig(str(base) + '.png', dpi=600, facecolor='white')
    if with_contours:
        fig.savefig(str(base) + '.pdf', facecolor='white')
    plt.close(fig)
    return {'encoder_display_range': [float(enc_vmin), float(enc_vmax)],
            'decoder_display_range': [float(dec_vmin), float(dec_vmax)],
            'layout': '2 rows x 5 main panels', 'samples': IDS,
            'feature_keys': [k for k, _ in keys], 'cmap': 'magma',
            'interpolation': 'bicubic', 'contours': with_contours}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {'with_contours': render(True), 'without_contours': render(False),
             'notes': ['neck/fusion, full-model, difference, prediction, statistics omitted',
                       'TCC columns are labeled TCC-SAM; ECC has no independent encoder/decoder']}
    (OUT / 'openworldsam_tccsam_dense_style_2x5_preview_audit.json').write_text(__import__('json').dumps(audit, indent=2) + '\n')
    print(__import__('json').dumps(audit, indent=2))

if __name__ == '__main__':
    main()
