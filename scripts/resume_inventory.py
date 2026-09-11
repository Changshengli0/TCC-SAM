from pathlib import Path
import csv

OUT = Path('outputs/paper_figs/openworldsam_tccsam_full_model_response')
ids = [
    'refcocog_val_umd_562474_p000_aa13191e8e',
    'refcocog_val_umd_325545_p001_4e3dda118f',
    'refcocog_val_umd_581042_p000_ba67ba21f6',
]
fields = ['sample_id', 'openworldsam_encoder_exists', 'openworldsam_neck_exists',
          'openworldsam_decoder_exists', 'tccsam_encoder_exists', 'tccsam_neck_exists',
          'tccsam_decoder_exists', 'fusion_npz_exists', 'preview_exists']
rows = []
for sid in ids:
    raw = OUT / 'raw_features' / sid
    def has(method, pattern):
        return any((raw / method).glob(pattern))
    rows.append({
        'sample_id': sid,
        'openworldsam_encoder_exists': has('openworldsam', 'encoder_stage*.npy'),
        'openworldsam_neck_exists': has('openworldsam', 'neck_scale*.npy'),
        'openworldsam_decoder_exists': (raw / 'openworldsam' / 'decoder_spatial.npy').is_file(),
        'tccsam_encoder_exists': has('tcc_sam', 'encoder_stage*.npy'),
        'tccsam_neck_exists': has('tcc_sam', 'neck_scale*.npy'),
        'tccsam_decoder_exists': (raw / 'tcc_sam' / 'decoder_spatial.npy').is_file(),
        'fusion_npz_exists': (OUT / f'{sid}_full_model_responses.npz').is_file(),
        'preview_exists': (OUT / f'{sid}_full_model_preview.png').is_file(),
    })
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'resume_inventory_after_linux_recovery.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print('\n'.join(','.join(str(r[k]) for k in fields) for r in rows))
