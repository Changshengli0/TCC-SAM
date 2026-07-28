TCC-SAM

Official PyTorch implementation of TCC-SAM for Referring Image Segmentation (RIS).

TCC-SAM introduces two main components:

Text-Conditioned Visual Modulator (TVM): injects expression-aware information into high-level visual features while preserving the general segmentation capability of SAM.

Evidence-Guided Candidate Calibrator (ECC): evaluates and recalibrates native SAM mask candidates using visual, linguistic, and candidate-level evidence.

Repository Status

The source code is currently being organized for public release.

The following materials will be added or refined progressively:

training and evaluation scripts;

dataset preparation instructions;

pretrained checkpoints;

inference examples;

configuration files;

detailed reproduction instructions.

Repository Structure

TCC-SAM/
├── demo/                  # Inference and visualization examples
├── evaluation/            # Evaluation code
├── model/                 # TCC-SAM model implementation
├── scripts/               # Training and evaluation scripts
├── third_party_baselines/ # Required third-party components
├── tools/                 # Auxiliary tools
├── utils/                 # Utility functions
├── DATASETS.md            # Dataset preparation
├── LICENSE.txt            # License information
├── requirements.txt       # Python dependencies
└── train_net.py           # Main training and evaluation entry

Installation
Clone this repository and install the required dependencies:

git clone https://github.com/Changshengli0/TCC-SAM.git
cd TCC-SAM
pip install -r requirements.txt

Please refer to DATASETS.md for dataset preparation.

Training and Evaluation

Detailed commands for training, evaluation, and inference will be provided after the public code is fully organized.

Checkpoints

Pretrained TCC-SAM checkpoints will be released separately. Large model weights are not stored directly in this repository.

Citation

The paper citation will be added after the manuscript information becomes publicly available.

@article{tccsam,
  title   = {TCC-SAM},
  author  = {To be updated},
  journal = {To be updated},
  year    = {To be updated}
}
Acknowledgements

This project is developed based on several excellent open-source projects, including SAM, BEiT-3, Detectron2, and related referring image segmentation frameworks. We sincerely thank their authors for making their code and models publicly available.

License

Please refer to LICENSE.txt for license information.
