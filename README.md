# deCIFer: Crystal Structure Prediction from Powder Diffraction Data

**Shortened Abstract**  
deCIFer is an autoregressive language model for crystal structure prediction (CSP) from powder X-ray diffraction (PXRD) data, generating Crystallographic Information Files (CIFs) directly from diffraction patterns. Trained on nearly 2.3 million crystal structures, it is validated on diverse PXRD datasets for inorganic systems. Evaluations using residual weighted profile (Rwp) and Wasserstein distance show improved predictions when conditioned on diffraction data. deCIFer achieves a 94% match rate on unseen data, bridging experimental diffraction with computational CSP for materials discovery.

## Table of Contents
1. [Setup](#setup)
2. [Data Preparation](#data-preparation)
3. [Training from Scratch](#training-from-scratch)
4. [Resuming Training](#resuming-training)
5. [Evaluation Pipeline](#evaluation)
6. [Additional Functionalities](#additional-functionalities)
   - [Unconditioned Structure Generation](#unconditioned-structure-generation)
   - [Conditioned Structure Generation](#conditioned-structure-generation)
   - [Consistency Checks with `generation_consistency.py`](#consistency-checks-with-generation_consistencypy)
7. [License](#license)

# Setup
We recommend using **Python 3.9**, as deCIFer was developed and tested with this version. Other versions may work but have not been verified.

1. **Clone the repository**:
```bash
git clone https://github.com/XXXX/deCIFer.git
cd deCIFer
```

2. **Install deCIFer using setup.py.** We recommend an isolated environment (e.g., with Conda or venv):
```bash
pip install -e .
```

3. **Ensure that you have PyTorch installed:**
Follow the instructions on the official PyTorch website to install the appropriate version for your system: PyTorch Installation Guide. (https://pytorch.org/get-started/locally/)

4. **Install other dependencies**:
```bash
pip install numpy pandas matplotlib seaborn pyYAML tqdm omegaconf h5py pymatgen periodictable scikit-learn
```

# Data Preparation
```python
python bin/prepare_dataset.py --data-dir data/noma/ --name noma-1k --debug-max 1000 --all --raw-from-gzip
```
