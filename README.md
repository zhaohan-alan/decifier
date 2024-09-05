# deCIFer: Deep Learning for Crystal Structure Generation
deCIFer is a project aimed at developing a transformer-based deep learning model to generate crystal structures. This project builds on the CHILI-100K dataset and incorporates X-ray diffraction (XRD) data to enhance the accuracy of generated crystal structures. Future goals include making the model accessible via a user-friendly HuggingFace wrapper and potentially integrating reinforcement learning for optimization.

## Project Goals
Train a transformer model to generate plausible crystal structures without conditioning.
Develop models that condition on XRD data using prefix-based and encoded-value approaches.
Evaluate model performance against simulated XRD data.
Provide an easy-to-use API for researchers via HuggingFace.
Optionally integrate reinforcement learning for structure optimization.
Folder Structure
This project follows a structured organization to ensure ease of experimentation and reproducibility. The key directories are:

```plaintext
deCIFer/
│
├── data/
│   ├── chili100k/                    # CHILI-100K dataset files
│   │   ├── raw/                      # Raw CHILI-100K files
│   │   ├── preprocessed/             # Pickled train/val/test dataset splits for CHILI-100K
│   │   │   ├── train_dataset.pkl         
│   │   │   ├── val_dataset.pkl
│   │   │   └── test_dataset.pkl
│   │   ├── xrd_data/                 # XRD data for CHILI-100K
│   │   │   ├── train_xrd_data.pkl
│   │   │   ├── val_xrd_data.pkl
│   │   │   └── test_xrd_data.pkl
│   │   ├── embeddings/               # Embeddings of the XRD data for CHILI-100K
│   │       ├── train_embeddings.pkl
│   │       ├── val_embeddings.pkl
│   │       └── test_embeddings.pkl
│   │
│   ├── another_dataset/              # Placeholder for another dataset
│   │   ├── raw/                      # Raw files for the other dataset
│   │   ├── preprocessed/             # Preprocessed files for the other dataset
│   │   ├── xrd_data/                 # XRD data for the other dataset
│   │   └── embeddings/               # Embeddings for the other dataset
│
├── models/
│   ├── deCIFer_unconditioned.py      # Unconditioned model architecture
│   ├── deCIFer_prefix_conditioned.py # Prefix-based conditioning model
│   ├── deCIFer_encoded_conditioned.py # Encoded XRD conditioned model
│   ├── Reinforcement_Layer.py        # Reinforcement learning integration (optional)
│
├── experiments/
│   ├── unconditioned/
│   │   ├── experiment_01/            # Logs, checkpoints, and results from the first unconditioned experiment
│   └── prefix_conditioning/
│       ├── experiment_01/            # Prefix conditioning experiments
│   └── encoded_conditioning/
│       ├── experiment_01/            # Encoded conditioning experiments
│
├── scripts/
│   ├── preprocess_data.py            # Script to preprocess the dataset and split it into train/val/test sets
│   ├── train_model.py                # Script to train models
│   ├── evaluate_model.py             # Script to evaluate model performance
│   └── xrd_simulation.py             # Script to simulate XRD patterns
│
├── checkpoints/                      # Model checkpoints for different experiments
│   ├── unconditioned_model.pt        
│   ├── prefix_conditioned_model.pt   
│   ├── encoded_conditioned_model.pt  
│
└── docs/
    ├── HuggingFace_API.md            # Documentation for the HuggingFace API
    └── Experiment_Results.md         # Logs and summaries of all experiments
```

## Dataset Information
* CHILI-100K Dataset: A large dataset of crystal structures used as the primary input for model training. The dataset is loaded and split into training, validation, and test sets.
* XRD Data: X-ray diffraction data is calculated for each crystal structure in the dataset. This data is used to condition the model during training.
* Embeddings: After calculating the XRD data, embeddings are generated to represent the XRD features and stored for later model input.

## Getting Started
1. Setup the Environment
* Clone the repository:

```bash
git clone https://github.com/your_username/deCIFer.git
cd deCIFer
```

* Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Dataset Preprocessing
* Download the CHILI-100K dataset and place it in data/chili100k/.

* Preprocess the dataset and create permanent splits (train/val/test):

```bash
python scripts/preprocess_data.py
```

3. XRD Data Calculation
* Calculate XRD data for each dataset split and store it:
```bash
python scripts/xrd_simulation.py
```

## Upcoming Experiments
### Phase 1: Unconditioned Model Training
* Objective: Train a transformer model without any conditioning data.
* Files:
	- models/deCIFer_unconditioned.py
	- experiments/unconditioned/
* Script: Use train_model.py to train the unconditioned model:
```bash
python scripts/train_model.py --model unconditioned
```

### Phase 2: Prefix-Based Conditioning
* Objective: Train a model using Bragg-peak XRD data as a prefix to condition the generation process.
* Files:
	- models/deCIFer_prefix_conditioned.py
	- experiments/prefix_conditioning/

* Script: Train the prefix-conditioned model:
```bash
python scripts/train_model.py --model prefix_conditioned
```

### Phase 3: Encoded XRD Conditioning
* Objective: Encode XRD data into the model and use it as a conditioning mechanism.
* Files:
	- models/deCIFer_encoded_conditioned.py
	- experiments/encoded_conditioning/

* Script: Train the encoded-conditioned model:
```bash
python scripts/train_model.py --model encoded_conditioned
```

## Additional Features
* Reinforcement Learning (Optional): If time permits, reinforcement learning will be integrated for further optimization. The RL module is under development.

* HuggingFace API: A wrapper for deploying the trained models via HuggingFace will be developed to make the models accessible to the broader research community.

## Contributing
Contributions are welcome! Please see docs/HuggingFace_API.md for more information on how to contribute to the HuggingFace wrapper or reach out via issues.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

