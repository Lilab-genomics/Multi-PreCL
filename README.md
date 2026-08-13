# Multi-PreCL

## Description

Multi-PreCL is a novel framework designed to predict the pathogenicity of synonymous mutations by integrating multi‑scale foundation language models with supervised contrastive learning (SupCon). Unlike traditional missense‑focused approaches, Multi‑PreCL explicitly targets synonymous variants—which are often overlooked—by capturing subtle functional signals across multiple biological scales.

The framework leverages three complementary pre‑trained language models to extract hierarchical features:
- AlphaGenome – encodes local genomic and residue‑level contexts using a DCNN trained on AlphaFold‑predicted structure embeddings.

- SpliceBERT – provides splicing‑aware representations via a CNN encoder, capturing regulatory motifs that influence splicing efficiency.

- GPN‑MSA – extracts evolutionary conservation and long‑range sequence dependencies using a TextCNN trained on multi‑species whole‑genome alignments.

These three modalities are first pre‑trained as independent feature extractors. Their sequence‑level embeddings are then adaptively fused through a gated fusion module, which dynamically weights each modality’s contribution. To maximize discriminative power between pathogenic and benign synonymous variants, the model is trained with a supervised contrastive loss—pulling embeddings from the same class closer together while pushing different classes apart—optionally combined with a binary cross‑entropy loss for end‑to‑end fine‑tuning.

The entire pipeline is modular, supporting various training strategies (contrastive‑only, classification‑only, hybrid, cross‑validation, and hyper‑parameter sweeping) to facilitate robust pathogenicity assessment for clinical and research applications.

## Installation
### Create a virtual environment (recommended)

```javascript
conda create -n Multi-PreCL python=3.11
conda activate Multi-PreCL
```
### Install PyTorch

```javascript
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### Install remaining dependencies
```javascript
pip install -r requirements.txt
```
## Data Preparation
All input feature files are expected to be pre‑processed and stored as .pth (PyTorch tensor) dictionaries under the Dataset/ directory.

#### Required files for unimodal feature extractor training:

- AlphaGenome_train.pth / AlphaGenome_test.pth – shape (N, seq_len, feature_dim) with sample identifiers.
- SpliceBERT_train.pth / SpliceBERT_test.pth – same structure.
- GPN_MSA_train.pth / GPN_MSA_test.pth – same format.

Each .pth file must contain:
- X : numpy array of shape (N, L, D)
- y : numpy array of shape (N,) with binary labels (0 = benign, 1 = pathogenic)
- keys : list of unique sample IDs for tracking

If you have pre‑extracted hidden representations, you may place them in hidden_layer/ to skip the feature extraction step.
## Usage
### Pipeline Overview
The main entry point is main.py, which executes a two‑stage workflow:

1. Unimodal feature extraction – trains each foundation model encoder (if not already cached) and extracts sequence‑level embeddings for both training and test sets.

2. Multi‑scale fusion & contrastive learning – loads the extracted embeddings, constructs a multi‑modal dataset, and trains the gated fusion model using the selected supervised contrastive learning strategy.

To run the full pipeline:
```javascript
python main.py
```

## Outputs
After a successful run, the following artifacts are generated:
#### Checkpoints – saved under ./checkpoint/:
1. dcnn_best.pt – best AlphaGenome encoder
2. cnn_best.pt – best SpliceBERT encoder
3. textcnn_best.pt – best GPN‑MSA encoder
4. gatefusion_best.pt – best fusion model

#### Extracted features – saved under ./hidden_layer/:
1. *_sequence.pth – sequence‑level embeddings for each modality and data split.

#### Results – a CSV file containing detailed performance metrics

## Customisation
You can modify hyperparameters directly within the function calls in main.py (e.g., batch size, learning rate, number of epochs, patience, temperature for contrastive loss). The unimodal training functions (train_alphagenome, train_splicebert, train_gpnmsa) also expose configurable defaults.

For advanced users, the fusion model architectures (GateFusion, GateFusion_MLP_5floder, etc.) and training routines are defined in separate modules (models.py, train.py) and can be extended or replaced as needed.

## Contact
For questions, bug reports, or collaboration inquiries, please contact the repository maintainer:q24301229@stu.ahu.edu.cn



**祝您使用愉快！**