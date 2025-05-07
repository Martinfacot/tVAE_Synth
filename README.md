# Synthetic Healthcare Data Generation with tVAE
![Reparameterized Variational Autoencoder](https://github.com/Martinfacot/tVAE_project/raw/dev/Images/Reparameterized_Variational_Autoencoder.png)

## Project Overview

This project implements a Tabular Variational Autoencoder (tVAE) for generating synthetic healthcare data that preserves the statistical properties and relationships of the original dataset while ensuring privacy based on Synthetic Data Vault (SDV) framework. It was developed as part of my External Pharmacy Internship at the CHU (University Hospital) of Strasbourg.

The implementation focuses on the Right Heart Catheterization (RHC) dataset with an emphasis on data quality, proper handling of missing values, and utility preservation.

## Key Objectives
- **Miniaturize SDV Code**: Extract and optimize the core tVAE functionality from the SDV framework for specific healthcare applications
- **Generate High-Quality Synthetic Data**: Produce synthetic healthcare datasets that maintain statistical properties of the original data
- **Implement Evaluation Methods**: Develop and apply metrics to assess the quality and utility of the generated synthetic data
- **Privacy Preservation Analysis**: Evaluate the risk of patient re-identification or information leakage
- **Anonymization Research**: Study techniques to ensure that synthetic data maintains privacy guarantees

## About the Project

This project represents an exploration into privacy-preserving synthetic data generation methods for healthcare applications at CHU Strasbourg. The goal is to create high-quality synthetic data that can be safely shared and used for research purposes without compromising patient privacy.

By using a Variational Autoencoder approach, we can generate new synthetic patient records that maintain the complex relationships between medical variables while ensuring no actual patient data is revealed.

## Next Steps

Further development will focus on expanding the evaluation metrics, implement privacy preservation & anonymization research and testing with additional healthcare datasets.

---
**This project is part of an External Pharmacy Internship at CHU Strasbourg.**
---

## How to Use

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/tVAE_Synth.git
   cd tVAE_Synth
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. Place your CSV dataset(s) in the `data/` directory
2. Create a metadata.json file describing your data structure (see example in data/metadata.json)

### Training a Model

```python
# Import required modules
from tvae.tvae_wrapper import TVAESynthesizer
from tvae.data_loader import load_csvs
from sdv.metadata import Metadata

# Load your data
datasets = load_csvs(folder_name='data/')
data = datasets['your_dataset_name']

# Load metadata
metadata = Metadata.load_from_json('data/metadata.json')
metadata.validate()

# Initialize and train the model
model = TVAESynthesizer(
    metadata,
    embedding_dim=32,          # latent space dimensions
    compress_dims=(128, 32),   # encoder dimensions
    decompress_dims=(32, 128), # decoder dimensions
    l2scale=1e-5,
    batch_size=256,
    verbose=True,
    epochs=500
)

model.fit(data)
```

### Generating Synthetic Data

```python
# Generate synthetic samples
synthetic_data = model.sample(num_samples=len(data))

# Save the synthetic data to CSV
synthetic_data.to_csv('synthetic_data.csv', index=False)
```

### Evaluating the Model

```python
# Visualize training loss
model.plot_loss(show_batch_loss=True)

# Save your trained model
model.save(filepath='trained_model.pkl')

# Evaluate synthetic data quality
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality

# Run diagnostic tests
diagnostic = run_diagnostic(
    real_data=data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

# Get quality metrics
quality_report = evaluate_quality(
    data,
    synthetic_data,
    metadata
)
```

### Visualizing Latent Space

```python
from examples.latent_space_rhc import visualize_latent_space_rhc

# Visualize the latent space 
fig, latent_coords, umap_model = visualize_latent_space_rhc(
    tvae_model=model,
    data=data,
    color_by='target_variable',
    hover_data=['feature1', 'feature2', 'feature3'],
    interactive=True
)
fig.show()
```

Check the example notebooks in the `examples/` directory for more detailed demonstrations.
