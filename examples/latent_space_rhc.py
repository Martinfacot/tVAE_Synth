"""Example script for visualizing TVAE latent space with UMAP."""

import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sdv.metadata import SingleTableMetadata
from sdv.metadata import Metadata

# Add the project root directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tvae.tvae_wrapper import TVAESynthesizer
from tvae.visualization import (
    visualize_latent_space,
    compare_original_synthetic_latent,
    calculate_latent_statistics,
    plot_latent_dimensions
)


def load_model(model_path):
    """Load a trained TVAE model from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def main():
    # Path to the data and model
    data_path = os.path.join('data', 'rhc.csv')
    metadata_path = os.path.join('data', 'metadata.json')
    
    # Directory for saving visualizations
    output_dir = os.path.join('Images', 'latent_space')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print("Loading data...")
    data = pd.read_csv(data_path)
    
    # Use existing trained model or create a new one
    model_path = os.path.join('examples', 'test_model_tvae_ep1000_compress32.pkl')
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        tvae = load_model(model_path)
    else:
        print("Training a new TVAE model...")
        # Load metadata
        try:
            metadata = Metadata.load(metadata_path)
            # Extract single table metadata for 'rhc' table
            metadata = metadata.tables['rhc']
        except:
            # If loading fails, create new metadata
            print("Creating new metadata...")
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data)
        
        # Initialize and train TVAE
        tvae = TVAESynthesizer(
            metadata=metadata,
            epochs=50,  # Using fewer epochs for demonstration
            embedding_dim=32,
            compress_dims=(64, 32),
            decompress_dims=(32, 64),
            verbose=True
        )
        tvae.fit(data)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(tvae, f)
    
    # Load synthetic data if it exists or generate new data
    synthetic_path = os.path.join('examples', 'synthetic_data_tvae_ep1000_compress32.csv')
    
    if os.path.exists(synthetic_path):
        print(f"Loading existing synthetic data from {synthetic_path}")
        synthetic_data = pd.read_csv(synthetic_path)
    else:
        print("Generating synthetic data...")
        synthetic_data = tvae.sample(len(data))
        synthetic_data.to_csv(synthetic_path, index=False)
    
    # Visualize latent space colored by various attributes
    print("Visualizing latent space...")
    
    # Try to color by different interesting columns
    for color_column in ['death', 'sex', 'age']:
        if color_column in data.columns:
            save_path = os.path.join(output_dir, f'latent_space_{color_column}.png')
            fig, latent_emb, umap_emb = visualize_latent_space(
                tvae_synthesizer=tvae,
                data=data,
                color_by=color_column,
                save_path=save_path
            )
            plt.close(fig)
            print(f"Visualization saved to {save_path}")
    
    # Compare original and synthetic data in latent space
    print("Comparing original and synthetic data in latent space...")
    compare_path = os.path.join(output_dir, 'compare_original_synthetic.png')
    compare_fig, compare_latent, compare_umap = compare_original_synthetic_latent(
        tvae_synthesizer=tvae,
        original_data=data,
        synthetic_data=synthetic_data,
        save_path=compare_path
    )
    plt.close(compare_fig)
    print(f"Comparison visualization saved to {compare_path}")
    
    # Calculate and display statistics of the latent dimensions
    print("Calculating latent space statistics...")
    latent_stats = calculate_latent_statistics(latent_emb)
    print(latent_stats)
    
    # Plot distributions of latent dimensions
    print("Plotting latent dimension distributions...")
    dims_path = os.path.join(output_dir, 'latent_dimensions.png')
    dims_fig = plot_latent_dimensions(
        latent_embeddings=latent_emb,
        n_dims=min(20, tvae.embedding_dim),  # Plot at most 20 dimensions
        save_path=dims_path
    )
    plt.close(dims_fig)
    print(f"Dimension distributions saved to {dims_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()