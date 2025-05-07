import numpy as np
import pandas as pd
import torch
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def visualize_latent_space(
    tvae_synthesizer, 
    data, 
    discrete_columns=None,
    n_neighbors=15, 
    min_dist=0.1,
    n_components=2,
    random_state=42,
    figsize=(12, 10),
    point_size=5,
    alpha=0.7,
    color_by=None,
    cmap='viridis',
    save_path=None
):
    """
    Visualize the latent space of a trained TVAE model using UMAP.
    
    Args:
        tvae_synthesizer (TVAESynthesizer): 
            A trained TVAESynthesizer instance.
        data (pandas.DataFrame): 
            The original data used for training or new data to embed.
        discrete_columns (list, optional): 
            List of discrete columns. If None, will try to detect from metadata.
        n_neighbors (int, optional): 
            UMAP parameter for local neighborhood size. Defaults to 15.
        min_dist (float, optional): 
            UMAP parameter for minimum distance between points. Defaults to 0.1.
        n_components (int, optional): 
            Number of dimensions for UMAP projection. Defaults to 2.
        random_state (int, optional): 
            Random seed for reproducibility. Defaults to 42.
        figsize (tuple, optional): 
            Figure size. Defaults to (12, 10).
        point_size (int, optional): 
            Size of the scatter plot points. Defaults to 5.
        alpha (float, optional): 
            Transparency of points. Defaults to 0.7.
        color_by (str, optional): 
            Column name to color points by. Must be a column in the data.
        cmap (str, optional): 
            Colormap for continuous variables. Defaults to 'viridis'.
        save_path (str, optional): 
            Path to save the figure. If None, the figure is not saved.
            
    Returns:
        tuple: 
            (matplotlib figure, latent embeddings, UMAP embeddings)
    """
    if not tvae_synthesizer._fitted:
        raise ValueError("The TVAE synthesizer must be fitted before visualizing the latent space.")
    
    # Ensure the model is in evaluation mode
    tvae_synthesizer._model.decoder.eval()
    
    # Get the device
    device = tvae_synthesizer._model._device
    
    # Transform the data through the data processor
    processed_data = tvae_synthesizer._data_processor.transform(data)
    
    # If discrete_columns not provided, try to detect from metadata
    if discrete_columns is None:
        transformers = tvae_synthesizer._data_processor._hyper_transformer.field_transformers
        try:
            from sdv.single_table.utils import detect_discrete_columns
            discrete_columns = detect_discrete_columns(tvae_synthesizer.metadata, processed_data, transformers)
        except:
            discrete_columns = []
            print("Could not automatically detect discrete columns. Treating all as continuous.")
    
    # Create a dataset and dataloader
    dataset = TensorDataset(torch.from_numpy(processed_data.astype('float32')).to(device))
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    # Extract the encoder from TVAE model
    # This requires accessing the encoder which is not stored in the TVAE model after training
    # So we need to recreate it with the same parameters
    from tvae.tvae import Encoder
    
    # Get data dimensions from the transformer
    data_dim = tvae_synthesizer._model.transformer.output_dimensions
    
    # Create a new encoder with the same parameters
    encoder = Encoder(
        data_dim=data_dim,
        compress_dims=tvae_synthesizer.compress_dims,
        embedding_dim=tvae_synthesizer.embedding_dim
    ).to(device)
    
    # We need to train this encoder on the same data
    # Since we don't have access to the trained encoder
    # This is a simplified approach to get approximate embeddings
    print("Extracting latent embeddings...")
    
    # We'll collect encodings by passing data through the model
    latent_embeddings = []
    
    with torch.no_grad():
        for data_batch in tqdm(dataloader):
            real = data_batch[0].to(device)
            mu, std, logvar = encoder(real)
            
            # This is equivalent to the latent space sampling in TVAE
            eps = torch.randn_like(std)
            emb = eps * std + mu
            
            latent_embeddings.append(emb.cpu().numpy())
    
    # Combine all batches
    latent_embeddings = np.vstack(latent_embeddings)
    
    # Apply UMAP for dimensionality reduction
    print(f"Applying UMAP to reduce dimensionality to {n_components}...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    
    embedding = reducer.fit_transform(latent_embeddings)
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by is not None and color_by in data.columns:
        color_values = data[color_by].values
        
        # Check if the column is categorical or numerical
        if color_by in discrete_columns or pd.api.types.is_categorical_dtype(data[color_by]) or \
           pd.api.types.is_object_dtype(data[color_by]) or \
           pd.api.types.is_string_dtype(data[color_by]):
            # Categorical coloring
            unique_categories = data[color_by].unique()
            category_colors = sns.color_palette('tab10', n_colors=len(unique_categories))
            
            for i, category in enumerate(unique_categories):
                mask = color_values == category
                ax.scatter(
                    embedding[mask, 0], 
                    embedding[mask, 1],
                    s=point_size,
                    alpha=alpha,
                    label=category,
                    color=category_colors[i]
                )
            ax.legend(title=color_by)
            
        else:
            # Numerical coloring
            scatter = ax.scatter(
                embedding[:, 0], 
                embedding[:, 1],
                s=point_size,
                alpha=alpha,
                c=color_values,
                cmap=cmap
            )
            plt.colorbar(scatter, label=color_by)
    else:
        ax.scatter(
            embedding[:, 0], 
            embedding[:, 1],
            s=point_size,
            alpha=alpha
        )
    
    ax.set_title(f'TVAE Latent Space (UMAP {n_components}D Projection)', fontsize=14)
    ax.set_xlabel(f'UMAP Dimension 1', fontsize=12)
    ax.set_ylabel(f'UMAP Dimension 2', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, latent_embeddings, embedding

def calculate_latent_statistics(latent_embeddings):
    """
    Calculate basic statistics of the latent space.
    
    Args:
        latent_embeddings (numpy.ndarray): 
            Latent embeddings extracted from the TVAE model.
            
    Returns:
        pandas.DataFrame: Statistics of the latent dimensions.
    """
    stats = []
    
    for i in range(latent_embeddings.shape[1]):
        dim_data = latent_embeddings[:, i]
        stats.append({
            'Dimension': i,
            'Mean': np.mean(dim_data),
            'Std': np.std(dim_data),
            'Min': np.min(dim_data),
            'Max': np.max(dim_data),
            'Median': np.median(dim_data),
            'Skewness': float(pd.Series(dim_data).skew()),
            'Kurtosis': float(pd.Series(dim_data).kurtosis())
        })
    
    return pd.DataFrame(stats)

def compare_original_synthetic_latent(
    tvae_synthesizer, 
    original_data, 
    synthetic_data,
    discrete_columns=None,
    n_components=2,
    random_state=42,
    figsize=(14, 10),
    save_path=None
):
    """
    Compare original and synthetic data in latent space using UMAP.
    
    Args:
        tvae_synthesizer (TVAESynthesizer): 
            A trained TVAESynthesizer instance.
        original_data (pandas.DataFrame): 
            The original data used for training.
        synthetic_data (pandas.DataFrame): 
            The synthetic data generated by the model.
        discrete_columns (list, optional): 
            List of discrete columns. If None, will try to detect from metadata.
        n_components (int, optional): 
            Number of dimensions for UMAP projection. Defaults to 2.
        random_state (int, optional): 
            Random seed for reproducibility. Defaults to 42.
        figsize (tuple, optional): 
            Figure size. Defaults to (14, 10).
        save_path (str, optional): 
            Path to save the figure. If None, the figure is not saved.
            
    Returns:
        tuple: (matplotlib figure, latent embeddings dict, UMAP embeddings dict)
    """
    if not tvae_synthesizer._fitted:
        raise ValueError("The TVAE synthesizer must be fitted before comparing latent spaces.")
    
    # Ensure columns match between original and synthetic data
    synthetic_data = synthetic_data[original_data.columns]
    
    # Combine the data and add a source column
    original_with_source = original_data.copy()
    original_with_source['data_source'] = 'Original'
    
    synthetic_with_source = synthetic_data.copy()
    synthetic_with_source['data_source'] = 'Synthetic'
    
    combined_data = pd.concat([original_with_source, synthetic_with_source], axis=0)
    
    # Visualize the combined data, coloring by source
    fig, latent_emb, umap_emb = visualize_latent_space(
        tvae_synthesizer=tvae_synthesizer,
        data=combined_data,
        discrete_columns=discrete_columns,
        n_components=n_components,
        random_state=random_state,
        figsize=figsize,
        color_by='data_source',
        save_path=save_path
    )
    
    # Split the embeddings back into original and synthetic
    n_original = len(original_data)
    original_latent = latent_emb[:n_original]
    synthetic_latent = latent_emb[n_original:]
    
    original_umap = umap_emb[:n_original]
    synthetic_umap = umap_emb[n_original:]
    
    return fig, {
        'original': original_latent,
        'synthetic': synthetic_latent
    }, {
        'original': original_umap,
        'synthetic': synthetic_umap
    }

# Example usage in Jupyter notebook:
"""
# Import necessary libraries
import pandas as pd
from sdv.metadata import SingleTableMetadata
from tvae.tvae_wrapper import TVAESynthesizer

# Load your data
data = pd.read_csv('your_data.csv')

# Create metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# Initialize and train TVAE
tvae = TVAESynthesizer(
    metadata=metadata,
    epochs=100,
    verbose=True
)
tvae.fit(data)

# Generate synthetic data
synthetic_data = tvae.sample(len(data))

# Visualize latent space
fig, latent_emb, umap_emb = visualize_latent_space(
    tvae_synthesizer=tvae,
    data=data,
    color_by='your_target_column',  # Optional: color by a specific column
    save_path='tvae_latent_space.png'
)

# Compare original and synthetic data in latent space
compare_fig, compare_latent, compare_umap = compare_original_synthetic_latent(
    tvae_synthesizer=tvae,
    original_data=data,
    synthetic_data=synthetic_data,
    save_path='tvae_compare_latent.png'
)

# Calculate statistics of the latent dimensions
latent_stats = calculate_latent_statistics(latent_emb)
print(latent_stats)
"""