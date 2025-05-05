import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import torch
from umap import UMAP
from tqdm import tqdm
import os

from sdv.metadata import SingleTableMetadata, Metadata

from tvae.tvae_wrapper import TVAESynthesizer, LossValuesMixin
from tvae.data_loader import load_csvs

def visualize_latent_space_rhc(tvae_model, data, n_neighbors=15, min_dist=0.1, 
                          color_by=None, hover_data=None, title="Espace latent TVAE (UMAP)",
                          interactive=True, figsize=(10, 8), random_state=42):
    """
    Visualise l'espace latent d'un modèle TVAE pour les données RHC en utilisant UMAP.
    
    Args:
        tvae_model (TVAESynthesizer): 
            Un modèle TVAESynthesizer entraîné
        data (pd.DataFrame): 
            Données RHC à projeter dans l'espace latent (originales ou synthétiques)
        n_neighbors (int): 
            Paramètre de voisinage pour UMAP. Contrôle l'équilibre entre structure locale et globale.
        min_dist (float): 
            Distance minimale entre points dans l'espace de projection UMAP.
        color_by (str): 
            Nom de la colonne à utiliser pour la couleur des points
        hover_data (list): 
            Liste des colonnes à afficher lors du survol (uniquement pour le graphique interactif)
        title (str): 
            Titre du graphique
        interactive (bool): 
            Si True, utilise Plotly pour un graphique interactif, sinon Matplotlib
        figsize (tuple): 
            Taille de la figure pour Matplotlib
        random_state (int): 
            Graine aléatoire pour la reproductibilité
            
    Returns:
        figure: Objet figure Matplotlib ou Plotly selon le paramètre interactive
        latent_2d (np.ndarray): Coordonnées 2D des projections UMAP
        umap_model: Modèle UMAP entraîné pour projection future de nouvelles données
    """
    # Vérifier que le modèle est entraîné
    if not tvae_model._fitted:
        raise ValueError("Le modèle TVAE doit être entraîné avant de visualiser l'espace latent.")
    
    # Transformer les données selon le format attendu par le modèle
    processed_data = tvae_model._data_processor.transform(data)
    
    # Préparer le device et les données pour PyTorch
    device = tvae_model._model._device
    tensor_data = torch.from_numpy(processed_data.astype('float32')).to(device)
    
    # Extraire les représentations latentes en utilisant l'encodeur du modèle TVAE
    # Pour accéder à l'encodeur, nous devons créer un nouvel encodeur avec la même architecture
    # et lui transférer les poids du modèle entraîné
    print("Extraction des représentations latentes...")
    
    # Créer un nouvel encodeur avec la même architecture
    from tvae import Encoder
    data_dim = tvae_model._model.transformer.output_dimensions
    compress_dims = tvae_model.compress_dims
    embedding_dim = tvae_model.embedding_dim
    
    # Construction de l'encodeur
    encoder = Encoder(data_dim, compress_dims, embedding_dim).to(device)
    
    # Tentative de transfert des poids - attention: cela peut être limité par l'accès aux paramètres
    try:
        # Essai d'accès direct à l'encodeur du modèle
        if hasattr(tvae_model._model, 'encoder'):
            encoder.load_state_dict(tvae_model._model.encoder.state_dict())
            print("Encodeur récupéré avec succès depuis le modèle entraîné.")
        else:
            print("AVERTISSEMENT: Impossible d'accéder directement à l'encodeur entraîné.")
            print("Les représentations latentes ne correspondront pas exactement à celles du modèle.")
    except Exception as e:
        print(f"Erreur lors de la récupération de l'encodeur: {e}")
        print("Utilisation d'un encodeur non-entraîné pour la démonstration.")
    
    # Extraction des vecteurs latents
    encoder.eval()
    latent_vectors = []
    
    with torch.no_grad():
        batch_size = 256  # Utiliser la même taille de batch que dans l'entraînement
        for i in tqdm(range(0, len(tensor_data), batch_size), desc="Encodage"):
            batch = tensor_data[i:i+batch_size]
            mu, _, _ = encoder(batch)  # On récupère seulement la moyenne (mu)
            latent_vectors.append(mu.cpu().numpy())
    
    # Concaténer tous les vecteurs latents
    latent_space = np.concatenate(latent_vectors, axis=0)
    print(f"Dimensions de l'espace latent: {latent_space.shape}")
    
    # Réduction de dimensionnalité avec UMAP
    print(f"Réduction de dimensionnalité avec UMAP de {latent_space.shape[1]}D à 2D...")
    umap_reducer = UMAP(n_components=2, 
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=random_state)
    
    latent_2d = umap_reducer.fit_transform(latent_space)
    
    # Créer un DataFrame pour la visualisation
    viz_df = pd.DataFrame(latent_2d, columns=['UMAP1', 'UMAP2'])
    
    # Ajouter les données d'origine pour la coloration et les infobulles
    if color_by is not None and color_by in data.columns:
        viz_df[color_by] = data[color_by].reset_index(drop=True)
    
    if hover_data is not None:
        for col in hover_data:
            if col in data.columns:
                viz_df[col] = data[col].reset_index(drop=True)
    
    # Visualisation
    if interactive:
        # Graphique Plotly interactif
        hover_cols = hover_data if hover_data is not None else []
        if color_by is not None:
            fig = px.scatter(
                viz_df, x='UMAP1', y='UMAP2', 
                color=color_by, 
                hover_data=hover_cols,
                title=title,
                width=figsize[0]*100, height=figsize[1]*100
            )
        else:
            fig = px.scatter(
                viz_df, x='UMAP1', y='UMAP2',
                hover_data=hover_cols,
                title=title,
                width=figsize[0]*100, height=figsize[1]*100
            )
        
        # Améliorer le style du graphique
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.update_layout(
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(
                showgrid=True, gridwidth=1, gridcolor='lightgray',
                zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'
            ),
            yaxis=dict(
                showgrid=True, gridwidth=1, gridcolor='lightgray',
                zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'
            )
        )
    else:
        # Graphique Matplotlib statique
        fig, ax = plt.subplots(figsize=figsize)
        if color_by is not None and color_by in data.columns:
            scatter = ax.scatter(viz_df['UMAP1'], viz_df['UMAP2'], c=viz_df[color_by], 
                            alpha=0.7, s=30, cmap='viridis')
            if pd.api.types.is_numeric_dtype(viz_df[color_by]):
                plt.colorbar(scatter, ax=ax, label=color_by)
            else:
                # Légende pour variables catégorielles
                categories = viz_df[color_by].unique()
                handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                    markersize=8) for i, _ in enumerate(categories)]
                ax.legend(handles, categories, title=color_by, loc='best')
        else:
            ax.scatter(viz_df['UMAP1'], viz_df['UMAP2'], alpha=0.7, s=30)
        
        ax.set_title(title)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
    
    return fig, latent_2d, umap_reducer


def compare_original_synthetic_latent_rhc(tvae_model, original_data, synthetic_data, 
                                     n_neighbors=15, min_dist=0.1, color_label='Type',
                                     title="Comparaison des espaces latents", figsize=(12, 10),
                                     sample_size=None, random_state=42,
                                     target_column=None, additional_hover_columns=None):
    """
    Compare les espaces latents des données RHC originales et synthétiques.
    
    Args:
        tvae_model (TVAESynthesizer): 
            Un modèle TVAESynthesizer entraîné
        original_data (pd.DataFrame): 
            Données RHC originales
        synthetic_data (pd.DataFrame): 
            Données RHC synthétiques générées par le modèle
        n_neighbors (int): 
            Paramètre de voisinage pour UMAP
        min_dist (float): 
            Distance minimale entre points pour UMAP
        color_label (str): 
            Nom à utiliser pour la colonne de type dans la légende
        title (str): 
            Titre du graphique
        figsize (tuple): 
            Taille de la figure
        sample_size (int, optional): 
            Nombre d'échantillons à utiliser de chaque jeu de données
        random_state (int): 
            Graine aléatoire pour la reproductibilité
        target_column (str, optional):
            Colonne cible à visualiser en plus du type de données
        additional_hover_columns (list, optional):
            Liste de colonnes supplémentaires à afficher lors du survol
            
    Returns:
        figure: Objet figure Plotly
        combined_df: DataFrame combiné avec les projections UMAP
    """
    # Échantillonnage si demandé
    if sample_size is not None and sample_size < len(original_data):
        orig_sample = original_data.sample(sample_size, random_state=random_state)
    else:
        orig_sample = original_data
        
    if sample_size is not None and sample_size < len(synthetic_data):
        synth_sample = synthetic_data.sample(sample_size, random_state=random_state)
    else:
        synth_sample = synthetic_data
    
    # Préparer les hover_data
    hover_data = []
    if additional_hover_columns:
        hover_data.extend(additional_hover_columns)
    
    # Combiner les deux jeux de données
    orig_sample_copy = orig_sample.copy()
    synth_sample_copy = synth_sample.copy()
    
    orig_sample_copy[color_label] = "Original"
    synth_sample_copy[color_label] = "Synthétique"
    
    combined_data = pd.concat([orig_sample_copy, synth_sample_copy], axis=0).reset_index(drop=True)
    
    # Générer la visualisation
    fig, latent_2d, umap_model = visualize_latent_space_rhc(
        tvae_model=tvae_model,
        data=combined_data,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        color_by=color_label,
        hover_data=hover_data,
        title=title,
        interactive=True,
        figsize=figsize,
        random_state=random_state
    )
    
    # Si une colonne cible est spécifiée, créer une deuxième visualisation 
    # avec la même projection UMAP mais colorée par la colonne cible
    if target_column and target_column in combined_data.columns:
        # Création d'un DataFrame avec les coordonnées UMAP et la colonne cible
        target_viz_df = pd.DataFrame(latent_2d, columns=['UMAP1', 'UMAP2'])
        target_viz_df[target_column] = combined_data[target_column].reset_index(drop=True)
        target_viz_df[color_label] = combined_data[color_label].reset_index(drop=True)
        
        # Création du graphique Plotly
        target_fig = px.scatter(
            target_viz_df, x='UMAP1', y='UMAP2',
            color=target_column,
            symbol=color_label,  # Utiliser symboles différents pour Original/Synthétique
            hover_data=[target_column, color_label] + (hover_data if hover_data else []),
            title=f"{title} - Coloré par {target_column}",
            width=figsize[0]*100, height=figsize[1]*100
        )
        
        # Améliorer le style du graphique
        target_fig.update_traces(marker=dict(size=6, opacity=0.7))
        target_fig.update_layout(
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(
                showgrid=True, gridwidth=1, gridcolor='lightgray',
                zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'
            ),
            yaxis=dict(
                showgrid=True, gridwidth=1, gridcolor='lightgray',
                zeroline=True, zerolinewidth=1, zerolinecolor='lightgray'
            )
        )
        
        return fig, target_fig, latent_2d, umap_model
    
    return fig, latent_2d, umap_model


def save_latent_space(latent_2d, data, filename, include_columns=None):
    """
    Sauvegarde les coordonnées de l'espace latent avec les données d'origine.
    
    Args:
        latent_2d (np.ndarray): Coordonnées 2D des projections UMAP
        data (pd.DataFrame): Données d'origine
        filename (str): Nom du fichier de sortie
        include_columns (list, optional): Colonnes à inclure dans le fichier de sortie
    """
    # Créer un DataFrame avec les coordonnées UMAP
    latent_df = pd.DataFrame(latent_2d, columns=['UMAP1', 'UMAP2'])
    
    # Si des colonnes sont spécifiées, les ajouter
    if include_columns:
        for col in include_columns:
            if col in data.columns:
                latent_df[col] = data[col].reset_index(drop=True)
    
    # Sauvegarder le DataFrame
    latent_df.to_csv(filename, index=False)
    print(f"Espace latent sauvegardé dans {filename}")
    
    return latent_df