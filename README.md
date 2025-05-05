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

*This project is part of an External Pharmacy Internship at CHU Strasbourg.*