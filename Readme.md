# Genomic data management and processing

## UPDATE
2025.03.03 - Due to university policy, the data required for recreation has been removed from the repository

## Overview
This repository contains a Proof of Concept (POC) for [briefly describe your project, e.g., a data processing pipeline, a web application, etc.]. The primary goal of this POC is to explore the feasibility and demonstrate core functionalities that will be part of the finalized version of the project.

## Important Notice
Please note that this is a preliminary version of the project. The finalized version will differ significantly in terms of code structure, optimizations, and additional features.

## Features
### Data Handling and Processing
- Developed a comprehensive data processing system to handle large amounts of genomic data.
- Utilized PostgreSQL for optimized data storage and retrieval.

### Docker Containerization
- Containerized architecture using Docker to ensure consistent and reproducible environments.
- Enabled data upload and database management in isolated containers for better stability.

### Scalable Infrastructure
- The Docker setup is designed to be scalable, with Kubernetes or Docker Swarm proposed for future improvements.
- Potential for load balancing and scaling using Kubernetes.

### Machine Learning Integration
- Applied various machine learning models on genomic data (VCF files) to determine their effectiveness.
- Implemented hyperparameter optimization techniques like grid search and Bayesian optimization for model tuning.
- Evaluated performance on genomic datasets and clinical datasets for model comparison.

### Optimization Techniques
- Improved database query performance using indexing, optimized joins, and block-based techniques.
- Compared unoptimized and optimized query execution times.

### Testing Environment
- Employed Python scripts for data cleaning and uploading.
- Divided datasets into training and testing subsets for accurate performance evaluation.

## Future Work
### Further Improvements
- Recommendations include advanced imputation methods for missing data, enhancing system scalability, and creating a user-friendly web interface for the model.
- Future development aims for better container orchestration using Kubernetes and potential web-based interfaces for ease of use.

## Disclaimer
This POC is intended for testing and demonstration purposes only.

---
For questions or suggestions, please contact marcell.lenkei@gmail.com.

