Project Overview

This repository contains a machine learning workflow developed to analyze student depression data and build predictive models using a structured, modular pipeline. The goal of the project is to process raw survey data, explore underlying patterns, engineer meaningful features, and evaluate multiple supervised learning models to understand the factors contributing to depressive tendencies.

The analysis is carried out in a single central notebook (ML_notebook.ipynb), while all reusable logic is separated into standalone Python modules inside the funcs/ directory. This ensures cleaner experimentation, easier updates, and reproducibility.

The project follows a complete end-to-end ML lifecycle, beginning with data loading and preparation, followed by exploratory analysis, feature transformation, model training, hyperparameter optimization, and performance evaluation. Each stage is modularized so individual components can be reused, replaced, or extended without modifying the notebook.

The repository is organized to allow a new user to quickly understand where data is stored, how processing functions are structured, and how the notebook links everything together to produce results.

Repository Structure

1. data/

Contains the dataset used in the study:

student_depression_dataset.csv

Raw input file including demographic, behavioral, academic, and mental-health-related fields.

2. funcs/

Holds all modular processing scripts. Each file corresponds to a specific stage in the ML pipeline:

data_preprocessing.py

Functions for cleaning, handling missing values, encoding, and scaling.

eda_plotting.py

Utilities for visualizing distributions, correlations, and variable relationships.

feature_selection.py

Techniques such as variance filtering, correlation checks, and model-based ranking.

classification.py and regression.py

Model training utilities for supervised tasks.

clustering.py

Unsupervised grouping to explore hidden sub-patterns in student responses.

model_evaluation.py

Metrics calculation, confusion matrices, and reporting.

hyperparameter_tuning.py

Grid/random search to optimize final model performance.

plot_results.py

Output visualizations for comparisons and summaries.

3. ML_notebook.ipynb

Main execution environment where the full workflow is run. It imports all functions from funcs/, executes them sequentially, and documents the findings.

4. README.md

Top-level description and usage instructions.

Notebook Workflow Summary

The notebook performs the following:

Data Import and Initial Inspection

Loads the dataset, examines structure, identifies missing values, and reviews variable types.

Preprocessing and Cleaning

Applies encoding, normalization, outlier handling, and trains/validation splits using functions from data_preprocessing.py.

Exploratory Data Analysis

Generates distribution plots, correlation heatmaps, and category-based comparisons to understand key drivers and trends.

Feature Selection

Reduces dimensionality through statistical and model-based methods to retain only the most informative predictors.

Clustering Analysis

Uses unsupervised techniques to identify behavior-based subgroups within the student population.

Model Training and Comparison

Trains multiple classifiers such as logistic regression, SVM, random forest, and evaluates performance.

Hyperparameter Optimization

Improves selected models using tuning strategies to enhance accuracy and generalizability.

Final Evaluation and Interpretation

Produces metrics, confusion matrices, and insights into which variables contribute most to predictions.

How to Use the Repository

Run ML_notebook.ipynb to reproduce the full analysis.
