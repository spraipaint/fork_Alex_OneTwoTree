```@meta
CurrentModule = OneTwoTree
```

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nichtJakob.github.io/OneTwoTree.jl/dev/)
[![Build Status](https://github.com/nichtJakob/OneTwoTree.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/nichtJakob/OneTwoTree.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/nichtJakob/OneTwoTree.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nichtJakob/OneTwoTree.jl)

## Brief Explanation

[Decision Trees](https://en.wikipedia.org/wiki/Decision_tree) are a supervised learning algorithm used for classification and regression tasks. They split the data into subsets based on feature values, forming a tree-like structure where each internal node represents a decision based on a feature, and each leaf node represents a predicted outcome.

[Random Forests](https://en.wikipedia.org/wiki/Random_forest) improve on Decision Trees by creating an ensemble of multiple decision trees, each trained on a random subset of the data. The final prediction is made by averaging the outputs of all trees (for regression) or using a majority vote (for classification), which helps reduce overfitting and improves model accuracy.

## 🛠️ Prerequisites

| Prerequisite | Version | Installation Guide | Required |
|--------------|---------|--------------------|----------|
| Julia       | 1.10    | [![Julia](https://img.shields.io/badge/Julia-v1.10-blue)](https://julialang.org/downloads/) | ✅ |


## Explanation of Folders and Files 
Explanation of Folders and Files
- `docs/`
  - Contains everything related to documentation.
  - `make.jl`: Script to build the documentation using Documenter.jl.
  - `Project.toml` & Manifest.toml: Separate environment for documentation dependencies.
  - `src/`: Markdown files structured for different sections of the documentation.
  - `index.md`: Main landing page for the documentation.
  - `functions.md`: Lists functions and usage.
  - `examples/`: Specific examples for classification and regression use cases.
  - `api/`: Detailed API reference for classifiers, regressors, and utility functions.
  - `assets/`: Store images or other resources for the documentation.
- `build/`
  - Contains the generated documentation output (ignored by Git).
- `src/`
  - The main Julia module for OneTwoTree resides here. Documentation and code examples will often reference these files.
- `test/`
  - Unit tests ensure code correctness and complement examples in the documentation.
- `README.md`
    - A concise overview of the project for GitHub visitors. It should link to the generated documentation hosted (e.g., on GitHub Pages).


## Index for Documentation

Documentation for [OneTwoTree](https://github.com/nichtJakob/OneTwoTree.jl).

```@index
```
