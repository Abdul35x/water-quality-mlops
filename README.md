# Water Quality Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

# Overview
This project demonstrates a production-grade Machine Learning pipeline designed to predict water potability based on chemical properties. Unlike standard academic scripts, this repository focuses on MLOps best practices, including modularity, reproducibility, type safety, and containerization.

The pipeline processes raw water quality data, handles missing values via robust imputation strategies, and trains a Random Forest Classifier to detect potable water samples.

# Key Engineering Features
*Scikit-Learn Pipelines:* Prevents data leakage by encapsulating preprocessing logic (imputation, scaling, encoding) within the cross-validation fold.

*Type Hinting & Docstrings:* Fully typed codebase (Python 3.9+) for better maintainability and static analysis.

*Dockerized Environment:* Ensures the application runs identically on any machine, eliminating "it works on my machine" issues.

*Robust Error Handling:* Implements `try/except` blocks and professional logging instead of standard `print` statements.

*Unit Testing:* Includes `pytest` fixtures and mocks to test data cleaning logic in isolation without dependencies on the file system.

# Tech Stack
*Language:* Python 3.9
*ML Framework:* Scikit-Learn
*Data Processing:* Pandas, NumPy
*Testing:* Pytest
*Containerization:* Docker

# Project Structure
```text
.
├── main.py                 # Core pipeline logic (ETL -> Train -> Evaluate)
├── test_main.py            # Unit tests with mocking and fixtures
├── Dockerfile              # Multi-layer Docker build instruction
├── requirements.txt        # Pinned dependencies
├── waterquality.csv        # Dataset (Semicolon separated)
└── README.md               # Documentation