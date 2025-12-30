Real Estate Price Prediction System

A machine learning–based decision support tool for estimating residential home sale prices using structured property data.

Overview

This project implements an end-to-end machine learning pipeline to predict residential real estate sale prices based on key property attributes. The goal is to provide a clear, repeatable, data-driven pricing workflow that supports scenario analysis and market insight.

The system is designed to augment professional judgment, not replace it, by producing objective and defensible price estimates derived from historical data.

Dataset

The model is trained on a structured residential real estate dataset containing approximately 1,000 South Florida property listings. The data includes a mix of real and synthetic records to ensure sufficient sample size while maintaining realistic feature distributions.

Key features include:
- Listing price
- Sale price
- Square footage
- Bedrooms and bathrooms
- Lot size
- Year built
- ZIP code
- Pool availability

The dataset was cleaned and validated to ensure consistent data types, valid value ranges, and suitability for machine learning.

Modeling Approach

Multiple regression models were evaluated, including:
- Linear Regression (baseline)
- Random Forest Regression

Models were compared using standard regression metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

The Random Forest model was selected due to its improved predictive accuracy and ability to capture non-linear relationships between property features and sale price.

Feature importance analysis identified listing price, square footage, and year built as among the most influential predictors.

Project Structure

- notebooks/ contains the full data preparation, analysis, and model training pipeline
- model/ contains the trained Random Forest model used by the application
- src/app.py provides an interactive Streamlit application for generating price predictions and visual insights

Running Locally

To install dependencies:

	pip install -r requirements.txt

To launch the Streamlit application:

	streamlit run src/app.py

Reproducibility

The full preprocessing and training pipeline is implemented in the notebook and can be run end-to-end to retrain the model.  
A trained model artifact is included in the repository for demonstration purposes so the application runs out of the box.

Technologies Used

- Python
- pandas, NumPy
- scikit-learn
- Jupyter Notebook
- Streamlit
- matplotlib, seaborn

Notes

This project was originally developed as a computer science capstone and has been adapted into a professional portfolio project. Academic submission artifacts have been removed in favor of clarity, usability, and code transparency.
