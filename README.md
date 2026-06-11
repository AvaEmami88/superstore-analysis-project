# Sales Data Analysis Project

A comprehensive data analysis and machine learning project for retail sales data, featuring EDA, regression modeling, KNN classification, and automated reporting.

## 📋 Overview

This project processes retail sales data (Superstore dataset) to extract business insights, build predictive models, and generate reports. The pipeline handles data cleaning, feature engineering, exploratory data analysis, regression (linear & polynomial), KNN classification, and visualization.

## 🏗️ Project Structure
├── data_loader.py # CSV loading and schema validation
├── data_cleaner.py # Data cleaning (nulls, duplicates, dtypes, encoding)
├── feature_engineer.py # Feature creation (month/year, profit margin, aggregation)
├── sales_analyzer.py # EDA and sales analysis (top products, trends, heatmaps)
├── regression_model.py # Linear and polynomial regression models
├── knn_classifier.py # KNN classifier for profitability prediction
├── report_pipeline.py # Orchestrates all analysis and reporting
├── visualizer.py # Visualization utilities
├── main.py # Main execution script
├── Superstore.csv # Source dataset
└── output_for_powerbi.csv # Aggregated data for Power BI

## 🚀 Features

### 1. Data Loading & Validation
- Load CSV with encoding handling
- Validate schema, null counts, and data types

### 2. Data Cleaning
- Remove duplicates and null values
- Fix date formats
- Encode categorical variables (Category, Region)

### 3. Feature Engineering
- Extract month and year from Order Date
- Calculate profit margin (Profit/Sales)
- Create monthly sales aggregation

### 4. Exploratory Data Analysis
- Top 6 products by profit
- Sales by region
- Monthly sales trends
- Monthly sales heatmap (Year vs Month)

### 5. Regression Modeling
- Linear regression on monthly sales
- Polynomial regression (degrees 2 and 3)
- Evaluation metrics: R² and MSE

### 6. Classification (KNN)
- Predict if a transaction is profitable (Profit > 0)
- Test k-values: 3, 5, 7
- Accuracy scoring

### 7. Reporting
- Automated report pipeline
- CSV export for Power BI integration
- Results saved to `results.txt`

## 📊 Key Insights

| Metric | Value |
|--------|-------|
| Best KNN Accuracy | 93.5% (k=7) |
| Linear Regression R² | -0.07 (poor) |
| Polynomial (deg 2) R² | 0.56 |
| Polynomial (deg 3) R² | 0.58 |

### Top Products by Profit
1. Canon imageCLASS 2200 Advanced Copier - $25,200
2. Fellowes PB500 Electric Punch - $7,753
3. Hewlett Packard LaserJet 3310 Copier - $6,984

### Sales by Region
- West: $725,458
- East: $678,781
- Central: $501,240
- South: $391,722

## 🔧 Requirements

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
# Run complete pipeline
python main.py

# Or run individual components
from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from report_pipeline import ReportPipeline

# Quick analysis
pipeline = ReportPipeline("Superstore.csv")
results = pipeline.run_all()
pipeline.save_results()
pipeline.generate_csv()
