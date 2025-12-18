import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# Set style for plots
sns.set_theme(style="whitegrid")

def preprocess_data(file_path):
    print("--- Step 1: Preprocessing ---")
    # Load data
    df = pd.read_csv(file_path, sep=';')
    
    # Handle missing values: Interpolate and then backfill/forward fill
    # Converting to numeric first (some might be strings)
    for col in df.columns:
        if col != 'Tanggal':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Pre-interpolation: count nulls
    print(f"Initial null count: {df.isnull().sum().sum()}")
    
    # Interpolate
    df = df.interpolate(method='linear', limit_direction='both')
    
    # Final check for nulls
    print(f"Final null count: {df.isnull().sum().sum()}")
    
    # Convert 'Tanggal' to datetime
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')
    df = df.sort_values('Tanggal')
    
    # Add National Average Feature
    provinces = df.columns.drop('Tanggal')
    df['Nasional_Average'] = df[provinces].mean(axis=1)
    
    print("Preprocessing completed.\n")
    return df

def perform_eda(df):
    print("--- Step 2: Exploratory Data Analysis ---")
    
    # Summary stats
    print("Summary Statistics (Nasional Average):")
    print(df['Nasional_Average'].describe())
    
    # 1. Time-series plot of National Average Price
    plt.figure(figsize=(12, 6))
    plt.plot(df['Tanggal'], df['Nasional_Average'], color='blue', label='National Average')
    plt.title('Indonesia Rice Price Trend (2021-2024)')
    plt.xlabel('Date')
    plt.ylabel('Price (IDR)')
    plt.legend()
    plt.savefig('rice_price_trend.png')
    print("Saved 'rice_price_trend.png'")
    
    # 2. Distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Nasional_Average'], kde=True, color='green')
    plt.title('Distribution of Rice Prices')
    plt.savefig('rice_price_dist.png')
    print("Saved 'rice_price_dist.png'")
    
    # 3. Correlation Heatmap (Sample of 10 provinces to keep it readable)
    plt.figure(figsize=(12, 10))
    top_provinces = df.columns[1:11]
    sns.heatmap(df[top_provinces].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap (Sample of 10 Provinces)')
    plt.savefig('province_corr_heatmap.png')
    print("Saved 'province_corr_heatmap.png'")
    
    print("EDA completed.\n")

def perform_regression(df):
    print("--- Step 3: Regression Algorithms ---")
    
    # Feature Engineering for Regression
    df_reg = df.copy()
    df_reg['Year'] = df_reg['Tanggal'].dt.year
    df_reg['Month'] = df_reg['Tanggal'].dt.month
    df_reg['Day'] = df_reg['Tanggal'].dt.day
    df_reg['DayOfWeek'] = df_reg['Tanggal'].dt.dayofweek
    
    # Log transformation of price sometimes helps, but we'll try linear first
    # Lagrangian/Lag features: use previous day price to predict current day
    df_reg['Prev_Price'] = df_reg['Nasional_Average'].shift(1)
    df_reg = df_reg.dropna() # lose first row due to shift
    
    X = df_reg[['Year', 'Month', 'Day', 'DayOfWeek', 'Prev_Price']]
    y = df_reg['Nasional_Average']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({
            "Model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        })
        
        print(f"{name} Results:")
        print(f"  R2: {r2:.4f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}\n")
        
        # Plot prediction for visual check (sample)
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted Rice Price ({name})')
        plt.legend()
        plt.savefig(f'pred_{name.replace(" ", "_").lower()}.png')
    
    print("Regression modeling completed.\n")
    return results

if __name__ == "__main__":
    file_path = "Price Rice In Indonesia 2021-2024.csv"
    if os.path.exists(file_path):
        data = preprocess_data(file_path)
        perform_eda(data)
        perform_regression(data)
    else:
        print(f"Error: File '{file_path}' not found.")
