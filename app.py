import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os



st.title("Greenhouse Gas Emission Prediction App")

# Step 1: Upload Excel file
excel_file = st.file_uploader("Upload the Excel file", type=["xlsx"])

if excel_file is not None:
    years = range(2010, 2017)
    all_data = []

    for year in years:
        try:
            df_com = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Commodity')
            df_ind = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Industry')

            df_com['Source'] = 'Commodity'
            df_ind['Source'] = 'Industry'
            df_com['Year'] = df_ind['Year'] = year

            df_com.columns = df_com.columns.str.strip()
            df_ind.columns = df_ind.columns.str.strip()

            df_com.rename(columns={'Commodity Code': 'Code', 'Commodity Name': 'Name'}, inplace=True)
            df_ind.rename(columns={'Industry Code': 'Code', 'Industry Name': 'Name'}, inplace=True)

            all_data.append(pd.concat([df_com, df_ind], ignore_index=True))
        except Exception as e:
            st.warning(f"Error processing year {year}: {e}")

    df = pd.concat(all_data, ignore_index=True)

    # Step 2: Data Cleaning
    df.drop(columns=['Unnamed: 7'], errors='ignore', inplace=True)

    substance_map = {'carbon dioxide': 0, 'methane': 1, 'nitrous oxide': 2, 'other GHGs': 3}
    unit_map = {'kg/2018 USD, purchaser price': 0, 'kg CO2e/2018 USD, purchaser price': 1}
    source_map = {'Commodity': 0, 'Industry': 1}

    df['Substance'] = df['Substance'].map(substance_map)
    df['Unit'] = df['Unit'].map(unit_map)
    df['Source'] = df['Source'].map(source_map)

    st.subheader("Raw Data Sample")
    st.write(df.head())

    st.subheader("Top 10 Emitting Industries")
    top_emitters = df[['Name', 'Supply Chain Emission Factors with Margins']].groupby('Name').mean().sort_values(
        'Supply Chain Emission Factors with Margins', ascending=False).head(10).reset_index()
    st.write(top_emitters)


    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x='Supply Chain Emission Factors with Margins',
        y='Name',
        data=top_emitters,
        palette='viridis',
        ax=ax
    )

    for i, value in enumerate(top_emitters['Supply Chain Emission Factors with Margins']):
        ax.text(value + 0.01, i, f'#{i + 1}', va='center', fontsize=10)

    ax.set_title("Top 10 Emitting Industries", fontsize=14, fontweight='bold')
    ax.set_xlabel("Emission Factor (kg CO2e/unit)")
    ax.set_ylabel("Industry")
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    st.pyplot(fig)

    # Step 3: Model Training
    df.drop(columns=['Name', 'Code', 'Year'], inplace=True)

    X = df.drop(columns=['Supply Chain Emission Factors with Margins'])
    y = df['Supply Chain Emission Factors with Margins']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Random Forest
    RF_model = RandomForestRegressor(random_state=42)
    RF_model.fit(X_train, y_train)
    RF_pred = RF_model.predict(X_test)
    RF_mse = mean_squared_error(y_test, RF_pred)
    RF_rmse = np.sqrt(RF_mse)
    RF_r2 = r2_score(y_test, RF_pred)

    # Linear Regression
    LR_model = LinearRegression()
    LR_model.fit(X_train, y_train)
    LR_pred = LR_model.predict(X_test)
    LR_mse = mean_squared_error(y_test, LR_pred)
    LR_rmse = np.sqrt(LR_mse)
    LR_r2 = r2_score(y_test, LR_pred)

    # Hyperparameter Tuning
    st.subheader("Training and Evaluation Results")
    param_grid = {
        'n_estimators': [100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_pred = best_model.predict(X_test)

    HP_mse = mean_squared_error(y_test, best_pred)
    HP_rmse = np.sqrt(HP_mse)
    HP_r2 = r2_score(y_test, best_pred)

    results_df = pd.DataFrame({
        'Model': ['Random Forest (Default)', 'Linear Regression', 'Random Forest (Tuned)'],
        'MSE': [RF_mse, LR_mse, HP_mse],
        'RMSE': [RF_rmse, LR_rmse, HP_rmse],
        'R2 Score': [RF_r2, LR_r2, HP_r2]
    })
    st.dataframe(results_df)

    # Save model
    if not os.path.exists('models'):
        os.mkdir('models')
    joblib.dump(best_model, 'models/LR_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    st.success("Best model and scaler saved successfully!")

else:
    st.info("Upload an Excel file to begin analysis.")
