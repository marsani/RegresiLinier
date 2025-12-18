import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# Page Config
st.set_page_config(page_title="Rice Price Dashboard", layout="wide", page_icon="🌾")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stPlotlyChart { background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .equation-box { background-color: #eef2f7; padding: 20px; border-left: 5px solid #2980b9; border-radius: 5px; margin: 10px 0; }
    h1, h2, h3 { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_and_preprocess(file_path):
    try:
        # Determine how to read based on input source
        if isinstance(file_path, str):
            df = pd.read_csv(file_path, sep=';')
            if len(df.columns) < 2:
                df = pd.read_csv(file_path, sep=',')
        else:
            df = pd.read_csv(file_path, sep=';')
            if len(df.columns) < 2:
                file_path.seek(0)
                df = pd.read_csv(file_path, sep=',')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # Preprocessing
    for col in df.columns:
        if col != 'Tanggal':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    
    # Standardize Tanggal
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Tanggal']).sort_values('Tanggal')
    
    # Use normalize() to keep datetime64[ns] dtype but zero out time
    # This is much more compatible with Arrow than converting to object/date
    df['Tanggal'] = df['Tanggal'].dt.normalize()
    
    if 'Nasional_Average' not in df.columns:
        provinces = [c for c in df.columns if c != 'Tanggal']
        df['Nasional_Average'] = df[provinces].mean(axis=1)
        
    return df

@st.cache_data
def run_regression(df, target_col):
    df_reg = df.copy()
    df_reg['Year'] = df_reg['Tanggal'].dt.year
    df_reg['Month'] = df_reg['Tanggal'].dt.month
    df_reg['Day'] = df_reg['Tanggal'].dt.day
    df_reg['DayOfWeek'] = df_reg['Tanggal'].dt.dayofweek
    df_reg['Prev_Price'] = df_reg[target_col].shift(1)
    df_reg = df_reg.dropna()
    
    features = ['Year', 'Month', 'Day', 'DayOfWeek', 'Prev_Price']
    X = df_reg[features]
    y = df_reg[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        eq = None
        if name == "Linear Regression":
            intercept = model.intercept_
            coefs = model.coef_
            eq = f"Y = {intercept:.2f}"
            for c, f in zip(coefs, features):
                sign = "+" if c >= 0 else "-"
                eq += f" {sign} {abs(c):.4f} \cdot {f}"
        
        results[name] = {
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "y_test": y_test,
            "y_pred": y_pred,
            "equation": eq
        }
    return results

# --- APP LAYOUT ---
st.title("🌾 Rice Price Analytics Dashboard")

# Sidebar - Data Input
st.sidebar.header("📁 Pengaturan Data")
uploaded_file = st.sidebar.file_uploader("Unggah CSV Dataset", type=["csv"])

DEFAULT_PATH = "Price Rice In Indonesia 2021-2024.csv"

if uploaded_file is not None:
    df = load_and_preprocess(uploaded_file)
elif os.path.exists(DEFAULT_PATH):
    with open(DEFAULT_PATH, 'rb') as f:
        df = load_and_preprocess(f)
else:
    st.warning("Silakan unggah dataset CSV untuk memulai.")
    df = None

if df is not None:
    menu = st.sidebar.radio("Navigasi", ["Dashboard & EDA", "Prediksi Regresi"])
    available_regions = df.columns.drop('Tanggal').tolist()

    if menu == "Dashboard & EDA":
        st.header("📊 Dashboard Overview")
        
        m1, m2, m3 = st.columns(3)
        current_avg = df['Nasional_Average'].iloc[-1]
        prev_avg = df['Nasional_Average'].iloc[-2]
        delta = ((current_avg - prev_avg) / prev_avg) * 100
        
        m1.metric("Harga Nasional Terkini", f"IDR {current_avg:,.2f}", f"{delta:+.2f}%")
        m2.metric("Rentang Waktu", f"{df['Tanggal'].min().year} - {df['Tanggal'].max().year}")
        m3.metric("Jumlah Wilayah", f"{len(available_regions)-1}")

        tab1, tab2, tab3 = st.tabs(["📈 Tren Per Wilayah", "⚖️ Perbandingan & Global", "📋 Dataset"])
        
        with tab1:
            selected_region = st.selectbox("Pilih Wilayah untuk Analisis", available_regions, index=available_regions.index('Nasional_Average'))
            fig_trend = px.line(df, x='Tanggal', y=selected_region, title=f"Tren Pergerakan Harga: {selected_region}", template="plotly_white")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            fig_dist = px.histogram(df, x=selected_region, nbins=50, title=f"Distribusi Harga: {selected_region}", color_discrete_sequence=['#e67e22'])
            st.plotly_chart(fig_dist, use_container_width=True)

        with tab2:
            st.subheader("📊 Perbandingan Antar Wilayah")
            target_list = [r for r in available_regions if r != 'Nasional_Average']
            selected_comparison = st.multiselect("Pilih Provinsi untuk Dibandingkan", options=target_list, default=target_list[:3])
            
            if selected_comparison:
                fig_comp = px.line(df, x='Tanggal', y=selected_comparison, title="Perbandingan Tren Harga Beras", template="plotly_white")
                st.plotly_chart(fig_comp, use_container_width=True)

            st.divider()
            st.subheader("🌐 Tren Seluruh Provinsi")
            with st.expander("Klik untuk melihat tren semua provinsi"):
                fig_global = px.line(df, x='Tanggal', y=target_list, title="Tren Harga Beras Seluruh Indonesia", template="plotly_white")
                st.plotly_chart(fig_global, use_container_width=True)

        with tab3:
            st.subheader("Sampel Data")
            # Convert Tanggal to string for display to avoid Arrow serialization error
            df_display = df.copy()
            df_display['Tanggal'] = df_display['Tanggal'].dt.strftime('%Y-%m-%d')
            st.dataframe(df_display, width='stretch')

    elif menu == "Prediksi Regresi":
        st.header("🤖 Machine Learning Prediction")
        predict_region = st.selectbox("Pilih Wilayah untuk Diprediksi", available_regions, index=available_regions.index('Nasional_Average'))
        
        results = run_regression(df, predict_region)
        
        cols = st.columns(len(results))
        for i, (name, metrics) in enumerate(results.items()):
            with cols[i]:
                st.markdown(f"#### {name}")
                st.metric("R² Score", f"{metrics['r2']:.4f}")
                st.write(f"MAE: **{metrics['mae']:.2f}**")
        
        st.divider()
        st.subheader("📝 Persamaan Regresi")
        if results['Linear Regression']['equation']:
            st.markdown(f"**Model: Linear Regression ({predict_region})**")
            st.latex(results['Linear Regression']['equation'])
        
        st.divider()
        selected_model = st.selectbox("Pilih Model untuk Visualisasi", list(results.keys()))
        res = results[selected_model]
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=res['y_test'].values, name="Aktual", line=dict(color='#2980b9')))
        fig_pred.add_trace(go.Scatter(y=res['y_pred'], name="Prediksi", line=dict(color='#c0392b', dash='dot')))
        fig_pred.update_layout(title=f"Aktual vs Prediksi: {selected_model} ({predict_region})", template="plotly_white")
        st.plotly_chart(fig_pred, use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("Enhanced Rice Price Analytics v2.0")
