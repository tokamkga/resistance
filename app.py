import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🏗️ Prédiction de Résistance du Béton",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    .feature-input {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    .comparison-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Charger et préparer les données"""
    df = pd.read_csv('streamlit_app/data/concrete.csv')
    return df

@st.cache_resource
def train_models(df):
    """Entraîner les modèles Random Forest et Deep Learning"""
    # Préparation des données
    X = df.drop('strength', axis=1)
    y = df['strength']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalisation pour le Deep Learning
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # 2. Deep Learning Model
    dl_model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    dl_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Entraînement silencieux
    dl_model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    dl_pred = dl_model.predict(X_test_scaled, verbose=0).flatten()
    
    # Métriques
    rf_metrics = {
        'mse': mean_squared_error(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'mae': mean_absolute_error(y_test, rf_pred),
        'r2': r2_score(y_test, rf_pred)
    }
    
    dl_metrics = {
        'mse': mean_squared_error(y_test, dl_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, dl_pred)),
        'mae': mean_absolute_error(y_test, dl_pred),
        'r2': r2_score(y_test, dl_pred)
    }
    
    return {
        'rf_model': rf_model,
        'dl_model': dl_model,
        'scaler': scaler,
        'rf_metrics': rf_metrics,
        'dl_metrics': dl_metrics,
        'X_test': X_test,
        'y_test': y_test,
        'rf_pred': rf_pred,
        'dl_pred': dl_pred
    }

def create_feature_importance_plot(model, feature_names):
    """Créer un graphique d'importance des features"""
    importance = model.feature_importances_
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        feature_df,
        x='importance',
        y='feature',
        orientation='h',
        title="Importance des Variables (Random Forest)",
        color='importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16,
        font=dict(size=12)
    )
    
    return fig

def create_prediction_comparison_plot(y_test, rf_pred, dl_pred):
    """Créer un graphique de comparaison des prédictions"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Random Forest', 'Deep Learning'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Random Forest
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=rf_pred,
            mode='markers',
            name='RF Predictions',
            marker=dict(color='#667eea', size=6, opacity=0.7)
        ),
        row=1, col=1
    )
    
    # Deep Learning
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=dl_pred,
            mode='markers',
            name='DL Predictions',
            marker=dict(color='#764ba2', size=6, opacity=0.7)
        ),
        row=1, col=2
    )
    
    # Ligne parfaite
    min_val = min(y_test.min(), rf_pred.min(), dl_pred.min())
    max_val = max(y_test.max(), rf_pred.max(), dl_pred.max())
    
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Ligne parfaite',
                line=dict(color='red', dash='dash'),
                showlegend=False if col == 2 else True
            ),
            row=1, col=col
        )
    
    fig.update_xaxes(title_text="Valeurs Réelles", row=1, col=1)
    fig.update_xaxes(title_text="Valeurs Réelles", row=1, col=2)
    fig.update_yaxes(title_text="Prédictions", row=1, col=1)
    fig.update_yaxes(title_text="Prédictions", row=1, col=2)
    
    fig.update_layout(
        height=500,
        title_text="Comparaison des Prédictions vs Valeurs Réelles",
        title_font_size=16
    )
    
    return fig

def main():
    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ Prédiction de Résistance du Béton</h1>
        <p>Application de Machine Learning pour prédire la résistance à la compression du béton</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données et entraînement des modèles
    with st.spinner('🔄 Chargement des données et entraînement des modèles...'):
        df = load_data()
        models_data = train_models(df)
    
    # Sidebar pour les paramètres d'entrée
    st.sidebar.markdown("## 🎛️ Paramètres d'Entrée")
    st.sidebar.markdown("Ajustez les valeurs pour faire une prédiction:")
    
    # Inputs pour les features
    cement = st.sidebar.slider(
        "🏗️ Ciment (kg/m³)",
        min_value=float(df['cement'].min()),
        max_value=float(df['cement'].max()),
        value=float(df['cement'].mean()),
        step=1.0
    )
    
    slag = st.sidebar.slider(
        "⚫ Laitier (kg/m³)",
        min_value=float(df['slag'].min()),
        max_value=float(df['slag'].max()),
        value=float(df['slag'].mean()),
        step=1.0
    )
    
    ash = st.sidebar.slider(
        "🌋 Cendres volantes (kg/m³)",
        min_value=float(df['ash'].min()),
        max_value=float(df['ash'].max()),
        value=float(df['ash'].mean()),
        step=1.0
    )
    
    water = st.sidebar.slider(
        "💧 Eau (kg/m³)",
        min_value=float(df['water'].min()),
        max_value=float(df['water'].max()),
        value=float(df['water'].mean()),
        step=1.0
    )
    
    superplastic = st.sidebar.slider(
        "🧪 Superplastifiant (kg/m³)",
        min_value=float(df['superplastic'].min()),
        max_value=float(df['superplastic'].max()),
        value=float(df['superplastic'].mean()),
        step=0.1
    )
    
    coarseagg = st.sidebar.slider(
        "🪨 Agrégats grossiers (kg/m³)",
        min_value=float(df['coarseagg'].min()),
        max_value=float(df['coarseagg'].max()),
        value=float(df['coarseagg'].mean()),
        step=1.0
    )
    
    fineagg = st.sidebar.slider(
        "⚪ Agrégats fins (kg/m³)",
        min_value=float(df['fineagg'].min()),
        max_value=float(df['fineagg'].max()),
        value=float(df['fineagg'].mean()),
        step=1.0
    )
    
    age = st.sidebar.slider(
        "📅 Âge (jours)",
        min_value=int(df['age'].min()),
        max_value=int(df['age'].max()),
        value=int(df['age'].mean()),
        step=1
    )
    
    # Prédictions
    input_data = np.array([[cement, slag, ash, water, superplastic, coarseagg, fineagg, age]])
    input_data_scaled = models_data['scaler'].transform(input_data)
    
    rf_prediction = models_data['rf_model'].predict(input_data)[0]
    dl_prediction = models_data['dl_model'].predict(input_data_scaled, verbose=0)[0][0]
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Résultats des prédictions
        st.markdown("## 🎯 Prédictions")
        
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            st.markdown(f"""
            <div class="prediction-card">
                <h3>🌲 Random Forest</h3>
                <h2>{rf_prediction:.2f} MPa</h2>
                <p>R² Score: {models_data['rf_metrics']['r2']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_col2:
            st.markdown(f"""
            <div class="prediction-card">
                <h3>🧠 Deep Learning</h3>
                <h2>{dl_prediction:.2f} MPa</h2>
                <p>R² Score: {models_data['dl_metrics']['r2']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Différence entre les modèles
        diff = abs(rf_prediction - dl_prediction)
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Analyse Comparative</h4>
            <p><strong>Différence entre les modèles:</strong> {diff:.2f} MPa</p>
            <p><strong>Moyenne des prédictions:</strong> {(rf_prediction + dl_prediction)/2:.2f} MPa</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Métriques des modèles
        st.markdown("## 📈 Performance des Modèles")
        
        st.markdown("""
        <div class="comparison-card">
            <h4>🌲 Random Forest</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_rf1, col_rf2 = st.columns(2)
        with col_rf1:
            st.metric("RMSE", f"{models_data['rf_metrics']['rmse']:.2f}")
            st.metric("R²", f"{models_data['rf_metrics']['r2']:.3f}")
        with col_rf2:
            st.metric("MAE", f"{models_data['rf_metrics']['mae']:.2f}")
            st.metric("MSE", f"{models_data['rf_metrics']['mse']:.2f}")
        
        st.markdown("""
        <div class="comparison-card">
            <h4>🧠 Deep Learning</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.metric("RMSE", f"{models_data['dl_metrics']['rmse']:.2f}")
            st.metric("R²", f"{models_data['dl_metrics']['r2']:.3f}")
        with col_dl2:
            st.metric("MAE", f"{models_data['dl_metrics']['mae']:.2f}")
            st.metric("MSE", f"{models_data['dl_metrics']['mse']:.2f}")
    
    # Graphiques d'analyse
    st.markdown("## 📊 Analyse des Modèles")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Comparaison Prédictions", "🌲 Importance des Variables", "📈 Distribution des Données"])
    
    with tab1:
        fig_comparison = create_prediction_comparison_plot(
            models_data['y_test'],
            models_data['rf_pred'],
            models_data['dl_pred']
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab2:
        fig_importance = create_feature_importance_plot(
            models_data['rf_model'],
            df.drop('strength', axis=1).columns
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Explication des variables importantes
        st.markdown("""
        <div class="metric-card">
            <h4>💡 Interprétation</h4>
            <p>Les variables les plus importantes pour prédire la résistance du béton sont généralement:</p>
            <ul>
                <li><strong>Âge:</strong> Plus le béton vieillit, plus il devient résistant</li>
                <li><strong>Ciment:</strong> Composant principal qui détermine la résistance</li>
                <li><strong>Eau:</strong> Le rapport eau/ciment est crucial</li>
                <li><strong>Superplastifiant:</strong> Améliore la workabilité et la résistance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        # Distribution de la variable cible
        fig_dist = px.histogram(
            df,
            x='strength',
            nbins=30,
            title="Distribution de la Résistance du Béton",
            color_discrete_sequence=['#667eea']
        )
        fig_dist.update_layout(
            xaxis_title="Résistance (MPa)",
            yaxis_title="Fréquence",
            showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Statistiques descriptives
        st.markdown("### 📋 Statistiques Descriptives")
        st.dataframe(df.describe().round(2), use_container_width=True)
    
    # Informations sur le dataset
    st.markdown("## 📚 À propos du Dataset")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Données</h4>
            <p><strong>Échantillons:</strong> {}</p>
            <p><strong>Variables:</strong> {}</p>
            <p><strong>Cible:</strong> Résistance (MPa)</p>
        </div>
        """.format(len(df), len(df.columns)-1), unsafe_allow_html=True)
    
    with info_col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Objectif</h4>
            <p>Prédire la résistance à la compression du béton en fonction de sa composition et de son âge.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🔬 Modèles</h4>
            <p><strong>Random Forest:</strong> Ensemble de 100 arbres</p>
            <p><strong>Deep Learning:</strong> Réseau dense 4 couches</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()