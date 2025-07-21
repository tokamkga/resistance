import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🎯 Prédiction Batch",
    page_icon="🎯",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .batch-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .upload-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 2px dashed #667eea;
        margin: 1rem 0;
        text-align: center;
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Charger les données"""
    return pd.read_csv('streamlit_app/data/concrete.csv')

@st.cache_resource
def train_models_for_batch(df):
    """Entraîner les modèles pour les prédictions batch"""
    X = df.drop('strength', axis=1)
    y = df['strength']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Deep Learning
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
    
    dl_model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    return {
        'rf_model': rf_model,
        'dl_model': dl_model,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }

def create_template_file():
    """Créer un fichier template pour l'upload"""
    template_data = {
        'cement': [540.0, 332.5, 198.6],
        'slag': [0.0, 142.5, 132.4],
        'ash': [0.0, 0.0, 0.0],
        'water': [162.0, 228.0, 192.0],
        'superplastic': [2.5, 0.0, 0.0],
        'coarseagg': [1040.0, 932.0, 978.4],
        'fineagg': [676.0, 594.0, 825.5],
        'age': [28, 270, 360]
    }
    
    return pd.DataFrame(template_data)

def validate_input_data(df, required_columns):
    """Valider les données d'entrée"""
    errors = []
    
    # Vérifier les colonnes
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Colonnes manquantes: {', '.join(missing_cols)}")
    
    # Vérifier les valeurs numériques
    for col in required_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"La colonne '{col}' doit contenir des valeurs numériques")
            
            if df[col].isnull().any():
                errors.append(f"La colonne '{col}' contient des valeurs manquantes")
            
            if (df[col] < 0).any():
                errors.append(f"La colonne '{col}' contient des valeurs négatives")
    
    return errors

def create_prediction_visualization(df_results):
    """Créer une visualisation des résultats de prédiction"""
    fig = go.Figure()
    
    # Random Forest
    fig.add_trace(go.Scatter(
        x=df_results.index,
        y=df_results['RF_Prediction'],
        mode='lines+markers',
        name='Random Forest',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    # Deep Learning
    fig.add_trace(go.Scatter(
        x=df_results.index,
        y=df_results['DL_Prediction'],
        mode='lines+markers',
        name='Deep Learning',
        line=dict(color='#764ba2', width=3),
        marker=dict(size=8)
    ))
    
    # Moyenne
    fig.add_trace(go.Scatter(
        x=df_results.index,
        y=df_results['Average_Prediction'],
        mode='lines+markers',
        name='Moyenne',
        line=dict(color='#ff6b6b', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Prédictions de Résistance par Échantillon",
        xaxis_title="Numéro d'Échantillon",
        yaxis_title="Résistance Prédite (MPa)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def main():
    # En-tête
    st.markdown("""
    <div class="batch-header">
        <h1>🎯 Prédiction en Lot (Batch)</h1>
        <p>Téléchargez un fichier CSV pour obtenir des prédictions sur plusieurs échantillons</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des modèles
    with st.spinner('🔄 Chargement des modèles...'):
        df = load_data()
        models_data = train_models_for_batch(df)
    
    # Instructions et template
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📋 Instructions")
        st.markdown("""
        <div class="result-card">
            <h4>📝 Format du fichier CSV</h4>
            <p>Votre fichier doit contenir les colonnes suivantes (dans n'importe quel ordre):</p>
            <ul>
                <li><strong>cement:</strong> Quantité de ciment (kg/m³)</li>
                <li><strong>slag:</strong> Quantité de laitier (kg/m³)</li>
                <li><strong>ash:</strong> Quantité de cendres volantes (kg/m³)</li>
                <li><strong>water:</strong> Quantité d'eau (kg/m³)</li>
                <li><strong>superplastic:</strong> Quantité de superplastifiant (kg/m³)</li>
                <li><strong>coarseagg:</strong> Quantité d'agrégats grossiers (kg/m³)</li>
                <li><strong>fineagg:</strong> Quantité d'agrégats fins (kg/m³)</li>
                <li><strong>age:</strong> Âge du béton (jours)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 📥 Fichier Template")
        
        # Créer et afficher le template
        template_df = create_template_file()
        st.dataframe(template_df, use_container_width=True)
        
        # Bouton de téléchargement du template
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger Template",
            data=csv_template,
            file_name="template_concrete.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Zone d'upload
    st.markdown("## 📤 Téléchargement de Fichier")
    
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV",
        type=['csv'],
        help="Téléchargez un fichier CSV contenant les données de composition du béton"
    )
    
    if uploaded_file is not None:
        try:
            # Lire le fichier
            input_df = pd.read_csv(uploaded_file)
            
            st.markdown("### 📊 Aperçu des Données Téléchargées")
            st.dataframe(input_df.head(), use_container_width=True)
            
            # Validation
            errors = validate_input_data(input_df, models_data['feature_names'])
            
            if errors:
                st.error("❌ Erreurs dans le fichier:")
                for error in errors:
                    st.error(f"• {error}")
            else:
                st.success(f"✅ Fichier valide! {len(input_df)} échantillons détectés.")
                
                # Prédictions
                with st.spinner('🔮 Génération des prédictions...'):
                    # Préparer les données
                    X_input = input_df[models_data['feature_names']]
                    X_input_scaled = models_data['scaler'].transform(X_input)
                    
                    # Prédictions Random Forest
                    rf_predictions = models_data['rf_model'].predict(X_input)
                    
                    # Prédictions Deep Learning
                    dl_predictions = models_data['dl_model'].predict(X_input_scaled, verbose=0).flatten()
                    
                    # Créer le dataframe de résultats
                    results_df = input_df.copy()
                    results_df['RF_Prediction'] = rf_predictions
                    results_df['DL_Prediction'] = dl_predictions
                    results_df['Average_Prediction'] = (rf_predictions + dl_predictions) / 2
                    results_df['Difference'] = np.abs(rf_predictions - dl_predictions)
                
                # Affichage des résultats
                st.markdown("## 🎯 Résultats des Prédictions")
                
                # Métriques globales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "📊 Échantillons",
                        len(results_df)
                    )
                
                with col2:
                    st.metric(
                        "🌲 RF Moyenne",
                        f"{rf_predictions.mean():.1f} MPa"
                    )
                
                with col3:
                    st.metric(
                        "🧠 DL Moyenne",
                        f"{dl_predictions.mean():.1f} MPa"
                    )
                
                with col4:
                    st.metric(
                        "📏 Différence Moy.",
                        f"{results_df['Difference'].mean():.1f} MPa"
                    )
                
                # Visualisation
                fig_predictions = create_prediction_visualization(results_df)
                st.plotly_chart(fig_predictions, use_container_width=True)
                
                # Tableau des résultats
                st.markdown("### 📋 Tableau Détaillé des Résultats")
                
                # Formater le dataframe pour l'affichage
                display_df = results_df.round(2)
                st.dataframe(display_df, use_container_width=True)
                
                # Analyse des résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    # Statistiques des prédictions
                    st.markdown("""
                    <div class="result-card">
                        <h4>📈 Statistiques des Prédictions</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    stats_df = pd.DataFrame({
                        'Random Forest': [
                            f"{rf_predictions.min():.1f}",
                            f"{rf_predictions.max():.1f}",
                            f"{rf_predictions.mean():.1f}",
                            f"{rf_predictions.std():.1f}"
                        ],
                        'Deep Learning': [
                            f"{dl_predictions.min():.1f}",
                            f"{dl_predictions.max():.1f}",
                            f"{dl_predictions.mean():.1f}",
                            f"{dl_predictions.std():.1f}"
                        ]
                    }, index=['Min (MPa)', 'Max (MPa)', 'Moyenne (MPa)', 'Écart-type'])
                    
                    st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    # Distribution des différences
                    fig_diff = px.histogram(
                        results_df,
                        x='Difference',
                        nbins=20,
                        title="Distribution des Différences entre Modèles",
                        color_discrete_sequence=['#667eea']
                    )
                    fig_diff.update_layout(
                        xaxis_title="Différence Absolue (MPa)",
                        yaxis_title="Fréquence",
                        height=300
                    )
                    st.plotly_chart(fig_diff, use_container_width=True)
                
                # Téléchargement des résultats
                st.markdown("### 💾 Téléchargement des Résultats")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV complet
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger Résultats Complets (CSV)",
                        data=csv_results,
                        file_name="predictions_concrete_strength.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # CSV résumé
                    summary_df = results_df[['RF_Prediction', 'DL_Prediction', 'Average_Prediction', 'Difference']]
                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger Résumé (CSV)",
                        data=csv_summary,
                        file_name="predictions_summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Recommandations
                st.markdown("""
                <div class="result-card">
                    <h4>💡 Recommandations</h4>
                    <ul>
                        <li><strong>Différence faible (&lt;5 MPa):</strong> Les deux modèles sont en accord, prédiction fiable</li>
                        <li><strong>Différence modérée (5-10 MPa):</strong> Vérifier les valeurs d'entrée, utiliser la moyenne</li>
                        <li><strong>Différence élevée (&gt;10 MPa):</strong> Données potentiellement hors distribution, prudence requise</li>
                        <li><strong>Valeurs extrêmes:</strong> Vérifier la cohérence avec les données d'entraînement</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier: {str(e)}")
            st.info("💡 Vérifiez que votre fichier est au format CSV et contient les bonnes colonnes.")
    
    else:
        # Affichage de la zone d'upload vide
        st.markdown("""
        <div class="upload-card">
            <h3>📤 Aucun fichier sélectionné</h3>
            <p>Téléchargez un fichier CSV pour commencer les prédictions</p>
            <p>Utilisez le template ci-dessus comme exemple</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()