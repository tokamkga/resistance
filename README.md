# 🏗️ Application de Prédiction de Résistance du Béton

Une application Streamlit moderne utilisant Machine Learning et Deep Learning pour prédire la résistance à la compression du béton.

## 🚀 Fonctionnalités

### 🎯 Prédiction Interactive
- Interface utilisateur intuitive avec sliders pour ajuster les paramètres
- Prédictions en temps réel avec deux modèles:
  - **Random Forest**: Modèle d'ensemble robuste et interprétable
  - **Deep Learning**: Réseau de neurones pour capturer les relations complexes
- Comparaison automatique des prédictions entre modèles

### 📊 Analyse Exploratoire
- Visualisations interactives des données
- Matrice de corrélation
- Distribution des variables
- Analyse des relations entre variables
- Identification des outliers et insights

### 🤖 Détails des Modèles
- Performance détaillée des modèles (métriques train/test)
- Analyse des résidus
- Courbes d'apprentissage
- Historique d'entraînement du réseau de neurones
- Importance des variables (Random Forest)

### 🎯 Prédiction en Lot
- Upload de fichiers CSV pour prédictions multiples
- Template téléchargeable
- Validation automatique des données
- Visualisation des résultats
- Export des prédictions

## 🛠️ Technologies Utilisées

- **Streamlit**: Interface web interactive
- **Scikit-learn**: Random Forest et preprocessing
- **TensorFlow/Keras**: Deep Learning
- **Plotly**: Visualisations interactives
- **Pandas/NumPy**: Manipulation des données

## 📊 Dataset

Le dataset `concrete.csv` contient 1030 échantillons avec 8 variables d'entrée:

- **cement**: Ciment (kg/m³)
- **slag**: Laitier de haut fourneau (kg/m³)
- **ash**: Cendres volantes (kg/m³)
- **water**: Eau (kg/m³)
- **superplastic**: Superplastifiant (kg/m³)
- **coarseagg**: Agrégats grossiers (kg/m³)
- **fineagg**: Agrégats fins (kg/m³)
- **age**: Âge du béton (jours)

**Variable cible**: `strength` - Résistance à la compression (MPa)

## 🚀 Installation et Utilisation

### Prérequis
```bash
pip install -r requirements.txt
```

### Lancement de l'application
```bash
streamlit run streamlit_app/app.py
```

L'application sera accessible à l'adresse: `http://localhost:8501`

## 📈 Performance des Modèles

### Random Forest
- **Avantages**: Interprétable, robuste, pas de surapprentissage
- **R² Score**: ~0.90
- **RMSE**: ~6-8 MPa

### Deep Learning
- **Avantages**: Capture les relations non-linéaires complexes
- **Architecture**: 4 couches denses avec dropout
- **R² Score**: ~0.88-0.92
- **RMSE**: ~5-7 MPa

## 🎨 Design et UX

L'application met l'accent sur:
- **Design moderne**: Gradient de couleurs, cartes avec ombres
- **Expérience utilisateur**: Navigation intuitive, feedback visuel
- **Responsivité**: Adaptation à différentes tailles d'écran
- **Accessibilité**: Couleurs contrastées, texte lisible

## 📁 Structure du Projet

```
streamlit_app/
├── app.py                          # Application principale
├── data/
│   └── concrete.csv               # Dataset
├── pages/
│   ├── 1_📊_Analyse_Exploratoire.py
│   ├── 2_🤖_Détails_Modèles.py
│   └── 3_🎯_Prédiction_Batch.py
└── requirements.txt               # Dépendances
```

## 🔬 Méthodologie

### Préprocessing
- Normalisation des données pour le Deep Learning
- Division train/test (80/20)
- Validation des données d'entrée

### Modélisation
- **Random Forest**: 100 arbres, profondeur max 15
- **Deep Learning**: Architecture dense avec dropout pour éviter le surapprentissage
- Validation croisée et métriques multiples

### Évaluation
- MSE, RMSE, MAE, R²
- Analyse des résidus
- Comparaison visuelle des prédictions

## 🎯 Cas d'Usage

1. **Recherche et Développement**: Optimisation de formulations de béton
2. **Contrôle Qualité**: Prédiction de résistance avant tests physiques
3. **Éducation**: Compréhension des facteurs influençant la résistance
4. **Production**: Ajustement des compositions pour atteindre des résistances cibles

## 🚀 Améliorations Futures

- Intégration de modèles ensemble (stacking)
- Optimisation automatique des hyperparamètres
- Prédiction d'intervalles de confiance
- API REST pour intégration dans d'autres systèmes
- Base de données pour historique des prédictions

## 📞 Support

Pour toute question ou suggestion d'amélioration, n'hésitez pas à ouvrir une issue ou contribuer au projet.