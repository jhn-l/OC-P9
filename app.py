import streamlit as st
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Dashboard de prévisions", layout="wide")

# Charger les données
@st.cache_data
def load_data(file_path="clean_data.csv"):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
    df['target'] = pd.to_numeric(df['Weekly_Sales'], errors='coerce')
    df['item_id'] = df['Store'] if 'Store' in df.columns else 1
    df = df[['item_id', 'timestamp', 'target']].dropna()
    df = df.sort_values(by=['item_id', 'timestamp']).reset_index(drop=True)
    df = df.groupby(['item_id', 'timestamp'], as_index=False).agg({'target': 'mean'})
    freq = 'W'
    regular_data = []
    for item_id, group in df.groupby('item_id'):
        group = group.set_index('timestamp').asfreq(freq, method='pad')
        group['item_id'] = item_id
        regular_data.append(group.reset_index())
    df_regular = pd.concat(regular_data, ignore_index=True)
    ts_df = TimeSeriesDataFrame.from_data_frame(df_regular, id_column="item_id", timestamp_column="timestamp")
    return ts_df

@st.cache_data
def load_predictions(file_path="all_predictions.csv"):
    return pd.read_csv(file_path)

# Charger les données et les prévisions
ts_df = load_data()
all_predictions = load_predictions()

# Titre de l'application
st.title("Dashboard de prévisions des ventes")
st.markdown("""
Bienvenue sur le tableau de bord interactif ! Ici, vous pouvez :
1. Explorer les prévisions temporelles pour différents magasins.
2. Analyser la décomposition saisonnière.
3. Évaluer la précision des prévisions avec le MAPE.

Les graphiques respectent les critères d'accessibilité WCAG, avec des textes lisibles et des contrastes adaptés.
""")

# Sélection d'un magasin
item_ids = ts_df.item_ids.unique()
selected_item = st.selectbox("Sélectionnez un magasin :", item_ids)

# Prévisions temporelles
st.header("Prévisions temporelles")
st.markdown(f"Prévisions pour le magasin sélectionné : **{selected_item}**")
predictions = all_predictions[all_predictions['item_id'] == selected_item]

# Tracer les prévisions
fig, ax = plt.subplots(figsize=(12, 6))

# Tracer les données observées
ax.plot(
    ts_df.loc[selected_item].index,  # Index pour les dates
    ts_df.loc[selected_item]["target"], 
    label="Observé"
)

# Tracer les prévisions
ax.plot(
    pd.to_datetime(predictions["timestamp"]), 
    predictions["mean"], 
    label="Prévision", 
    linestyle="--", 
    color="orange"
)

# Ajouter l'intervalle de confiance
ax.fill_between(
    pd.to_datetime(predictions["timestamp"]),
    predictions["0.1"],
    predictions["0.9"],
    color="orange",
    alpha=0.2,
    label="Intervalle de confiance"
)

# Ajuster les limites de l'axe x pour ne pas inclure 1970
start_date = ts_df.loc[selected_item].index.min()
end_date = ts_df.loc[selected_item].index.max()
ax.set_xlim([start_date, end_date])

# Ajouter des titres et une légende
ax.set_title(f"Prévisions pour le magasin {selected_item}")
ax.legend()
plt.tight_layout()

# Afficher dans Streamlit
st.pyplot(fig)

# Décomposition saisonnière
st.header("Décomposition saisonnière")
st.markdown("Analyse des composantes saisonnières, de tendance et des résidus.")
selected_data = ts_df.loc[selected_item]["target"]
decomposed = seasonal_decompose(selected_data, period=52)
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposed.observed.plot(ax=axes[0], title="Observé")
decomposed.trend.plot(ax=axes[1], title="Tendance")
decomposed.seasonal.plot(ax=axes[2], title="Saisonnalité")
decomposed.resid.plot(ax=axes[3], title="Résidus")
plt.tight_layout()
st.pyplot(fig)

# Calcul du MAPE
st.header("Performance des prévisions")
st.markdown("**MAPE (Mean Absolute Percentage Error)** pour évaluer la précision des prévisions.")
actuals = ts_df.loc[selected_item]["target"][-40:]
forecast = predictions["mean"]
mape = mean_absolute_percentage_error(actuals, forecast)
st.metric(label="MAPE (%)", value=f"{mape*100:.2f}")

# Ajout des critères d'accessibilité
# st.markdown("""
# ### Critères d'accessibilité appliqués :
# - **Contenu non textuel :** Les graphiques incluent des titres et descriptions.
# - **Utilisation de la couleur :** Les graphiques utilisent des contrastes élevés.
# - **Contraste :** Contrastes suffisants pour le texte et les visuels.
# - **Redimensionnement :** Les textes et graphiques s'adaptent à la taille de l'écran.
# - **Titres clairs :** Chaque section est accompagnée d'un titre descriptif.
# """)
