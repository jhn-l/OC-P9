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
    # Charger les données
    df = pd.read_csv(file_path)
    
    # Convertir les colonnes nécessaires
    df['timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
    df['target'] = pd.to_numeric(df['Weekly_Sales'], errors='coerce')
    df['item_id'] = df['Store'] if 'Store' in df.columns else 1

    # Garder les colonnes nécessaires (avec covariates)
    covariates = ['IsHoliday', 'Super_Bowl', 'Labor_Day', 'Thanksgiving', 'Christmas']
    df = df[['item_id', 'timestamp', 'target'] + covariates].dropna()

    # Trier les données
    df = df.sort_values(by=['item_id', 'timestamp']).reset_index(drop=True)

    # Éliminer les doublons dans `timestamp` par `item_id`
    df = df.groupby(['item_id', 'timestamp'], as_index=False).mean()

    # Rééchantillonnage (fréquence hebdomadaire)
    freq = 'W'
    regular_data = []
    for item_id, group in df.groupby('item_id'):
        group = group.set_index('timestamp').asfreq(freq, method='pad')  # Remplir les valeurs manquantes
        group['item_id'] = item_id  # Ajouter l'ID
        regular_data.append(group.reset_index())

    # Combiner les données rééchantillonnées
    df_regular = pd.concat(regular_data, ignore_index=True)

    df_regular[covariates] = df_regular[covariates].apply(lambda x: x.astype(int).astype(object)) # Convertir en type catégorique

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

# Prévisions temporelles (Graphique interactif 1)
st.header("Prévisions temporelles")
st.markdown(f"Prévisions pour le magasin sélectionné : **{selected_item}**")
predictions = all_predictions[all_predictions['item_id'] == selected_item]

# Tracer les prévisions
fig, ax = plt.subplots(figsize=(12, 6))

#Tracer les données observées
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

# Analyse de la saisonnalité (Graphique interactif 2)
st.header("Analyse de la saisonnalité")
st.markdown("Explorez les composantes : Tendance, Saisonnière ou Résidus.")

# Décomposer les données
selected_data = ts_df.loc[selected_item]["target"]
decomposed = seasonal_decompose(selected_data, period=52)

# Options pour les composantes
component = st.selectbox(
    "Choisissez une composante à explorer :",
    ["Tendance", "Saisonnière", "Résidus"]
)

# Explications pour chaque composante
component_explanations = {
    "Tendance": (
        "La composante Tendance représente l'évolution générale des ventes sur une longue période. "
        "Elle permet d'identifier si les ventes augmentent, diminuent ou restent stables dans le temps."
    ),
    "Saisonnière": (
        "La composante Saisonnière capture les variations périodiques répétées, comme les effets "
        "des saisons ou des événements récurrents (par exemple, Noël ou les vacances scolaires). "
        "Elle est utile pour comprendre les schémas cycliques."
    ),
    "Résidus": (
        "La composante Résidus représente les variations imprévisibles qui ne peuvent pas être expliquées "
        "par la tendance ou la saisonnalité. Elle reflète les fluctuations dues à des événements aléatoires ou exceptionnels."
    ),
}

# Afficher l'explication de la composante sélectionnée
st.markdown(f"**À quoi sert cette composante ?** {component_explanations[component]}")


# Tracer la composante sélectionnée
fig, ax = plt.subplots(figsize=(12, 6))
if component == "Tendance":
    ax.plot(decomposed.trend, label="Tendance", color="green")
elif component == "Saisonnière":
    ax.plot(decomposed.seasonal, label="Saisonnière", color="orange")
elif component == "Résidus":
    ax.plot(decomposed.resid, label="Résidus", color="red")

ax.set_title(f"Composante : {component}")
ax.legend()
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
