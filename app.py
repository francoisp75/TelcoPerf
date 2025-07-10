
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re

@st.cache
def load_data():
    return pd.read_excel("Base_v4.xlsx", sheet_name="CA")

df_raw = load_data()
df = df_raw[df_raw['indicator'].str.lower() == 'growth']

# --- Extraire les trimestres et les trier ---
quarter_cols = [col for col in df.columns if re.match(r'\dQ\d{2}', str(col))]

def quarter_sort_key(q):
    match = re.match(r"([1-4])Q(\d{2})", q)
    if match:
        quarter_num, year_suffix = match.groups()
        return int("20" + year_suffix) * 10 + int(quarter_num)
    return 0

quarter_cols = sorted(quarter_cols, key=quarter_sort_key)

# --- UI ---
st.title("📊 Benchmark - Croissance CA par trimestre")

selected_operators = st.multiselect("👥 Choisir les opérateurs", df['operator'].dropna().unique())

# Slider d'index avec affichage des noms de trimestres sélectionnés
min_idx = 0
max_idx = len(quarter_cols) - 1
default_range = (max(0, max_idx - 3), max_idx)

quarter_range = st.slider(
    "📆 Choisir la plage de trimestres",
    min_value=min_idx,
    max_value=max_idx,
    value=default_range,
    step=1
)

selected_quarters = quarter_cols[quarter_range[0]:quarter_range[1] + 1]

# Affichage des trimestres sélectionnés
selected_label = " → ".join([selected_quarters[0], selected_quarters[-1]]) if selected_quarters else "Aucun"
st.markdown(f"**Trimestres sélectionnés :** {selected_label}")

# --- Vérifications ---
if not selected_operators or not selected_quarters:
    st.info("Veuillez sélectionner au moins un opérateur et une plage de trimestres.")
    st.stop()

# --- Affichage des graphiques ---
for op in selected_operators:
    df_op = df[df['operator'] == op]

    data = {
        "Trimestre": [],
        "Europe": [],
        "Non Europe": [],
        "Group": []
    }

    for quarter in selected_quarters:
        data["Trimestre"].append(quarter)

        europe_val = df_op[df_op['Scope'] == 'Europe'][quarter].values
        noneurope_val = df_op[df_op['Scope'] == 'Non Europe'][quarter].values
        group_val = df_op[df_op['Scope'] == 'Group'][quarter].values

        data["Europe"].append(europe_val[0] if len(europe_val) > 0 else None)
        data["Non Europe"].append(noneurope_val[0] if len(noneurope_val) > 0 else None)
        data["Group"].append(group_val[0] if len(group_val) > 0 else None)

    df_plot = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_plot['Trimestre'], y=df_plot['Europe'], name="Europe", marker_color="blue",
    textposition="outside"))
    fig.add_trace(go.Bar(x=df_plot['Trimestre'], y=df_plot['Non Europe'], name="Non-Europe", marker_color="deeppink",
    textposition="outside"))
    fig.add_trace(go.Scatter(x=df_plot['Trimestre'], y=df_plot['Group'], name="Groupe", mode="lines+markers", line=dict(color="black", width=3,
)))

    fig.update_layout(
        title=f"{op} - Croissance par trimestre",
        barmode="group",
        yaxis_title="Croissance (%)",
        yaxis=dict(visible=False),
        xaxis_title="Trimestre",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
# --- GRAPHIQUE GLOBAL GROUP PAR OPÉRATEUR ---
st.markdown("## 📈 Croissance Groupe par opérateur")



# Filtrer pour les opérateurs sélectionnés et scope = Group
df_group = df[(df['Scope'] == 'Group') & (df['operator'].isin(selected_operators))]

# Palette de couleurs personnalisée
operator_colors = {
    "O": "orange",
    "D": "deeppink",
    "B": "purple",
    "TEF": "blue",
    "V": "red",
    "T": "green"
}

fig_group = go.Figure()

for op in selected_operators:
    df_op = df_group[df_group['operator'] == op]
    y_values = []
    for quarter in selected_quarters:
        val = df_op[quarter].values
        y_values.append(val[0] if len(val) > 0 else None)

    fig_group.add_trace(go.Scatter(
        x=selected_quarters,
        y=y_values,
        mode='lines+markers+text',
        name=op,
        line=dict(color=operator_colors.get(op, 'gray'), width=2),
        text=[f"{y * 100:+.1f}%" if i == len(selected_quarters)-1 and y is not None else "" for i, y in enumerate(y_values)],
        textposition="top right"
    ))

fig_group.update_layout(
    title="Évolution de la croissance Groupe par opérateur",
    xaxis_title="Trimestre",
    yaxis_title="Croissance (%)",
    yaxis_tickformat=".1f%",  # ⬅️ Affichage en pourcentage
    height=500
)

st.plotly_chart(fig_group, use_container_width=True)



st.markdown("## 🌍 Carte de croissance par pays (dernier trimestre)")

# Filtrer les lignes de croissance avec un pays défini
df_map = df[
    (df["indicator"] == "growth") &
    (df["country"].notna())
].copy()

# ✅ Détection correcte des colonnes trimestre
quarter_cols = [col for col in df.columns if isinstance(col, str) and col[0] in "1234" and "Q" in col]

# 🛡️ Sécurité si rien détecté
if not quarter_cols:
    st.error("Aucune colonne de trimestre détectée.")
    st.stop()

# 🔚 Prendre le dernier trimestre disponible
latest_quarter = quarter_cols[-1]
st.markdown(f"### Trimestre utilisé : **{latest_quarter}**")

# 🖼️ Générer une carte pour chaque opérateur sélectionné
if not selected_operators:
    st.warning("Aucun opérateur sélectionné.")
else:
    for op in selected_operators:
        df_op = df_map[
            (df_map["operator"] == op) &
            (df_map[latest_quarter].notna())
        ].copy()

        if df_op.empty:
            st.info(f"Aucune donnée pour {op} au trimestre {latest_quarter}.")
            continue

        df_op["Growth %"] = df_op[latest_quarter] * 100

        fig = px.choropleth(
            df_op,
            locations="country",
            locationmode="country names",
            color="Growth %",
            color_continuous_scale="RdYlGn",
            range_color=[-20, 20],
            title=f"{op} – Croissance par pays ({latest_quarter})"
        )

        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=False),
            coloraxis_colorbar=dict(title="Croissance (%)"),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)


"""
# ---- Config ----
#st.set_page_config(layout="wide")

# ---- Chargement des données ----
@st.cache
def load_data():
    df_ebitda = pd.read_excel("Base_v4.xlsx", sheet_name="EBITDA")
    return df_ebitda

df_ebitda = load_data()

# ---- Détection des colonnes trimestre ----
quarter_cols = [col for col in df_ebitda.columns if isinstance(col, str) and col.endswith("Q25") or col.endswith("Q24") or col.endswith("Q23")]

# Dernier trimestre
last_quarter = quarter_cols[-1]

# ---- Filtrage des données EBITDAal rate groupe ----
df_plot = df_ebitda[
    (df_ebitda["indicator"] == "EBITDAal rate") &
    (df_ebitda["country"].str.lower() == "group")  # filtre sur country = "Group"
].copy()

# ---- Traitement des valeurs ----
df_plot["EBITDAal Rate (%)"] = pd.to_numeric(df_plot[last_quarter], errors="coerce") * 100
df_plot.dropna(subset=["EBITDAal Rate (%)"], inplace=True)
df_plot["EBITDAal Rate (%)"] = df_plot["EBITDAal Rate (%)"].round(2)
df_plot["label"] = df_plot["EBITDAal Rate (%)"].astype(str) + "%"

# ---- Gestion de la superposition : décalage horizontal (jitter) ----
value_counts = df_plot["EBITDAal Rate (%)"].value_counts()
df_plot["dup_count"] = df_plot["EBITDAal Rate (%)"].map(value_counts)

def jitter_offset(row):
    if row["dup_count"] > 1:
        matches = df_plot[df_plot["EBITDAal Rate (%)"] == row["EBITDAal Rate (%)"]].index
        idx = list(matches).index(row.name)
        return -0.15 + 0.15 * idx
    return 0.0

df_plot["x_offset"] = df_plot.apply(jitter_offset, axis=1)

# ---- Scatter Plot ----
fig = px.scatter(
    df_plot,
    x=df_plot["x_offset"],
    y="EBITDAal Rate (%)",
    color="operator",
    text="label",
    height=600
)

fig.update_traces(
    marker=dict(size=14),
    textposition="top center"
)

fig.update_layout(
    showlegend=True,
    xaxis=dict(
        showticklabels=False,
        title=""
    ),
    yaxis=dict(
        title="EBITDAal Rate (%)",
        tickformat=".2f"
    )
)

st.header(f"Taux d'EBITDAal - {last_quarter}")
st.plotly_chart(fig, use_container_width=True)


"""
# Charger les données
@st.cache
def load_ebitda_data():
    return pd.read_excel("Base_v4.xlsx", sheet_name="EBITDA")

df_ebitda = load_ebitda_data()

# Nettoyer les colonnes
df_ebitda.columns = df_ebitda.columns.str.strip()

# Filtrer sur EBITDAal rate au niveau Groupe
df_rate = df_ebitda[
    (df_ebitda["indicator"].astype(str).str.strip().str.lower() == "ebitdaal rate") &
    (df_ebitda["scope"].astype(str).str.strip().str.lower() == "group")
].copy()

# Détecter les colonnes de trimestre
quarter_cols = [col for col in df_rate.columns if isinstance(col, str) and col[:2] in ["1Q", "2Q", "3Q", "4Q"]]
if not quarter_cols:
    st.error("Aucune colonne de trimestre détectée.")
    st.stop()

# Dernier trimestre
latest_quarter = quarter_cols[-1]
st.subheader(f"Taux d'EBITDAal – Trimestre : **{latest_quarter}**")

# Nettoyage des valeurs
df_rate[latest_quarter] = (
    df_rate[latest_quarter].astype(str)
    .str.replace("%", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.strip()
)
df_rate[latest_quarter] = pd.to_numeric(df_rate[latest_quarter], errors="coerce")

# Filtrer et préparer
df_plot = df_rate[["operator", latest_quarter]].dropna()
df_plot["EBITDAal Rate (%)"] = (df_plot[latest_quarter] * 100).round(2)
df_plot["label"] = df_plot["EBITDAal Rate (%)"].astype(str) + "%"

if df_plot.empty:
    st.warning("Aucune donnée à afficher pour ce trimestre.")
else:
    # Afficher scatter plot vertical
    fig = px.scatter(
        df_plot,
        x=[""] * len(df_plot),
        y="EBITDAal Rate (%)",
        color="operator",
        text="label",
        height=600
    )
    fig.update_traces(marker=dict(size=14), textposition="top center")
    fig.update_layout(
        showlegend=True,
        yaxis_title="",
        xaxis_title="",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    st.plotly_chart(fig, use_container_width=True)
