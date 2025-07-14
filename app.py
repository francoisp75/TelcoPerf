
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import numpy as np

@st.cache_data
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
    fig.add_trace(go.Bar(x=df_plot['Trimestre'], y=df_plot['Europe'], name="Europe", marker_color="blue",text=[f"{val * 100:.1f}%" for val in df_plot['Europe']],
    textposition="outside"))
    fig.add_trace(go.Bar(x=df_plot['Trimestre'], y=df_plot['Non Europe'], name="Non-Europe", marker_color="deeppink",text=[f"{val * 100:.1f}%" for val in df_plot['Non Europe']],
    textposition="outside"))
    fig.add_trace(go.Scatter(x=df_plot['Trimestre'], y=df_plot['Group'], name="Groupe", mode="lines+markers+text", line=dict(color="black", width=3),text=(df_plot['Group'] * 100).round(1).astype(str) + '%',
    textposition="top center"

))

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
    "TE": "blue",
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
        #text=[f"{y * 100:+.1f}%" if i == len(selected_quarters)-1 and y is not None else "" for i, y in enumerate(y_values)],
        text=["n/a" if y is None else f"{y * 100:+.1f}%" for y in y_values],
        textposition="top right"
    ))

fig_group.update_layout(
    title="Évolution de la croissance Groupe par opérateur",
    xaxis_title="Trimestre",
    yaxis_title="Croissance (%)",
    yaxis_tickformat=".0%",
    #yaxis_tickformat=".1f%",  # ⬅️ Affichage en pourcentage
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



# Charger les données
@st.cache_data
def load_ebitda_data():
    return pd.read_excel("Base_v4.xlsx", sheet_name="EBITDA")

df_ebitda = load_ebitda_data()
df_ebitda.columns = df_ebitda.columns.str.strip()

# Filtrer les données pertinentes
df_rate = df_ebitda[
    (df_ebitda["indicator"].astype(str).str.strip().str.lower() == "ebitdaal rate") &
    (df_ebitda["scope"].astype(str).str.strip().str.lower() == "group")
].copy()

# Détecter les colonnes de trimestre
quarter_cols = [col for col in df_rate.columns if isinstance(col, str) and re.match(r"\dQ\d\d", col)]

if not quarter_cols:
    #st.warning("Aucune colonne de trimestre détectée.")
    st.stop()

# Identifier le dernier trimestre et celui N-1
quarter_cols_sorted = sorted(quarter_cols, key=lambda x: (int(x[-2:]), int(x[0])))
latest_quarter = quarter_cols_sorted[-1]
prev_quarter = f"{latest_quarter[0]}Q{int(latest_quarter[-2:]) - 1:02d}"  # ex: 1Q25 -> 1Q24

st.subheader(f"Taux d'EBITDAal – **{latest_quarter}**")

# Nettoyage des valeurs
for col in [latest_quarter, prev_quarter]:
    if col in df_rate.columns:
        df_rate[col] = (
            df_rate[col].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df_rate[col] = pd.to_numeric(df_rate[col], errors="coerce")
    else:
        df_rate[col] = None

# Menu pour sélectionner les opérateurs
all_operators = sorted(df_rate["operator"].dropna().unique())
selected_operators = st.multiselect(
    "Sélectionner les opérateurs à afficher :",
    options=all_operators,
    default=all_operators  # tu peux choisir d’en mettre moins si besoin
)

# Filtrer selon sélection
df_rate = df_rate[df_rate["operator"].isin(selected_operators)]


# Préparer données
df_plot = df_rate[["operator", latest_quarter, prev_quarter]].dropna(subset=[latest_quarter])
df_plot["current"] = df_plot[latest_quarter]
df_plot["previous"] = df_plot[prev_quarter]
df_plot["variation"] = df_plot["current"] - df_plot["previous"]
df_plot["rate_percent"] = (df_plot["current"] * 100).round(2)

# Couleurs personnalisées
operator_colors = {
    "O": "orange", "D": "deeppink", "B": "purple",
    "TE": "blue", "V": "red", "T": "green"
}
df_plot["color"] = df_plot["operator"].map(operator_colors).fillna("gray")

# Ajouter labels avec variation
df_plot["label"] = df_plot.apply(
    lambda row: f"{row['operator']}<br>{row['rate_percent']:.2f}%"
    + (f"<br>({row['variation']*100:+.2f} pts)" if pd.notnull(row['variation']) else ""),
    axis=1
)

# Ajouter léger décalage horizontal (jitter) pour éviter superposition
jitter_map = df_plot.groupby("rate_percent").cumcount()
jitter = 0.04
df_plot["x"] = jitter_map * jitter - jitter * jitter_map.max() / 2

# Afficher scatter
fig = px.scatter(
    df_plot,
    x="x",
    y="rate_percent",
    color="operator",
    color_discrete_map=operator_colors,
    text="label",
    height=650
)
fig.update_traces(marker=dict(size=14), textposition="top center")
fig.update_layout(
    showlegend=True,
    yaxis_title="Taux EBITDAal (%)",
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=None),
    yaxis=dict(showticklabels=True, gridcolor='lightgray'),
    plot_bgcolor='white'
)

st.plotly_chart(fig, use_container_width=True)
