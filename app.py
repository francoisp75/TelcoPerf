"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

@st.cache
def load_data():
    return pd.read_excel("Base_v4.xlsx", sheet_name="CA")

df_raw = load_data()
df = df_raw[df_raw['indicator'].str.lower() == 'growth']

# Liste des trimestres
quarter_cols = [col for col in df.columns if "Q" in str(col) and str(col)[:2] in ["1Q", "2Q", "3Q", "4Q"]]

st.title("Benchmark - Croissance CA par trimestre (valeurs brutes)")
selected_operators = st.multiselect("Choisir un ou plusieurs opérateurs", df['operator'].dropna().unique())

if not selected_operators:
    st.warning("Veuillez sélectionner au moins un opérateur.")
    st.stop()

for op in selected_operators:
    df_op = df[df['operator'] == op]

    data = {
        "Trimestre": [],
        "Europe": [],
        "Non Europe": [],
        "Group": []
    }

    for quarter in quarter_cols:
        data["Trimestre"].append(quarter)

        # Prendre la première valeur trouvée ou None
        europe_val = df_op[df_op['Scope'] == 'Europe'][quarter].values
        noneurope_val = df_op[df_op['Scope'] == 'Non Europe'][quarter].values
        group_val = df_op[df_op['Scope'] == 'Group'][quarter].values

        data["Europe"].append(europe_val[0] if len(europe_val) > 0 else None)
        data["Non Europe"].append(noneurope_val[0] if len(noneurope_val) > 0 else None)
        data["Group"].append(group_val[0] if len(group_val) > 0 else None)

    df_plot = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_plot['Trimestre'], y=df_plot['Europe'], name="Europe", marker_color="blue"))
    fig.add_trace(go.Bar(x=df_plot['Trimestre'], y=df_plot['Non Europe'], name="Non-Europe", marker_color="deeppink"))
    fig.add_trace(go.Scatter(x=df_plot['Trimestre'], y=df_plot['Group'], name="Groupe", mode="lines+markers", line=dict(color="black", width=3)))

    fig.update_layout(
        title=f"{op} - Croissance trimestrielle",
        barmode="group",
        yaxis_title="Croissance (%)",
        xaxis_title="Trimestre",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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
