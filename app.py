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
