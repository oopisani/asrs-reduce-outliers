import plotly.graph_objects as go

labels = [
    "Ruído (HDBSCAN)\n1783",
    "Subset TF-IDF\n183",
    "Tópico 6 (recuperado)\n33",
    "Outros tópicos\n150",
    "Ruído remanescente\n0"
]

source = [0, 1, 1, 1]
target = [1, 2, 3, 4]
values = [183, 33, 150, 0]

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels
    ),
    link=dict(
        source=source,
        target=target,
        value=values
    )
)])

fig.update_layout(
    title_text="Redistribuição de Outliers via reduce_outliers()",
    font_size=12
)

fig.show()
