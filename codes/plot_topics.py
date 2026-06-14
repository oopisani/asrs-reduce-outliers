fig_barras = topic_model.visualize_barchart(
    top_n_topics=12,
    n_words=8,
    title="Principais Palavras por Tópico"
)

fig_barras.update_layout(
    width=1200, 
    height=800, 
    showlegend=False
)
fig_barras.update_xaxes(tickangle=0)
fig_barras.update_yaxes(tickangle=0)

fig_barras.show()
