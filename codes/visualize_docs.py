fig_nuvem = topic_model.visualize_documents(
    docs,
    embeddings=embeddings,
    sample=0.2, 
    hide_document_hover=False, 
    hide_annotations=False
)

fig_nuvem.show()
