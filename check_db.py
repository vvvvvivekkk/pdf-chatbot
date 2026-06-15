from rag_pipeline import get_embeddings
from langchain_community.vectorstores import Chroma

emb = get_embeddings()

for collection in [
    "chroma_db",
    "chroma_db/multi_pdf_store",
    "chroma_db/sample_pdf",
    "chroma_db/knowledge_generation_engine__llm_wiki__pdf"
]:
    try:
        vs = Chroma(
            persist_directory=collection,
            embedding_function=emb
        )

        print(collection)
        print("count =", vs._collection.count())
        print("-" * 40)

    except Exception as e:
        print(collection, e)