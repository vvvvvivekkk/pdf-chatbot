import time

from rag_pipeline import get_embeddings
from langchain_community.vectorstores import Chroma

# =====================================
# CONFIG
# =====================================

CHROMA_PATH = "chroma_db/multi_pdf_store"
TOP_K = 8

# =====================================
# TEST QUESTIONS
# =====================================

TEST_CASES = [
    {
        "question": "What is a Knowledge Generation Engine?",
        "gold_text": "Knowledge Generation Engine"
    },
    {
        "question": "What are the main components of a knowledge engine?",
        "gold_text": "knowledge"
    },
    {
        "question": "How can Claude be used effectively?",
        "gold_text": "Claude"
    },
    {
        "question": "What skills are important when working with Claude?",
        "gold_text": "skill"
    }
]

# =====================================
# METRICS
# =====================================

def recall_at_k(retrieved_docs, gold_text):
    for doc in retrieved_docs:
        if gold_text.lower() in doc.page_content.lower():
            return 1
    return 0


def precision_at_k(retrieved_docs, gold_text):
    hits = 0

    for doc in retrieved_docs:
        if gold_text.lower() in doc.page_content.lower():
            hits += 1

    if len(retrieved_docs) == 0:
        return 0

    return hits / len(retrieved_docs)


def mrr(retrieved_docs, gold_text):
    for rank, doc in enumerate(retrieved_docs, start=1):
        if gold_text.lower() in doc.page_content.lower():
            return 1 / rank

    return 0


# =====================================
# MAIN EVALUATION
# =====================================

def run_evaluation():

    print("\nLoading embeddings...")
    embeddings = get_embeddings()

    print(f"\nLoading vector store from: {CHROMA_PATH}")

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    count = vectorstore._collection.count()

    print(f"Vector Count: {count}")

    if count == 0:
        raise ValueError(
            f"No vectors found in {CHROMA_PATH}"
        )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    total_recall = 0
    total_precision = 0
    total_mrr = 0

    retrieval_times = []

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 60)

    for idx, test in enumerate(TEST_CASES, start=1):

        question = test["question"]
        gold_text = test["gold_text"]

        start = time.time()

        docs = retriever.invoke(question)

        elapsed = time.time() - start

        retrieval_times.append(elapsed)

        recall = recall_at_k(docs, gold_text)
        precision = precision_at_k(docs, gold_text)
        mrr_score = mrr(docs, gold_text)

        total_recall += recall
        total_precision += precision
        total_mrr += mrr_score

        print("\n" + "-" * 60)
        print(f"Question {idx}")
        print(f"Q: {question}")
        print(f"Gold Text: {gold_text}")

        print(f"Recall@{TOP_K}: {recall:.2f}")
        print(f"Precision@{TOP_K}: {precision:.2f}")
        print(f"MRR: {mrr_score:.2f}")
        print(f"Latency: {elapsed:.3f}s")

        if docs:
            print("\nTop Retrieved Chunk:\n")
            print(docs[0].page_content[:400])

            print("\nMetadata:")
            print(docs[0].metadata)

    n = len(TEST_CASES)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Average Recall@{TOP_K}: {total_recall / n:.3f}")
    print(f"Average Precision@{TOP_K}: {total_precision / n:.3f}")
    print(f"Average MRR: {total_mrr / n:.3f}")

    avg_latency = sum(retrieval_times) / len(retrieval_times)

    print(f"Average Retrieval Latency: {avg_latency:.3f}s")


if __name__ == "__main__":
    run_evaluation()