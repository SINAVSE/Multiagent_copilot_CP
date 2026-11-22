
from unittest import result

from src.rag.generation.answer_generator import AnswerGenerator


def main():
    generator = AnswerGenerator(top_k=5)

    query = (
        "What are the meal and incidental expense (M&IE) "
        "per diem rules for sponsored business travel?"
    )

    print("\nUSER QUESTION:")
    print(query)

    print("\nGenerating answer...")

    result = generator.generate(query)

    print("\n" + "=" * 60)
    print("RAG ANSWER")
    print("=" * 60)

    print(result.answer)

    print("\nINSUFFICIENT EVIDENCE:")
    print(result.insufficient_evidence)

    print("\nMISSING INFORMATION:")
    for item in result.missing_information:
        print("-", item)

    print("\nSOURCE CITATIONS:")
    for citation in result.citations:
        print(f"\nChunk ID: {citation.chunk_id}")
        print(f"Document ID: {citation.document_id}")
        print(f"Title: {citation.title}")
        print(f"Section: {citation.section_path}")
        print(f"Source: {citation.source_url}")

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    print("\nRETRIEVAL METHOD:")
    print(result.retrieval_method)


if __name__ == "__main__":
    main()