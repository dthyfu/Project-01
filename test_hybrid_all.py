
from hybrid_search import HybridRAG


def main():
    rag = HybridRAG()

    try:
        while True:
            question = input(
                "\nNhap cau hoi (go 'exit' de thoat): "
            ).strip()

            if question.lower() == "exit":
                break

            result = rag.ask(question)

            print("\n========== CAU TRA LOI ==========")
            print(result["answer"])
            print("\nSo ket qua truy xuat:",
                  result["retrieved_count"])

            if result["sources"]:
                print("\nNguon:")
                for source in result["sources"]:
                    print("-", source["title"])
                    if source["url"]:
                        print(" ", source["url"])

    finally:
        rag.close()


if __name__ == "__main__":
    main()
