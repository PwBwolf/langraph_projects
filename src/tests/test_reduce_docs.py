# test_reduce_docs.py

from langchain_core.documents import Document
from src.shared.state import reduce_docs
# Testing the function
if __name__ == "__main__":
    # Test 1: Start with no existing docs, add a single string
    print("Test 1:")
    docs = reduce_docs(None, "Hello World")
    for d in docs:
        print(d.page_content, d.metadata)

    # Test 2: Add more strings on top of the existing doc
    print("\nTest 2:")
    docs = reduce_docs(docs, ["Another doc", "Yet another doc"])
    for d in docs:
        print(d.page_content, d.metadata)

    # Test 3: Add a dict-based doc
    print("\nTest 3:")
    new_docs = [{"page_content": "Doc from dict", "metadata": {"author": "Alice"}}]
    docs = reduce_docs(docs, new_docs)
    for d in docs:
        print(d.page_content, d.metadata)

    # Test 4: Add a Document object and see if uuids are handled
    print("\nTest 4:")
    doc_obj = Document(page_content="A doc object with no uuid", metadata={})
    docs = reduce_docs(docs, [doc_obj])
    for d in docs:
        print(d.page_content, d.metadata)

    # Test 5: Delete all docs
    print("\nTest 5:")
    docs = reduce_docs(docs, "delete")
    print("Remaining docs after delete:", docs)