from typing import Any, Dict, List, cast
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import BaseMessage

from src.services.milvus_handler import MilvusHandler
from src.services.embedding_handler import EmbeddingHandler
from src.shared.utils import load_chat_model, format_docs

from src.agent.configuration import Configuration
from src.agent.rag_self_reflection.state import ResearcherState, Grader, RewriterResponse

def retrieve_documents(
        state: ResearcherState, *, config: RunnableConfig
    ) -> dict[str, list[Document]]:
    
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    configuration = Configuration.from_runnable_config(config)
    question = state.question

    print(question)
    print(configuration)
    # We'll embed the question
    embedding_handler = EmbeddingHandler(model_name=configuration.embedding_model)
    query_vector = embedding_handler.generate_embeddings([question])
    # Then call your MilvusHandler:
    milvus_handler = MilvusHandler(
        host=configuration.milvus_host,
        port=configuration.milvus_port
    )
    milvus_handler.connect()
    # # Retrieval
    # documents = retriever.invoke(question)
    # If you want the summary text, specify output_fields=["summary"]
    results = milvus_handler.search(
        collection_name=config[configuration.milvus_collection],
        query_vectors=[query_vector],
        top_k=3,
        output_fields=[configuration.vector_output_fields]  # retrieve doc text
    )

    docs: List[Document] = []
    for hits in results:
        for hit in hits:
            # The doc text is stored in 'hit.entity["summary"]'
            doc_text = hit.entity.get("summary", "")
            d = Document(page_content=doc_text, metadata={"id": hit.id, "distance": hit.distance})
            docs.append(d)
    print(docs)
    return {"documents": docs}


async def grade_documents(state: ResearcherState, *, config: RunnableConfig) -> dict[str, Any]:
    configuration = Configuration.from_runnable_config(config)
    question = state.question
    documents = state.documents
    filtered_docs = []
    for d in documents:
        # need to grade each retruned document
        model = load_chat_model(configuration.query_model)

        messages = [
            {"role": "system", "content": configuration.grader_system_prompt},
            {"role": "human", "content": f"Question: {question}\n\nDocument: {d.page_content}"}
        ]

        print("Messages")
        print(messages)
        grade = cast(
            Grader, await model.with_structured_output(Grader).ainvoke(messages)
        )
        print(grade)
        if grade.type == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def decide_to_generate(state: ResearcherState) -> str:
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

async def generate(state: ResearcherState, *, config: RunnableConfig) -> dict[str, Any]:
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)
    context = format_docs(state.documents)
    prompt = configuration.response_system_prompt.format(context=context)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ] 
    generation = await model.ainvoke(messages)

    return {"documents": documents, "question": question, "generation": generation}

async def transform_query(state: ResearcherState, *, config: RunnableConfig) -> dict[str, Any]:
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    configuration = Configuration.from_runnable_config(config)
    # Load the LLM model
    model = load_chat_model(configuration.query_model)

       # Define the system prompt for rewriting the question
    system_prompt = """You are a question rewriter. Your job is to take an input question and improve it for use in a vector database retrieval system.
    Use the following guidelines:
    1. Focus on the underlying intent of the question.
    2. Make the question concise but specific.
    3. Ensure the rephrased question is semantically meaningful and optimized for retrieval."""


    human_prompt = f"""Original Question: {question}

    Improve this question for optimal vector database retrieval."""

    # Construct the messages for the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]

    response = cast(RewriterResponse, await model.ainvoke(messages))

    print(f"Rewritten Question: {response.rewritten_question}")
    print(f"Reasoning: {response.reasoning}")

    # Update the question in the state
    state.question = response.rewritten_question

    # Optionally log the reasoning for debugging
    state.generation.append(f"Reasoning: {response.reasoning}")

    return {"documents": documents, "question": state.question}  

async def grade_generation_v_documents_and_question(state: ResearcherState,  *, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    configuration = Configuration.from_runnable_config(config)
    model = load_chat_model(configuration.response_model)

    # Check hallucination
    system_prompt = """You are a grader assessing whether a generated response is grounded in / supported by a set of retrieved documents.
    Use the following criteria:
    1. If the response aligns with the content of the documents, grade it as 'yes'.
    2. If the response introduces information not found in the documents or conflicts with them, grade it as 'no'.
    Provide an explanation for your grade.

    Respond in the following format:
    - grounded: "yes" or "no"
    - reasoning: Explanation of why the response is or is not grounded."""

    human_prompt = f"""Set of documents: 
    {''.join([doc.page_content for doc in documents])}

    Question: {question}

    Generated Response: {generation}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]
    # Use the model to grade the generation
    grade = cast(Grader, await model.with_structured_output(Grader).ainvoke(messages))

    print(f"Hallucination Grading Result: {grade}")

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # score = answer_grader.invoke({"question": question, "generation": generation})
        # grade = score.binary_score
        grade = "yes"
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
builder = StateGraph(ResearcherState)
builder.add_node(retrieve_documents)
builder.add_node(grade_documents)
builder.add_node("generate", generate)  # generatae
builder.add_node("transform_query", transform_query)  # transform_query

builder.add_edge(START, "retrieve_documents")
builder.add_edge("retrieve_documents", "grade_documents") 
builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
builder.add_edge("transform_query", "retrieve_documents")
builder.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)
graph = builder.compile()
graph.name = "RagSelfReflection"
