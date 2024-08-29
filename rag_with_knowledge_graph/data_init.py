import os
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
import itertools
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from utils import stock_tickers
load_dotenv()


def create_graph_database():
    # Init graph
    graph = Neo4jGraph()

    # Clear the database
    graph.query("MATCH (n) DETACH DELETE n")

    # Add nodes for stocks
    for stock in stock_tickers:
        graph.query(
            "CREATE (:Stock {ticker_name: $ticker_name})",
            {"ticker_name": stock}
        )

    # Add a new article
    text_example = {
        "source_type": "Tweet",
        "text": "Google, Nvidia and Meta are one of the Big tech companies. This is a test text.",
        "date": "2020-01-01",
        "link": "https://google.com"
    }
    graph.query(
        "CREATE (article:Article {source_type: $source_type, text: $text, date: $date, link: $link})",
        {"source_type": text_example['source_type'], "text": text_example['text'], "date": text_example['date'], "link": text_example['link']},
    )

    # Get the vector index
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        node_label="Article",
        text_node_properties=['text'],
        embedding_node_property='text_embedded',
        index_name="vector_index_test"
    )

    stocks_impacted = ["GOOG", "META", "NVDA"]

    stock_pairs = list(itertools.combinations(stocks_impacted, 2))
    match_statement = "MATCH " + " ".join([f"({s}:Stock {{ticker_name: '{s}'}})," for s in stocks_impacted])[:-1]
    match_statement += ", (article:Article {source_type: $source_type, text: $text, date: $date, link: $link}) "
    merge_stock_stock = ' '.join([f"MERGE ({p1})-[:RELATED]-({p2})" for (p1, p2) in stock_pairs])
    merge_stock_article = ' '.join([f"MERGE (article)-[:IMPACTS]-({s})" for s in stocks_impacted])

    graph.query(
        match_statement+merge_stock_stock+merge_stock_article,
        {"source_type": text_example['source_type'], "text": text_example['text'], "date": text_example['date'],
         "link": text_example['link']}
    )


def test_relevant_node_selection(user_query):
    # Init graph
    graph = Neo4jGraph()

    # Get the vector index
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        node_label="Article",
        text_node_properties=['text'],
        embedding_node_property='text_embedded',
        index_name="vector_index_test"
    )

    # Create a chain
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        ChatOpenAI(), retrieval_qa_chat_prompt
    )
    vector_qa = create_retrieval_chain(retriever=vector_index.as_retriever(), combine_docs_chain=combine_docs_chain)

    # Perform RAG
    response = vector_qa.invoke({"input": user_query})

    # Parse RAG output
    answer = response["answer"]
    contexts = response["context"]
    texts = [context.page_content for context in contexts]
    dates = [context.metadata["date"] for context in contexts]
    links = [context.metadata["link"] for context in contexts]
    source_types = [context.metadata["source_type"] for context in contexts]

    # Get the related nodes
    related_nodes_lists = [
        [node["stock_node"] for node in graph.query(
            "MATCH (article:Article {link:$link, source_type:$source_type, date:$date})"
            "MATCH (article)-[:IMPACTS]->(stock_node)"
            "RETURN stock_node",
            {
                "link": context.metadata["link"],
                "source_type": context.metadata["source_type"],
                "date": context.metadata["date"]}
        )]
        for context in response["context"]
    ]
    related_stocks_lists = [[node["ticker_name"] for node in node_list] for node_list in related_nodes_lists]

    # Print the output
    print("ANSWER", answer)
    print("TEXTS", [t[:10] for t in texts])
    print("DATES", dates)
    print("LINKS", links)
    print("SOURCES", source_types)
    print("RELATED NODES", related_stocks_lists)

if __name__ == "__main__":
    create_graph_database()
    # user_query = "How is the upcoming season for the US stock market?"
    # test_relevant_node_selection(user_query)
