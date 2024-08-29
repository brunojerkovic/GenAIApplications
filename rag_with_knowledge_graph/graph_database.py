from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from typing import List
import itertools
load_dotenv()


class Neo4jGraphDatabase:
    def __init__(self, embedding_model: str = "text-embedding-ada-002", verbose: bool = True):
        # Init graph
        self.graph = Neo4jGraph()
        self.verbose = verbose

        # For vector-retrieval
        self.vector_index_name = "vector_index_test"
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(model="text-embedding-ada-002"),
            node_label="Article",
            text_node_properties=['text'],
            embedding_node_property='text_embedded',
            index_name="vector_index_test"
        )

    def clear(self):
        # Clear the database
        self.graph.query("MATCH (n) DETACH DELETE n")

    def add_article_node(self, source_type, text_input, link, selected_date, stocks_related, verbose=False):
        # Create a dict of atributes for the new text node
        article_attributes = {
            "source_type": source_type,
            "text": text_input,
            "date": selected_date,
            "link": link
        }

        # Check if a node already exists
        result = self.graph.query(
            "MATCH (article:Article {source_type: $source_type, date: $date, link: $link}) "
            "RETURN article",
            article_attributes
        )
        if result:
            if self.verbose:
                print("This text is already in the dataset. Skipping...")
            return

        # Add a new article
        self.graph.query(
            "CREATE (article:Article {source_type: $source_type, text: $text, date: $date, link: $link})",
            article_attributes
        )

        # Embed the text of a new node
        self.vector_index = Neo4jVector.from_existing_graph(
            self.embedding_model,
            node_label="Article",
            text_node_properties=['text'],
            embedding_node_property='text_embedded',
            index_name=self.vector_index_name
        )

        # Create related stock relationships
        if stocks_related:
            # All stock permutations
            stock_pairs = list(itertools.combinations(stocks_related, 2))

            # Build a query for adding relationships
            match_statement = "MATCH " + " ".join([f"({s}:Stock {{ticker_name: '{s}'}})," for s in stocks_related])[:-1]
            match_statement += ", (article:Article {source_type: $source_type, text: $text, date: $date, link: $link}) "
            merge_stock_stock = ' '.join([f"MERGE ({p1})-[:RELATED]-({p2})" for (p1, p2) in stock_pairs])
            merge_stock_article = ' '.join([f"MERGE (article)-[:IMPACTS]-({s})" for s in stocks_related])

            # Create stock relationships
            self.graph.query(
                match_statement + merge_stock_stock + merge_stock_article,
                article_attributes
            )
        if self.verbose:
            print("Successfully added a new text article!")

    def get_stocks_impacted_by_article(self, contexts) -> List[List[str]]:
        related_nodes_lists = [
            [node["stock_node"] for node in self.graph.query(
                "MATCH (article:Article {link:$link, source_type:$source_type, date:$date})"
                "MATCH (article)-[:IMPACTS]->(stock_node)"
                "RETURN stock_node",
                {
                    "link": context.metadata["link"],
                    "source_type": context.metadata["source_type"],
                    "date": context.metadata["date"]}
            )]
            for context in contexts
        ]
        related_stocks_lists = [[node["ticker_name"] for node in node_list] for node_list in related_nodes_lists]

        return related_stocks_lists
