from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from custom_chain import CustomSequentialChain
from stock_data import StockData
from datetime import datetime, timedelta
from utils import stock_tickers
from graph_database import Neo4jGraphDatabase
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub


class OpenAILLM:
    def __init__(self, temperature: float = 1.,
                 model_name: str = 'gpt-3.5-turbo-0125',
                 k: int = 5):
        # Model-related instantiations
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)

        # Create chains
        self.chain_data_input = CustomSequentialChain()
        self.__create_data_input_chain()

        # Init the graph database
        self.graph_database = Neo4jGraphDatabase(verbose=True)

        # Init retriever
        combine_docs_chain = create_stuff_documents_chain(
            ChatOpenAI(),
            hub.pull("langchain-ai/retrieval-qa-chat")
        )
        self.retrieval = create_retrieval_chain(
            retriever=self.graph_database.vector_index.as_retriever(),
            combine_docs_chain=combine_docs_chain
        )

    def __create_data_input_chain(self):
        # CHAIN 1: Get the nodes
        response_schema = [
            ResponseSchema(name="stock_list_filtered", description="A list of stocks mentioned in this article.")
        ]
        output_format = StructuredOutputParser.from_response_schemas(response_schema).get_format_instructions()
        prompt_template = PromptTemplate(
            template="""
                Analyze this text, and return the list of stocks that are mentioned in this article.
                Here is the text: '''{text_input}'''.
                Only select the stocks from this list: '''{stock_list}'''.
                Do not change the stock names, only filter them to return the ones mentioned in the given text.                
                
                {output_format}
                Make sure that all quotes for output json are double quotes.
            """,
            input_variables=["text_input"],
            partial_variables={"output_format": output_format, "stock_list": stock_tickers}
        )
        self.chain_data_input.add_chain(llm=self.llm,
                                        prompt=prompt_template,
                                        chain_key="data_inputter",
                                        response_schema=response_schema)

        # CHAIN 2: Create text summary
        response_schema = [
            ResponseSchema(name="text_summarized", description="Summarized input text.")
        ]
        output_format = StructuredOutputParser.from_response_schemas(response_schema).get_format_instructions()
        prompt_template = PromptTemplate(
            template="""
                    Summarize this text in 500 words, and make sure that the summarized information does not miss on info relevant for the stocks.
                    Here is the stock list: '''{stock_list_filtered}'''.
                    Here is the text to summarize: '''{text_input}'''.

                    {output_format}
                    Make sure that all quotes for output json are double quotes.
                    """,
            input_variables=["text_input", "stock_list_filtered"],
            partial_variables={"output_format": output_format}
        )
        self.chain_data_input.add_chain(llm=self.llm,
                                        prompt=prompt_template,
                                        chain_key="text_summarizer",
                                        response_schema=response_schema)

    def get_news(self, user_query):
        # Get answer and associated documents
        response = self.retrieval.invoke({"input": user_query})
        answer = response["answer"]
        contexts = response["context"]

        # Extract information from RAG output context
        related_articles = [context.page_content for context in contexts]
        dates = [str(context.metadata["date"]) for context in contexts]
        links = [context.metadata["link"] for context in contexts]
        source_types = [context.metadata["source_type"] for context in contexts]

        # Get related stocks
        related_stocks = self.graph_database.get_stocks_impacted_by_article(contexts)

        # Get the price tables
        price_tables = []
        for date, stock_list in zip(dates, related_stocks):
            date_before = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
            date_after = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")
            price_tables.append(StockData(
                ticker_symbols=stock_list,
                start_date=date_before,
                end_date=date_after
            ).datasets)

        return answer, related_articles, dates, links, source_types, price_tables, related_stocks

    def save_source(self, source_selected, text_input, link, selected_date):
        # Get the list of stocks that are impacted by this article
        try:
            response = self.chain_data_input.run(text_input=text_input)
            stock_list = [stock for stock in response["stock_list_filtered"] if stock in stock_tickers]
            text_summarized = response["text_summarized"]
        except Exception as e:
            stock_list = []
            text_summarized = ""
            raise f"Exception: {e}"

        # Add this article to Neo4j database
        self.graph_database.add_article_node(
            source_type=source_selected,
            text_input=text_summarized,
            link=link,
            selected_date=selected_date,
            stocks_related=stock_list
        )
