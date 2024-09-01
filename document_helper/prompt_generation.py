from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from custom_chain import CustomSequentialChain
from langchain_pinecone import PineconeVectorStore
import json


class OpenAILLM:
    def __init__(self, temperature: float = 1.,
                 model_name: str = 'gpt-3.5-turbo-0125',
                 k: int = 5):
        # Model-related instantiations
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)

        # CV-related init
        self.custom_chain = CustomSequentialChain()
        self.create_chain()

        # Retrieval-related info
        self.index_name = "resume-write-info"
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.index_pc = PineconeVectorStore.from_existing_index(embedding=self.embedding_model, index_name=self.index_name)
        self.k = k

    def get_similar_docs(self, query, k=2):
        similar_docs = self.index_pc.similarity_search(query, k=k)
        return similar_docs

    def get_response(self, user_document: str, job_vacancy: str, document_type: str) -> str:
        try:
            # Retrieve context documents
            data_context = ""
            if document_type == "CV":
                query_index = f"Tips on how to improve this CV: ''{user_document}''"
                context_docs = self.get_similar_docs(query_index, k=self.k)
                if context_docs:
                    data_context = "\n".join([doc.page_content for doc in context_docs])

            # Get the response
            response = self.custom_chain.run(
                user_document=user_document,
                job_vacancy=job_vacancy,
                document_type=document_type,
                verbose=True,
                include_all_outputs=True,
                data_context=data_context
            )
            match_score, tips, cv_rewritten, new_match_score = response['match_score'], response['tips'], response['document_rewritten'], response['match_score_new']
        except Exception as e:
            match_score, tips, cv_rewritten, new_match_score = e, "", "", ""
        return match_score, tips, cv_rewritten, new_match_score

    def create_chain(self):
        # CHAIN 1: Get the match score
        response_schema_scorer = [
            ResponseSchema(name="match_score", description="Match score between the uploaded CV/cover letter and the job vacancy.")
        ]
        output_format_scorer = StructuredOutputParser.from_response_schemas(response_schema_scorer).get_format_instructions()
        prompt_template_scorer = PromptTemplate(
            template="""
                Analyze my CV/cover letter and a job vacancy in order to provide a match score between them.
                I want to know how good of a fit I am.
                Match score should be between 1 (not a good match at all) and 100 (perfect match).
                
                Here is my {document_type}: '{user_document}'.
                Here is the job vacancy: '{job_vacancy}'.
                
                {output_format}
                Make sure that all quotes for output json are double quotes.
            """,
            input_variables=["user_document", "job_vacancy", "document_type"],
            partial_variables={"output_format": output_format_scorer}
        )
        self.custom_chain.add_chain(llm=self.llm,
                                    prompt=prompt_template_scorer,
                                    chain_key="match_scorer",
                                    response_schema=response_schema_scorer)

        # CHAIN 2: Tip provider
        response_schema_tips = [
            ResponseSchema(name="tips", description="Tips on how to improve the uploaded CV to be a better match for the job vacancy.")
        ]
        output_format_tips = StructuredOutputParser.from_response_schemas(response_schema_tips).get_format_instructions()
        prompt_template_tips = PromptTemplate(
            template="""
                I want to improve my CV/cover letter for a job vacancy.
                Match score is a rating of how well my CV/cover letter fits the job vacancy, and it is currently {match_score}. 
                It is a value between 1 (not a good match at all) and 100 (perfect match).
                Provide me some tips on how can I rewrite my CV/cover letter so that the match score increases!
                
                Context: '''{data_context}'''.
                
                Here is my CV/cover letter: '{user_document}'.
                Here is the job vacancy: '{job_vacancy}'.
    
                {output_format}
                Make sure that all quotes for output json are double quotes.
            """,
            input_variables=["user_document", "job_vacancy", "match_score", "data_context"],
            partial_variables={"output_format": output_format_tips}
        )
        self.custom_chain.add_chain(llm=self.llm,
                                    prompt=prompt_template_tips,
                                    chain_key="tip_provider",
                                    response_schema=response_schema_tips)

        # CHAIN 2: CV/Cover Letter rewriter
        response_schema_tips = [
            ResponseSchema(name="document_rewritten", description="Rewritten CV/cover letter with incorporated tips that improve the original match score.")
        ]
        output_format_tips = StructuredOutputParser.from_response_schemas(
            response_schema_tips).get_format_instructions()
        prompt_template_tips = PromptTemplate(
            template="""
                I want to improve my CV/cover letter for a job vacancy.
                Match score is a rating of how well my CV/cover letter fits the job vacancy, and it is currently {match_score}. 
                It is a value between 1 (not a good match at all) and 100 (perfect match).
                Here are some tips on how I can improve my CV/cover letter for this job: '{tips}'!
                Rewrite my CV/cover letter so that it is a better match to the provided job vacancy!
    
                Here is my CV/cover letter: '{user_document}'.
                Here is the job vacancy: '{job_vacancy}'.
    
                {output_format}
                Make sure that all quotes for output json are double quotes.
            """,
            input_variables=["user_document", "job_vacancy", "match_score", "tips"],
            partial_variables={"output_format": output_format_tips}
        )
        self.custom_chain.add_chain(llm=self.llm,
                                    prompt=prompt_template_tips,
                                    chain_key="document_writer",
                                    response_schema=response_schema_tips)

        # CHAIN 4: Get the match score for the rewritten CV
        response_schema_scorer_new = [
            ResponseSchema(name="match_score_new", description="Match score between the uploaded CV/cover letter and the job vacancy.")
        ]
        output_format_scorer_new = StructuredOutputParser.from_response_schemas(response_schema_scorer_new).get_format_instructions()
        prompt_template_scorer_new = PromptTemplate(
            template="""
                Analyze my CV/cover letter and a job vacancy in order to provide a match score between them.
                I want to know how good of a fit I am.
                Match score should be between 1 (not a good match at all) and 100 (perfect match).

                Here is my {document_type}: '{document_rewritten}'.
                Here is the job vacancy: '{job_vacancy}'.

                {output_format}
                Make sure that all quotes for output json are double quotes.
            """,
            input_variables=["document_rewritten", "job_vacancy", "document_type"],
            partial_variables={"output_format": output_format_scorer_new}
        )
        self.custom_chain.add_chain(llm=self.llm,
                                    prompt=prompt_template_scorer_new,
                                    chain_key="match_scorer",
                                    response_schema=response_schema_scorer_new)
