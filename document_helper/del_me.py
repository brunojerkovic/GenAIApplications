from custom_chain import CustomSequentialChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from custom_chain import CustomSequentialChain
from utils import read_pdf
from dotenv import load_dotenv
import json
load_dotenv()


job_vacancy = open('job_description_del_me.txt', 'r', encoding='utf-8').read()
user_document = read_pdf("my_docs/bruno_jerkovic_cv.pdf")
document_type = "CV"

llm = ChatOpenAI(temperature=1., model_name="gpt-3.5-turbo-0125")

custom_chain = CustomSequentialChain()

# CHAIN 1: Get the match score
response_schema_scorer = [
    ResponseSchema(name="match_score",
                   description="Match score between the uploaded CV/cover letter and the job vacancy.")
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
    """,
    input_variables=["user_document", "job_vacancy", "document_type"],
    partial_variables={"output_format": output_format_scorer}
)
custom_chain.add_chain(llm=llm,
                            prompt=prompt_template_scorer,
                            chain_key="match_scorer",
                            response_schema=response_schema_scorer)

# CHAIN 2: Tip provider
response_schema_tips = [
    ResponseSchema(name="tips",
                   description="Tips on how to improve the uploaded CV to be a better match for the job vacancy.")
]
output_format_tips = StructuredOutputParser.from_response_schemas(response_schema_tips).get_format_instructions()
prompt_template_tips = PromptTemplate(
    template="""
        I want to improve my CV/cover letter for a job vacancy.
        Match score is a rating of how well my CV/cover letter fits the job vacancy, and it is currently {match_score}. 
        It is a value between 1 (not a good match at all) and 100 (perfect match).
        Provide me some tips on how can I rewrite my CV/cover letter so that the match score increases!

        Here is my CV/cover letter: '{user_document}'.
        Here is the job vacancy: '{job_vacancy}'.

        {output_format}
    """,
    input_variables=["user_document", "job_vacancy", "match_score"],
    partial_variables={"output_format": output_format_tips}
)
custom_chain.add_chain(llm=llm,
                            prompt=prompt_template_tips,
                            chain_key="tip_provider",
                            response_schema=response_schema_tips)

# CHAIN 2: CV/Cover Letter rewriter
response_schema_tips = [
    ResponseSchema(name="document_rewritten",
                   description="Rewritten CV/cover letter with incorporated tips that improve the original match score.")
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
    """,
    input_variables=["user_document", "job_vacancy", "match_score", "tips"],
    partial_variables={"output_format": output_format_tips}
)
custom_chain.add_chain(llm=llm,
                            prompt=prompt_template_tips,
                            chain_key="document_writer",
                            response_schema=response_schema_tips)

# CHAIN 4: Get the match score for the rewritten CV
response_schema_scorer_new = [
    ResponseSchema(name="match_score_new",
                   description="Match score between the uploaded CV/cover letter and the job vacancy.")
]
output_format_scorer_new = StructuredOutputParser.from_response_schemas(
    response_schema_scorer_new).get_format_instructions()
prompt_template_scorer_new = PromptTemplate(
    template="""
        Analyze my CV/cover letter and a job vacancy in order to provide a match score between them.
        I want to know how good of a fit I am.
        Match score should be between 1 (not a good match at all) and 100 (perfect match).

        Here is my {document_type}: '{document_rewritten}'.
        Here is the job vacancy: '{job_vacancy}'.

        {output_format}
    """,
    input_variables=["document_rewritten", "job_vacancy", "document_type"],
    partial_variables={"output_format": output_format_scorer_new}
)
custom_chain.add_chain(llm=llm,
                        prompt=prompt_template_scorer_new,
                        chain_key="match_scorer",
                        response_schema=response_schema_scorer_new)

response = custom_chain.run(
    user_document=user_document,
    job_vacancy=job_vacancy,
    document_type=document_type,
    verbose=True,
    include_all_outputs=True
)
print(response)