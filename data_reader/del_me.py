from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.schema import HumanMessage, SystemMessage
load_dotenv()

import pandas as pd

df = pd.read_csv('data_test.csv')
dataframe_description = (f"Dataframe types: {df.dtypes}." +
                              f"\n Dataframe examples: {df.head(10).to_dict(orient='list')}.")

from custom_chain import CustomSequentialChain

custom_chain = CustomSequentialChain()

llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-0125")
query = "The relation between person's age and their survival."

# CHAIN 1: Get the columns
response_schema_columns = [
    ResponseSchema(name="df_columns",
                   description="List of columns relevant for the user question.")
]
format_instructions_columns = StructuredOutputParser.from_response_schemas(
    response_schema_columns).get_format_instructions()
prompt_template_columns = PromptTemplate(
    template="""
        Based on the user question, determine which columns in this dataframe are relevant for answering the user question.
        This is the dataframe description: {dataframe_description}.
        This is the user question: {user_question}.

        {format_instructions_columns}
    """,
    input_variables=["user_question", "dataframe_description"],
    partial_variables={"format_instructions_columns": format_instructions_columns}
)
custom_chain.add_chain(llm=llm, prompt=prompt_template_columns, chain_key="column_picker", response_schema=response_schema_columns)

# CHAIN 2: Get the stat method
response_schema_stat_method = [
    ResponseSchema(name="statistical_test",
                   description="Statistical test to use on the columns to answer user question."),
    ResponseSchema(name="statistical_test_reasoning",
                   description="Reasoning of why was the statistical test chosen."),
]
format_instructions_stat_method = StructuredOutputParser.from_response_schemas(
    response_schema_stat_method).get_format_instructions()
prompt_template_stat_method = PromptTemplate(
    template="""
        Given relevant columns from a dataframe, determine which statistical test to use to answer user question.
        These are the columns: {df_columns}.
        This is the dataframe description: '''{dataframe_description}'''.
        Make sure that there is a Python library that has the implementation of the method for that statistical test.
        Make sure to look at the dataframe description when choosing a dataframe description!
        This is the user question: {user_question}

        {format_instructions_columns}
    """,
    input_variables=["dataframe_description", "user_question", "df_columns"],
    partial_variables={"format_instructions_columns": format_instructions_stat_method}
)
custom_chain.add_chain(llm=llm, prompt=prompt_template_stat_method, chain_key="stat_method_decider", response_schema=response_schema_stat_method)

# CHAIN 3: Code
response_schema_code = [
    ResponseSchema(name="code", description="Code to run on the columns in order to perform the statistical test."),
]
format_instructions_code = StructuredOutputParser.from_response_schemas(response_schema_code).get_format_instructions()
prompt_template_code = PromptTemplate(
    template="""
        Given the chosen statistical method: {statistical_test}, write a Python code to perform that statistical test.
        These are the columns: {df_columns}.
        The variable name of the dataframe is 'df'. 
        The result of the statistical test should be saved in the variable 'stat_test_result'.
        
        {format_instructions_columns}
    """,
    input_variables=["statistical_test", "df_columns"],
    partial_variables={"format_instructions_columns": format_instructions_code}
)
custom_chain.add_chain(llm=llm, prompt=prompt_template_code, chain_key="code_runner", response_schema=response_schema_code)

# CHAIN 4: Code
response_schema_code_verifier = [
    ResponseSchema(name="code_verified", description="The code that properly runs the task and works well."),
]
format_instructions_code_verifier = StructuredOutputParser.from_response_schemas(response_schema_code_verifier).get_format_instructions()
prompt_template_code_verifier = PromptTemplate(
    template="""
        Given the chosen statistical method: {statistical_test}, check that the following Python code does the job well. If it does not, then rewrite it!
        Here is the Python code: '''{code}'''.
        These are the columns of the dataframe to use: {df_columns}.
        The variable name can only be 'df'. Assume you already have the dataset accessible as a 'df' variable (with type pandas.DataFrame).
        The result of the statistical test should be saved in the variable 'stat_test_result'.
        The code must be runnable and without errors.

        {format_instructions_columns}
    """,
    input_variables=["code", "statistical_test", "df_columns"],
    partial_variables={"format_instructions_columns": format_instructions_code_verifier}
)
custom_chain.add_chain(llm=llm, prompt=prompt_template_code_verifier, chain_key="code_verifier",
                       response_schema=response_schema_code_verifier)

o = custom_chain.run(user_question=query,
                     dataframe_description=dataframe_description,
                     verbose=True,
                     include_all_outputs=True)

print(o)
