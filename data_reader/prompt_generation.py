from langchain_openai import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain.schema import SystemMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from custom_chain import CustomSequentialChain
import pandas as pd


class OpenAILLM:
    def __init__(self, temperature: float = 1.,
                 model_name: str = 'gpt-3.5-turbo-0125'):
        # Model-related instantiations
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)

        # Chat-related params
        self.chat_message_begin = "What would you like to chat about the data?"
        self.dataframe_agent = None

        # Visualization-related params
        self.df = None
        self.dataframe_description = None
        self.visualize_message_begin = "What would you like to visualize?"
        system_message = SystemMessage(
            """Write Python code to generate plots based on descriptions."""
        )
        human_message_template = HumanMessagePromptTemplate.from_template("""
            This is the description of the plot: {plot_description}.
            If it helps, this is the description of the dataframe: {dataframe_description}.
            Only use matplotlib to do this, and name the figure variable to be 'fig'.
            The variable for the dataframe is called 'self.df'. Do not assume that any other variables already exist!
            Next, write Python code to visualize this plot.
            Your answer should only contain python code.
        """)
        self.visualization_prompt = ChatPromptTemplate(
            messages=[system_message, human_message_template],
            input_variables=["plot_description", "dataframe_description"]
        )

    def upload_df(self, df):
        self.df = df
        self.dataframe_agent = create_pandas_dataframe_agent(
            self.llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
        )
        self.dataframe_description = (f"Dataframe types: {df.dtypes}." +
                                      f"\n Dataframe examples: {df.head(10).to_dict(orient='list')}.")

    def is_df_uploaded(self):
        return True if self.df is not None else False

    def empty_df(self):
        self.df = None
        self.dataframe_agent = None
        self.dataframe_description = None

    def get_chat_start_message(self):
        return self.chat_message_begin

    def get_visualize_start_message(self):
        return self.visualize_message_begin

    def get_stats_start_message(self):
        return self.visualize_message_begin

    def get_chat_response(self, user_input: str):
        try:
            response = self.dataframe_agent.invoke(user_input)['output']
        except Exception as e:
            response = e
        return response

    def get_visualization_response(self, plot_description: str):
        # Matplotlib figure
        fig = None

        try:
            response = self.llm.invoke(
                self.visualization_prompt.format_prompt(
                    plot_description=plot_description,
                    dataframe_description=self.dataframe_description
                )
            )
            response = "~~~py\n" + response.content[len("```python"):-len("```")] + "\n~~~"

            # Run the code to get the figure
            namespace = {'fig': fig, 'self': self}
            exec(response, namespace)
            fig = namespace['fig']
        except Exception as e:
            response = str(e)
        return fig, response

    def get_stats_response(self, user_question: str):
        # Get the columns from the dataset (and a few values)
        custom_chain = CustomSequentialChain()

        # CHAIN 1: Get the columns
        response_schema_columns = [
            ResponseSchema(name="df_columns", description="List of columns relevant for the user question.")
        ]
        format_instructions_columns = StructuredOutputParser.from_response_schemas(response_schema_columns).get_format_instructions()
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
        custom_chain.add_chain(llm=self.llm,
                               prompt=prompt_template_columns,
                               chain_key="column_picker",
                               response_schema=response_schema_columns)

        # CHAIN 2: Get the stat method
        response_schema_stat_method = [
            ResponseSchema(name="statistical_test", description="Statistical test to use on the columns to answer user question."),
            ResponseSchema(name="statistical_test_reasoning", description="Reasoning of why was the statistical test chosen."),
        ]
        format_instructions_stat_method = StructuredOutputParser.from_response_schemas(response_schema_stat_method).get_format_instructions()
        prompt_template_stat_method = PromptTemplate(
            template="""
                Given relevant columns from a dataframe, determine which statistical test to use to answer user question.
                These are the columns: {df_columns}.
                This is the dataframe description: '''{dataframe_description}'''.
                Make sure that there is a Python library that has the implementation of the method for that statistical test.
                Make sure to look at the dataframe description when choosing a dataframe description!
                This is the user question: {user_question}.

                {format_instructions_columns}
            """,
            input_variables=["dataframe_description", "user_question", "df_columns"],
            partial_variables={"format_instructions_columns": format_instructions_stat_method}
        )
        custom_chain.add_chain(llm=self.llm,
                               prompt=prompt_template_stat_method,
                               chain_key="stat_method_decider",
                               response_schema=response_schema_stat_method)

        # CHAIN 3: Code
        response_schema_code = [
            ResponseSchema(name="code", description="Code to run on the columns in order to perform the statistical test."),
        ]
        format_instructions_code = StructuredOutputParser.from_response_schemas(response_schema_code).get_format_instructions()
        prompt_template_code = PromptTemplate(
            template="""
                Given the chosen statistical method: {statistical_test}, write a Python code to perform that statistical test.
                These are the columns: {df_columns}.
                The variable name of the dataframe is 'self.df'. Do not assume that any other variable is available.
                The result of the statistical test should be saved in the variable 'stat_test_result'.
                Make sure to import all libraries necessary to run the code.

                {format_instructions_columns}
            """,
            input_variables=["statistical_test", "df_columns"],
            partial_variables={"format_instructions_columns": format_instructions_code}
        )
        custom_chain.add_chain(llm=self.llm,
                               prompt=prompt_template_code,
                               chain_key="code_runner",
                               response_schema=response_schema_code)

        # CHAIN 4: Code
        response_schema_code_verifier = [
            ResponseSchema(name="code_verified", description="The code that properly runs the task and works well."),
        ]
        format_instructions_code_verifier = StructuredOutputParser.from_response_schemas(
            response_schema_code_verifier).get_format_instructions()
        prompt_template_code_verifier = PromptTemplate(
            template="""
                Given the chosen statistical method: {statistical_test}, check that the following Python code does the job well. If it does not, then rewrite it!
                Here is the Python code: '''{code}'''.
                These are the columns of the dataframe to use: {df_columns}.
                The variable name can only be 'self.df'. Assume you already have the dataset accessible as a 'self.df' variable (with type pandas.DataFrame).
                The result of the statistical test should be saved in the variable 'stat_test_result'.
                The code must be runnable and without errors.

                {format_instructions_columns}
            """,
            input_variables=["code", "statistical_test", "df_columns"],
            partial_variables={"format_instructions_columns": format_instructions_code_verifier}
        )
        custom_chain.add_chain(llm=self.llm, prompt=prompt_template_code_verifier, chain_key="code_verifier",
                               response_schema=response_schema_code_verifier)

        try:
            response = custom_chain.run(user_question=user_question,
                                        dataframe_description=self.dataframe_description,
                                        verbose=False,
                                        include_all_outputs=True)

            # Get the outputs of the response
            response_formatted = f"I have chosen the following statistical test: {response['statistical_test']}. \nThis is because: {response['statistical_test_reasoning']}\n\n"
            code = "This is the code for the test: \n~~~py\n" + response['code_verified'] + "\n~~~"
        except Exception as e:
            response_formatted = str(e)
            code = ""

        return response_formatted, code
