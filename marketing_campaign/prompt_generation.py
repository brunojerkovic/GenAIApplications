from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate, LengthBasedExampleSelector
import pandas as pd


def prompt_generator(text: str, task_type: str, age_group: str, token_number: str):
    # Load the data
    data = pd.read_csv("data.csv")
    examples = data[(data.task_type == task_type) & (data.age_group == age_group)]

    if len(examples):
        # Set example prompt template
        example_template = """
        Question: {query}
        Response: {answer}
        """
        example_prompt = PromptTemplate(
            input_variables=["query", "answer"],
            template=example_template
        )
        print(examples.to_dict(orient='records'))
        example_selector = LengthBasedExampleSelector(
            examples=examples.to_dict(orient='records'),
            example_prompt=example_prompt,
            max_length=token_number
        )
        prefix = """
            You are a {template_age_option}, and rewrite the question text into {template_task_type}.
            Here are some examples:
            """
        suffix = """
            Question: {template_user_input}
            Response: ''
            """

        # Generate final prompt
        prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["template_user_input", "template_age_option", "template_task_type"],
            example_separator="\n"
        )
    else:
        template = """
        You are a {template_age_option}, and rewrite the question text  {template_task_type}.
        
        Question: {template_user_input}
        Response: ''
        """
        prompt = PromptTemplate(
            input_variables=["template_user_input", "template_age_option", "template_task_type"],
            template=template
        )

    # Pass the query to LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=.9)
    print(prompt.format(template_user_input=text, template_age_option=age_group, template_task_type=task_type))
    response = llm.invoke(prompt.format(template_user_input=text, template_age_option=age_group, template_task_type=task_type)).content

    return response
