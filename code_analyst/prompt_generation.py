import json
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import Tuple
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


class CustomLLM:
    def __init__(self,
                 temperature: float = 1.,
                 model_name: str = 'gpt-4o'):  # gpt-3.5-turbo-0125
        # Model
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)

        # Prompts: Code Reviewer
        response_schema_reviewer = [
            ResponseSchema(name="code_rewritten", description="Rewritten code with all mistakes fixed, and optimized for performance and readability. If no mistakes, leave empty"),
            ResponseSchema(name="comments", description="Comments about the uploaded code, reflecting on what was improved in 'code_rewritten'")
        ]
        output_format_reviewer = StructuredOutputParser.from_response_schemas(response_schema_reviewer).get_format_instructions()
        self.prompt_template_reviewer = PromptTemplate(
            template="""
                Analyze my code. Fix all mistakes, optimize it for maximum performance and readability.
                Here is my code: '''{code}'''
                
                Also, provide comments about the uploaded code. Provide these comments reflecting on what was improved in rewritten code.

                {output_format}
            """,
            input_variables=["code"],
            partial_variables={"output_format": output_format_reviewer}
        )

        # Prompts: Code Translator
        response_schema_translator = [
            ResponseSchema(name="code_translated", description="Code translated to a new language that the user requested."),
        ]
        output_format_translator = StructuredOutputParser.from_response_schemas(
            response_schema_translator).get_format_instructions()
        self.prompt_template_translator = PromptTemplate(
            template="""
                Translate my code to {language} language.
                Here is my code: '''{code}'''
                
                {output_format}
            """,
            input_variables=["code", "language"],
            partial_variables={"output_format": output_format_translator}
        )

    def review_code(self, uploaded_file) -> Tuple[str, str]:
        # Review code
        response = self.llm.invoke(self.prompt_template_reviewer.format_prompt(code=uploaded_file)).content.strip("```json").strip("```").strip()
        response_json = json.loads(response)
        code_rewritten, comments = response_json["code_rewritten"], response_json["comments"]

        return code_rewritten, comments

    def translate_code(self, uploaded_file, language: str) -> str:
        # Translate code
        response = self.llm.invoke(self.prompt_template_translator.format_prompt(code=uploaded_file, language=language)).content.strip("```json").strip("```").strip()
        print(response)
        code_translated = json.loads(response)["code_translated"]

        return code_translated
