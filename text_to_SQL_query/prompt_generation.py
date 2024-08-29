from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class OpenAILLM:
    def __init__(self,
                 temperature: float = 1.,
                 model_name: str = 'gpt-3.5-turbo-0125'):
        # Model-related instantiations
        self.client = ChatOpenAI()
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_template = PromptTemplate(
            template="""
            Answer only with the translated query.
            If it is a programming language, then write code in the output.
            Translate this '''{user_query}''' to this language: '''{language}'''.
            """,
            input_variables=["user_query", "language"]
        )

    def translate(self, user_query: str, language: str) -> str:
        # Get the response
        response = self.client.invoke(input=self.prompt_template.format(user_query=user_query, language=language))
        return response.content
