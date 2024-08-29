from langchain.prompts import PromptTemplate
from transformers import pipeline


class CustomLLM:
    def __init__(self,
                 temperature: float = 1.,
                 model_name: str = 'gpt-3.5-turbo-0125'):
        # Model-related instantiations
        self.llm_pipeline = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
        self.temperature = temperature
        self.prompt_template = PromptTemplate(
            template="""
                Write a email with {style} style and includes topic :{email_topic}.
                \n\nSender: {sender} \nRecipient: {recipient}
                \n\nEmail Text:
            """,
            input_variables=["style", "email_topic", "sender", "recipient"]
        )

    def translate(self, form_input: str, email_sender: str, email_receiver: str, email_style: str) -> str:
        # Get the response
        messages = [
            {
                "role": "user",
                "content": self.prompt_template.format(form_input=form_input, email_sender=email_sender, email_receiver=email_receiver, email_style=email_style)
            },
        ]
        response = self.llm_pipeline(messages)
        return response
