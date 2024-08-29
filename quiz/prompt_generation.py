from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json
import random


class OpenAILLM:
    def __init__(self, temperature: float = 1.,
                 model_name: str = 'gpt-4',
                 mcq_question_number: int = 10,
                 mcq_false_answer_number: int = 3):
        # Model-related instantiations
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)
        self.Memory = ConversationBufferMemory
        self.chain_summary = load_summarize_chain(self.llm, chain_type="map_reduce", verbose=True)
        self.chain_chat = ConversationChain(llm=self.llm, verbose=False, memory=self.Memory())

        # Other utils instantiation
        self.docs = []
        self.text_splitter = RecursiveCharacterTextSplitter()
        self.chat_document_intro = "Read the following document: "
        self.chat_message_begin = "What would you like to know about the uploaded document?"
        self.mcq_question_number = mcq_question_number
        self.mcq_false_answer_number = mcq_false_answer_number
        self.mcq_intro = f"""
            Generate a question, correct answer and {self.mcq_false_answer_number} possible false answers from the inputted document. 
            Make sure that it is unique from the ones you have generated before!
            Only create 3 possible false answers and a correct answers!
        """
        self.mcq_answer_sheet = []
        self.mcq_query = None

    def upload_text(self, text):
        texts = self.text_splitter.split_text(text)
        self.docs = [Document(text) for text in texts]

    def is_text_uploaded(self):
        return True if self.docs else False

    def empty_text(self):
        self.docs = []
        self.chain_chat.memory = self.Memory()
        self.mcq_answer_sheet = []

    def get_text_summary(self):
        summary = self.chain_summary.run(self.docs)
        return summary

    def start_chat(self):
        # Add document to the system's context
        self.chain_chat.memory.save_context({"input": self.chat_document_intro}, {"output": ""})
        for doc in self.docs:
            self.chain_chat.memory.save_context({"input": doc.page_content}, {"output": ""})

        return str(self.chain_chat.memory), self.chat_message_begin

    def get_chat_response(self, user_input: str):
        response = self.chain_chat.predict(input=user_input)
        return response

    def start_mcq(self):
        # Instantiate response schema to define JSON output
        response_schemas = [
            ResponseSchema(name="question", description="Question generated from provided document."),
            ResponseSchema(name="answer", description="One correct answer for the asked question."),
            ResponseSchema(name="choices",
                           description=f"{self.mcq_false_answer_number} available false options for a multiple-choice question in comma separated."),
        ]
        output_format_instructions = StructuredOutputParser.from_response_schemas(
            response_schemas).get_format_instructions()

        # Define the prompt that will be used for MCQ questions
        prompt = PromptTemplate(
            template="{task_instructions}\n {output_format_instructions}",
            input_variables=["task_instructions", "output_format_instructions"]
        )

        # Get the MCQ query based on the prompt (by filling in the prompt values)
        self.mcq_query = prompt.format(task_instructions=self.mcq_intro,
                                       output_format_instructions=output_format_instructions)

        # Upload the document to the model
        self.start_chat()

    def get_mcq_question(self):
        while True:
            try:
                response = self.chain_chat.predict(input=self.mcq_query)
                response_parsed = json.loads(response[len(r"```json"):-len(r"```")])

                question = response_parsed["question"]
                answers = [response_parsed["answer"]] + [false_answer.strip() for false_answer in
                                                         response_parsed["choices"].split(',')][:self.mcq_false_answer_number]
                break
            except Exception as e:
                print(e)

        self.mcq_answer_sheet.append({
            "question": question,
            "answer": answers[0],
            "user_answer": None,
            "choices": answers
        })
        return question, random.sample(answers, len(answers))

    def mcq_record_answer(self, answer):
        self.mcq_answer_sheet[-1]["user_answer"] = answer

    def get_mcq_score(self):
        score = sum([sheet['answer'] == sheet['user_answer'] for sheet in self.mcq_answer_sheet])
        score_perc = round(score / self.mcq_question_number, 4) * 100

        return score, score_perc
