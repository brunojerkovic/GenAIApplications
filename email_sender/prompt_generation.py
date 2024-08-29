from langchain.prompts import PromptTemplate
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.utilities import ZapierNLAWrapper
from langchain_community.agent_toolkits import ZapierToolkit
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub


class CustomLLM:
    def __init__(self,
                 temperature: float = 1.,
                 model_name: str = 'gpt-4o'):  # gpt-3.5-turbo-0125
        # Model-related instantiations
        # self.llm_pipeline = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
        self.llm_pipeline = ChatOpenAI(temperature=temperature, model_name=model_name)

        # Get the tools
        zapier = ZapierNLAWrapper()
        toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
        tools = toolkit.tools

        # Initialize agent
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm_pipeline, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Init whisper
        # self.model_whisper = whisper.load_model("tiny")
        self.model_whisper_client = OpenAI()

        # agent_executor.invoke({"input": "Who won 2023 world series?"})

        self.prompt_template = PromptTemplate(
            template="""
                Send an email to {receiver} via gmail summarizing the following text: '''{text}'''
            """,
            input_variables=["receiver", "text"]
        )

    def send_email(self, file, receiver: str) -> None:
        # Change email format
        #audio = AudioSegment.from_file(file, format="mp3")
        #raw_data = audio.raw_data
        #audio_array = np.frombuffer(raw_data, dtype=np.double)
        #if audio.channels == 2:
        #    print("CHANGED CHANNELS FROM SHAPE: ", audio_array.shape)
        #    audio_array = audio_array.reshape((-1, 2)).T
        #    print("INTO SHAPE: ", audio_array.shape)

        # Transcribe file
        #text = self.model_whisper.transcribe(audio_array)
        text = self.model_whisper_client.audio.transcriptions.create(
            model="whisper-1",
            file=file
        ).text

        # Send email using zapier
        self.agent_executor.invoke({"input": self.prompt_template.format(receiver=receiver, text=text)})  # TODO: finish after the update to langchain is resolved (currently raises error)
