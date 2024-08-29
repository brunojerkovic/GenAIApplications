from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain import hub
from langchain.prompts import PromptTemplate


@tool
def get_transcript(video_id: str) -> str:
    """Gets the transcript from the YouTube video with an ID of video_id"""
    if 'video_id' in video_id:
        video_id = video_id[len('video_id="'):-1]
    transcript_raw = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([t['text'] for t in transcript_raw])
    return transcript


class OpenAILLM:
    def __init__(self, temperature: float = 1.,
                 model_name: str = 'gpt-3.5-turbo-0125'):
        # Model-related instantiations
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)
        self.prompt_agent = hub.pull("hwchase17/react")
        self.prompt_template_input = PromptTemplate(
            template="Summarize the Youtube video to {word_number} words of a summary. The ID of the video is: {video_id}",
            input_variables=["word_number", "video_id"]
        )

    def get_transcript(self, video_url: str, word_number: int):
        video_id = video_url[len("https://www.youtube.com/watch?v="):]
        tools = [get_transcript]

        agent = create_react_agent(self.llm, tools, self.prompt_agent)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        try:
            prompt_input = self.prompt_template_input.format_prompt(video_id=video_id, word_number=word_number)
            response = agent_executor.invoke({"input": prompt_input})['output']
        except Exception as e:
            response = e
        return response
