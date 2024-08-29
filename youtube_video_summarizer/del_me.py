from youtube_transcript_api import YouTubeTranscriptApi
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent, tool
from langchain_openai import ChatOpenAI
from langchain import hub
from dotenv import load_dotenv
load_dotenv()


video_id = "hUrrHPVHeLM"
#transcript = YouTubeTranscriptApi.get_transcript(video_id)
#print(' '.join([t['text'] for t in transcript]))

prompt = hub.pull("hwchase17/react")


@tool
def get_transcript(video_id: str) -> str:
    """Gets the transcript from the Youtube video with an ID of video_id"""
    transcript_raw = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([t['text'] for t in transcript_raw])

    return transcript
tools = [get_transcript]
model = ChatOpenAI(temperature=0.5)
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

output = agent_executor.invoke({"input": f"Summarize the Youtube video in 500 words with the ID of: {video_id}"})['output']
print(output)