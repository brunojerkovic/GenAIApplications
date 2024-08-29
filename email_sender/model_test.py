from langchain_community.agent_toolkits import ZapierToolkit
from langchain_community.utilities.zapier import ZapierNLAWrapper
from dotenv import load_dotenv
load_dotenv()

# Get the tools
zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
tools = toolkit.get_tools()
print(tools)


print("HI")
