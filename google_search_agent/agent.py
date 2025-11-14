from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
import asyncio
import sys
import os
from dotenv import load_dotenv

# Load .env file from the current agent's directory
load_dotenv()
api_key = os.environ["GOOGLE_API_KEY"]

# Import shared configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import retry_config


root_agent = Agent(
    name="helpful_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=api_key,
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

""" runner = InMemoryRunner(agent=root_agent)
#session = runner.session
async def main():
    result = await runner.run_debug(
       "Who is Elon Musk?"
    )
    return result

result = asyncio.run(main()) """