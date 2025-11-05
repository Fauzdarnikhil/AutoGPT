import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()  # load API key from .env

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

# LLM setup using OpenRouter
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  
    temperature=0.3,
    openai_api_key=api_key,
    openai_api_base=api_base
)

def generate_subtasks(user_query):
    prompt = f"""
    You are a research assistant AI.
    Break this research question into clear, actionable subtasks.
    Each subtask should be specific and help move toward the final goal.
    Decide the number of subtasks based on question complexity.

    Research Question: "{user_query}"

    List the subtasks in numbered format.
    """
    response = llm.invoke(prompt)
    result = response.content

    
    subtasks = []
    for line in result.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            task = line.split(".", 1)[-1].strip()
            subtasks.append(task)
    return subtasks
