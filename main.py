from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def generate_pet(animal_type):

    prompt = PromptTemplate(
        input_variables = ['animal_type'],
        template = 'i have a {animal_type} pet and i want a cool name for it.' \
        'suggest me five for my pet '
    )

    llm = ChatOpenAI(
    model='gpt-4o',  # or your correct model name
    temperature=0.7,
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    openai_api_base=os.getenv('OPENAI_API_BASE'), # if using a custom base
    max_tokens=500
    )

    formatted_prompt = prompt.format(animal_type=animal_type)

    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    return response.content

print(generate_pet('dog'))