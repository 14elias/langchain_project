from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv
import os
from langchain.agents import initialize_agent, load_tools

load_dotenv()


llm = ChatOpenAI(
    model='gpt-4o',  # or your correct model name
    temperature=0.7,
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    openai_api_base=os.getenv('OPENAI_API_BASE'), # if using a custom base
    max_tokens=500
    )


def generate_pet(animal_type,animal_color):

    prompt = PromptTemplate(
        input_variables = ['animal_type','animal_color'],
        template = 'i have a {animal_type} pet and i want a cool name for it.' \
        'and its color is {animal_color}. suggest me five for my pet '
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain({'animal_type':animal_type, 'animal_color':animal_color})

    return response

def langchain_agent():

    agent = initialize_agent(
        llm=llm,
        tools=load_tools(['wikipedia','llm-math'], llm=llm),
        agent_type="zero-shot-react-description",
        verbose=True
    )

    response = agent.run("what is maximum age of dog. then multiply it by 3 and tell me the result")
    print(response)




if __name__ == "__main__":
    # print(generate_pet('dog','black'))
    langchain_agent()