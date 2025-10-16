from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model='gpt-4o',
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    max_tokens=500
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpfull assistant"),
        ("human","classify the sentiment of this feedback as positive,negative,neutral or escalate:{feedback}")
    ]
)


positive_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpfull assistant"),
        ("human","generate a thankyou note for this positive feedback:{feedback}")
    ]
)

negative_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpfull assistant"),
        ("human","generate a response addressing this negative feedback:{feedback}")
    ]
)

neutral_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpfull assistant"),
        ("human","generate a request for more details for this neutral feedback:{feedback}")
    ]
)

escalate_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpfull assistant"),
        ("human","generate a message to escalate this feedback to a human agent:{feedback}")
    ]
)


branch = RunnableBranch(
    (
       lambda x: 'positive' in x,
       positive_prompt_template | model | StrOutputParser()
    ),
    (
       lambda x: 'negative' in x,
       negative_prompt_template | model | StrOutputParser()
    ),
    (
       lambda x: 'positive' in x,
       positive_prompt_template | model | StrOutputParser()
    ),
    escalate_prompt_template | model | StrOutputParser()
    
)


chain = prompt_template | model | StrOutputParser() | branch

result = chain.invoke({'feedback':"the product is great. I really enjoyed using it."})

print(result)