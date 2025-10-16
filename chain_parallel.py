from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model='gpt-4o',
    temperature = 0.7,
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    openai_api_base = os.getenv("OPENAI_API_BASE"),
    max_tokens = 500
)


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","your are an expret product reviewer"),
        ("human", "list the main features of the product {product_name}")
    ]
)


def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are an expert product reviewer"),
            ("human", " given these features: {features} list the pros of these features")
        ]
    )

    return pros_template.format_prompt(features=features)


def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "you are an expert product reviewer"),
            ("human", " given these features: {features} list the cons of these features")
        ]
    )

    return cons_template.format_prompt(features=features)


def combine(pros,cons):
    return f"pros:\n {pros}\n cons:\n {cons}"


pros_chain = (
    RunnableLambda(lambda x:analyze_pros(x)) | model | StrOutputParser()
)

cons_chain = (
    RunnableLambda(lambda x:analyze_cons(x)) | model | StrOutputParser()
)


chain = (
    RunnableLambda(lambda x: prompt_template.format_prompt(**x)) 
    | model
    |StrOutputParser()
    |RunnableParallel(branch={"pro":pros_chain, "con":cons_chain})
    |RunnableLambda(lambda x: combine(x['branch']['pro'], x['branch']['con']))
)

result = chain.invoke({"product_name":"mackbook pro"})
print(result)