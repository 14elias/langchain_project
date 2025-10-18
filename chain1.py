from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()



llm = ChatOpenAI(
    model='gpt-4o',
    temperature = 0.7,
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    openai_api_base = os.getenv("OPENAI_API_BASE"),
    max_tokens = 500
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are comedian who tells jokes about {topic}'),
        ('user', 'tell me {joke_count} jokes')
    ]
)

format_template= RunnableLambda(lambda x : prompt_template.format_prompt(**x) )
# invoke_model = RunnableLambda(lambda x : llm.invoke(x.to_messages()))
# parse_output = RunnableLambda(lambda x : x.content)

#chain = RunnableSequence(first=format_template, middle=[invoke_model], last=parse_output)
chain = format_template | llm | StrOutputParser()

print(chain.invoke({"topic":"teacher","joke_count":2}))
#result = llm.invoke(prompt_template.format_prompt(topic='teacher', joke_count = 2))
#print(result.content)