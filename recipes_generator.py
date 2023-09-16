
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

load_dotenv()  # take environment variables from .env.
API_KEY = os.environ['API_KEY']
llm= OpenAI(openai_api_key=API_KEY)
# meal template
meal_template = PromptTemplate(
    input_variables=["ingredients"],
    template="Give me an example of 2 meals that could be made using the following ingredients: {ingredients}",
)

llm_chain = LLMChain(prompt=meal_template, llm=llm)
question= input()
print(llm_chain.run(question))






