
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
import re 

load_dotenv()  # take environment variables from .env.
API_KEY = os.environ['API_KEY']
llm= OpenAI(openai_api_key=API_KEY)
# meal template
meal_template = PromptTemplate(
    input_variables=["recipes"],
    template="Give me the ingredients of: {recipes} and each ingredient in one line",
)

llm_chain = LLMChain(prompt=meal_template, llm=llm)
question= input()
ingredients_string=llm_chain.run(question)

def get_ingredients(ingredients):
    lines = [line for line in ingredients.splitlines() if line.strip()]
    # Join the non-empty lines back into a single string
    ingredients_list=[]
    ingredients_string= '\n'.join(lines)
    for ingredient in ingredients_string.splitlines():
        match = re.search(r'\((.*?)\)', ingredient)
        if match:
            example_text = match.group(1)  # Get the text inside the parentheses
            example_list = [topping.strip() for topping in example_text.split(',')]
            ingredients_list.append(example_list)
            
        else:
             ingredients_list.append(ingredient)
        
    
    return ingredients_list

    
ingredients_list=get_ingredients(ingredients_string)
print(ingredients_list)





