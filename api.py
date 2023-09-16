from fastapi import FastAPI
from typing import List
import uvicorn
from functions import *
import json

app = FastAPI()

@app.get("/get_similar_products/{product}")
async def get_similar_products(product: str):
    return products.get(product, [])

@app.get("/get_recipe/{recipe}")
async def get_recipe(recipe: str):
    try:
        ingredients = get_ingredients_for_recipe_from_llm(recipe)
        possible_products = []
        print(ingredients)
        for i in ingredients:
            sust_prod = get_similar_sustainable_product_from_text(i)
            print(sust_prod)
            possible_products.append(sust_prod)
        
        return json.dumps(possible_products)
    except Exception as e:
        print(f"Error: {e}")
        return json.dumps(["error"])
        

@app.get("/get_special_recipe/{recipe}")
async def get_special_recipe(recipe: str):
    #try:
        ingredients = ["Thuna","Salad","Avocado"]
        possible_products = ["Tomatoes","Lentils","Cucumber"]
        for i in ingredients:
            sust_prod = get_product_from_db(i)
            print(sust_prod)
            possible_products.append(sust_prod)
        
        return json.dumps(possible_products)


    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
