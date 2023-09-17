from functions import *
import json
import ast
from flask import Flask, jsonify, request
from functions import *
import json
import ast

app = Flask(__name__)

@app.route("/get_recipe/<recipe>", methods=["GET"])
def get_recipe(recipe):
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
        

@app.route("/get_special_recipe/<recipe>", methods=["GET"])
def get_special_recipe(recipe):
    #try:
        ingredients = ["Thuna","Salad","Avocado"]
        possible_products = ["Tomatoes","Lentils","Cucumber"]
        for i in ingredients:
            sust_prod = get_product_from_db(i)
            print(sust_prod)
            possible_products.append(sust_prod)
        
        return json.dumps(possible_products)

@app.route("/get_specific_recipe/<recipe>", methods=["GET"])
def get_specific_recipe(recipe):
    product = pd.read_csv('Migros_case/sust_score.csv')
    # order by bigger sust_score
    product = product.sort_values(by=['sust_score'], ascending=False)
    # get top 10
    healthy_breakfast_ids = product.iloc[[0,7	,9]].product_id.tolist()
    healthy_breakfast_scores = product.iloc[[0,7	,9]].sust_score.tolist()

    product = pd.read_csv('Migros_case/products.csv')
    selected_product = product[product.id.isin(healthy_breakfast_ids)]
    selected_product["image_url"] = selected_product["image"].apply(lambda x: ast.literal_eval(x)["original"])
    selected_product["sust_score"] = healthy_breakfast_scores

    response = jsonify(selected_product[['id','name','sust_score','image_url']].to_dict(orient='records'))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route("/add_something/<product>", methods=["GET"])
def add_something(product: str):
    #try:
        # read product csv
        product = pd.read_csv("Migros_case/products.csv")
        item = '104232900000'
        # get product from id
        product = product[product['id'] == item]
        product["image_url"] = product["image"].apply(lambda x: ast.literal_eval(x)["original"])
        product["sust_score"] = 3.603000


        return product[['id','name','sust_score','image_url']].to_json(orient='records')

@app.route("/receive_user_relevant/<num_recommendations>", methods=["GET"])
def receive_user_relevant(num_recommendations):
        user_id = "100009"

        score_df = pd.read_csv('Migros_case/sust_score.csv')
        print(score_df.columns)
        #put into string
        score_df['product_id'] = score_df['product_id'].astype('string')

        triplets_df = pd.read_csv('Migros_case/triplets.csv')
        #put all into string
        triplets_df['product_id'] = triplets_df['product_id'].astype('string')
        triplets_df['user_id'] = triplets_df['user_id'].astype('string')

        similarity_df = pd.read_pickle('Migros_case/similarity_df.pkl')
        #put into string
        similarity_df.index = similarity_df.index.astype('string')
        similarity_df.columns = similarity_df.columns.astype('string')
        product_recommended = recommend_similar_users_sustainable_products(similarity_df, user_id, score_df, triplets_df, int(num_recommendations))
        
        products = pd.read_csv("Migros_case/products.csv")


        print(product_recommended)
        product_ids = product_recommended

        # check if all product_ids are in products
        product_ids = [x for x in product_ids if x in products['id'].values]
        print(product_ids)

        product_recommended = products[products['id'].isin(product_ids)]
        product_recommended["image_url"] = product_recommended["image"].apply(lambda x: ast.literal_eval(x)["original"])
        product_recommended["sust_score"] = product_recommended["id"].apply(lambda x: score_df[score_df['product_id'] == x]['sust_score'].values[0])
        return product_recommended[['id','name','sust_score','image_url']].to_json(orient='records')


    

if __name__ == "__main__":
    app.run(debug=True)