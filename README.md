### MIGROS HACKZURICH 2023 Sustanability AI Shopping Challenge

### Our recommender system

We integrate recommendations of both users (collaborative filtering) and products (content-based filtering) to provide a more personalized shopping experience.
In detail:

- Users with similar shopping behavior are clustered together and recommendations are made based on the items they have in common, using Matrix Factorization, all depending on the history of purchases.![Alt text](/assets/rs.png?raw=true).
- Products are clustered based on their similarity (in an embedding space) depending on the type of product and the brand. ![Alt text](/assets/sim.png?raw=true). This way, we are able to recommend similar products but greener. We make sure that the recommended products are not higher than 15% of the original price.

### File structure

- cloud_function: api deployed on google cloid that serves the recommendation systems to the frontend
- notebooks: various notebooks peforming processing of data and testing
- Migros_case: Data of Migros
- functions.py: all implementation functions of recommandation systems (user centric + similarity across products)
