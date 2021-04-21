## Project Prototype

### Introduction and Research Questions

Originally, in our project proposal, we had decided on two research questions we would be trying to approach. The first research question is shown below:

*Based on a user’s previous Yelp ratings and restaurants they’ve visited, can we predict another restaurant from the dataset that they will enjoy using a machine learning model? What factors are most important in identifying a “match” between a user’s previously high-rated restaurants and a new restaurant for them to try?*

Our second research question revolved around trying to predict what changes the most successful businesses were making during the Covid-19 pandemic to still attract customers. This research question would have required data about business performance during the pandemic, either in the form of yelp ratings or in the form of earnings/foot traffic. To support our analysis of this second research question, we had also thought about using sentiment analysis on business reviews left during the months of the pandemic. Sentiment analysis of reviews is a lengthy, computationally costly process, as we will discuss later in the prototype, so using it as one of the key parts of our project would have been risky and difficult to implement.

Eventually, we decided with the help and guidance of our TA, that approaching both research questions would be too much for the scope of this project. We decided to choose the first research question regarding a "recommender" system for a user based on their ratings and history. We think the research question listed above is substantial as it addresses both trying to discover which factors are most important in identifying a match between the user's previously visited restaurants and one of the restaurants in the dataset, and it also addresses the implementation of these factors in the form of a recommender system. The research question is feasible, as creating a recommender system based on the factors included in the dataset is definitely possible due to the size of the *Yelp Dataset*. The data is thorough and this helps streamline our process. This research question is also relevant. Recommender systems are an integral part of many real companies' goals; for example, after you eat at a Chinese restaurant, the Yelp app may recommend other Chinese restaurants for you, reducing the need to search for a similar experience. Furthermore, apps like Yelp, Doordash, and Uber Eats use your history to recommend similar restaurants for you to visit or order from in the future.

### Data Sources

The data sources used in this set include the *Yelp Dataset* and *Restaurant Business Rankings 2020*.  Both datasets are also hosted on Kaggle, which was preferred due to the faster download times.

The *Restaurant Business Rankings 2020* dataset was straightforward to use due to the direct comma-separated values format.  The *Yelp Dataset* required converting JSON files to CSV for easier processing with the `pandas` library.  Both datasets required little amounts of data-wrangling.  For the *Yelp Dataset*, important columns (such as the business category labels) that were missing data were dropped using `dropna` functionality from `pandas` before further processing.  

The Yelp dataset was chosen due to its massive size, relative completeness, and popularity, making it ideal for a recommendation system.  Prior to additional filtering, the Yelp dataset had a total of 160,585 businesses, 2,189,457 users, and 8,635,403 reviews.  The *Restaurant Business Rankings 2020* dataset was chosen to include a broad number of restaurant related terms, useful for semantic filtering.  Additionally, the *Yelp Dataset* is practically tailored for recommendation sets because the included files for `users`, `reviews`, and `businesses` encode ratings, textual description of each experience, and other attributes for each business that are ideal for comparison.

### Preliminary Results and Methods

[GitHub Repository](https://github.com/georgehong/216-final-proj)

The pretrained word embedder, the Universal Sentence Encoder (USE) has been central to both data wrangling and the internals of the recommender.  The USE is simply in a long line of embedders, preceeded by the older Word2Vec and GloVe.  Word embedders are often trained with neural nets fed with vast corpuses of text, and the resulting models convert words into vectors.  

$$\textrm{input text} \to \textrm{Embedder (Word2Vec/GloVe/USE)} \to \vec{v}$$

At the end of training, some astounding properties can be observed.  Synonyms and semantically related words (such as European countries: France, Spain, and Italy) are converted into vectors that are clustered closer together.  Finally, additional properties can be learned: the canonical example is that the embeddings for *King - Man + Woman* will approximate the embedding for *Queen*.  The USE is especially flexible and accepts single words, sentences, and paragraphs as input.  To compare the similarity between two pieces of text A and text B, the two embeddings $\vec{v}_A$ and $\vec{v}_B$ can be computed.  Cosine similarity, 

$$\textrm{similarity} = \cos{\theta} = \frac{\vec{v}_A \cdot \vec{v}_B}{||\vec{v}_A||||\vec{v}_B||}$$

can be computed between the embeddings to obtain an estimate of the similarity between the text.  Because the Yelp datasets do not explicitly label restaurants, the `categories` string (often multiple labels) for each business in `yelp_academic_dataset_business.csv` can be compared with a pool of cuisine-related words, provided by the *Restaurant Business Rankings 2020* dataset, `Top250.csv`, where all `Segment_Category` labels are pooled into a single String.  A short excerpt of this string derived from `Segment_Category` is:

`Casual Dining & Italian/Pizza Quick Service & Seafood Seafood Quick Service & Beverage BBQ Quick Service & Burger Family Style Casual Dining & Asian Casual Dining...`



The resulting distribution of similarity scores is given by:

![](https://i.imgur.com/jiAHAJ4.png)

Ideally, this distribution would be bimodal (businesses very related to food and businesses very unrelated to food).  However, by choosing a similarity score cutoff such as `threshold=0.15`, non-restaurant businesses should mostly be removed.  At this threshold, 52,700 restaurants (likely) are included.  Please see `semantic_filter.ipynb` for more details.

Based on *Introduction to recommender systems*, a user-item interaction matrix approach was first attempted which would record the users ratings (in stars) for each interaction.  Unfortunately, even when using a heavily reduced dataset of 6152 users and 697 businesses, the matrix was very sparse and the setup was slow.  For more details, please see `collaborative_rec.ipynb` in the repository.

Finally, we chose to match semantic similarity using the USE and weighing the scores of several categories: `categories` and `attributes` (but easily extendable to more).  This recommender system finds the most similar businesses and returns the $k$ best results.  We compare the embeddings of the query versus all of the businesses in the dataset to obtain the scores.

**Example:** For a single textual feature (such as `categories` or `attributes`)

$\textrm{Query Business} \to \textrm{USE} \to \vec{v} = \vec{x}_j$

$$
\textrm{All Business Properties} \to \textrm{USE} \to X = \begin{bmatrix}
   \vec{x}_{1} \\
   \vec{x}_{2} \\
   \vdots \\
   \vec{x}_{m}
\end{bmatrix}
$$
$$
\textrm{Similarity Scores} = \begin{bmatrix}
   \textrm{sim}(\vec{v}, \vec{x}_{1}) \\
   \textrm{sim}(\vec{v}, \vec{x}_{2}) \\
   \vdots \\
   \textrm{sim}(\vec{v}, \vec{x}_{m})
\end{bmatrix} = 
\begin{bmatrix}
   0.2 \\
   0.76\\
   \vdots \\
   0.15
\end{bmatrix} 
$$
We can then conclude business 2 (producing $x_2$) is the best with the embedded features by producing the highest score.


```python
def get_knn(input, embedded_attributes, embedded_categories, df_business, k=5, category_weighting=0.75, min_rating=-1):
    """
    Get the most similar businesses by knn search and sort

    :param input: input business, in the form of a Series
    :param embedded_attributes: services offered at the business
    :param embedded_categories: embedded categories (business type)
    :param df_business: DataFrame of all business info that must match embedded_attributes
    :param k: Number of top results desired
    :param category_weighting: emphasis placed on the contents of the restaurant.
    :return: DataFrame of similar businesses
    """
```
`attributes` is a nested JSON object that contains information about additional elements of the business.  The resulting attributes that are embedded are keys that are not declared `False`.  As a result, the `attribute` strings that are embedded include content such as `GoodForKids Alcohol RestaurantsGoodForGroups R...`.  When selecting restaurant matches, the user can specify greater similarity needed in restaurant type (`categories`) or offerings (`attributes`).

With the query as *Oskar Blues Taproom*, the $k-1$ closests matches are shown below:

![](https://i.imgur.com/m0SohAi.png)

Visual verification appears to confirm that these initial results are appropriate. Please see `business_similarity_rec.ipynb` for full implementation details and the ability to interact with this recommender system with additional queries.


### Reflection and Next Steps

- So far, USE has streamlined the process of filtering through large parts of the dataset, and provides many elements desirable for the final implementation.  Currently, the system appears to work well just by applying USE to two features `categories` and `attributes`. 

- The recommender system should include information from user ratings for a more adaptable system.  `categories` and `attributes` are only a subset of predefined possibilities.  Constructing a sentiment string of all customer reviews pooled together for each business to embed, the system would truly compare what users say about a restaurant and their feelings.  However, the Yelp dataset contains a massive amount of reviews, and the quantity of text (1 million user descriptions, even when limited to the first sentence) is truly a strain on text embedders (taking more than an hour without any results).   

- One important part of our project that we have not yet adressed is the need to split our dataset into training and testing datasets. At the moment, the recommender takes in a restaurant and uses its 'categories' and 'attributes' to find the nearest match. Now that we know that it is possible to create a recommender like the one we had imagined, we need to decide on our final goal and work on splitting the dataset into a training dataset and a testing dataset. This may be difficult, as we do not have data about the actual "closest" restaurants to the ones in our dataset.

- Instead of accepting a business, the recommender system should accept a user profile and return businesses that match businesses the user has had a positive experience with.  Although currently incomplete, given the current design of the system and structure of the dataset, this should be a straightforward inclusion.  

- As mentioned above, we would like to figure out a way to include users' ratings/recommendations into our recommender. Our original idea was to use a specific user's previous ratings and reviews to predict a restaurant that they might like to visit. If using reviews as a predictor seems to be too difficult, one idea we have is to take the user's highest rated previously visited restaurant, and use that restaurant's 'categories', 'attributes', and other features to find another that is similar. We could also take all of the user's previously visited restaurants that they rated higher than a certain number, for example 4 stars, and use the values from all those restaurants to find a match in the dataset using the USE discussed above.

### Sources
- [Semantic similarity with tfhub universal encoder](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder)
- [Introduction to recommender systems](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada)
- *Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow* - Aurelien Geron