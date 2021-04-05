# CS216 Final Project

## Dataset

[Yelp Dataset Source (Kaggle Link)](https://www.kaggle.com/yelp-dataset/yelp-dataset)

- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_user.json`

[Yelp Covid Response Dataset (Official)](https://engineeringblog.yelp.com/2020/06/how-businesses-have-reacted-to-covid-19-using-yelp-features.html)

- `yelp_academic_dataset_covid_features.json`

[Restaurant Business Rankings 2020](https://www.kaggle.com/michau96/restaurant-business-rankings-2020?select=Top250.csv)

- `Top250.csv`

## Quickstart

Place the above `.json` files into a folder called `data`

The following script converts the files into `.csv`.  *Caution: May take a while to run*

```
python3 load_data_toolkit.py
```

To directly convert from `.json` to a `DataFrame`:

```python
from load_data_toolkit import load_json_to_dataframe

yelp_df = load_json_to_dataframe('data/yelp_academic_dataset_business.json')
print(yelp_df.shape)
```

## Recommender System:

Based on a user’s previous Yelp ratings and restaurants they’ve visited, can we predict another restaurant from the
dataset that they will enjoy using a machine learning model? What factors are most important in identifying a “match”
between a user’s previously high-rated restaurants and a new restaurant for them to try?

### Filtering Data

`semantic_filter.ipynb` allows the data scientist to choose a text `query` and similarity `threshold` to return only the
businesses whose `categories` (classification) provides a sufficient semantic match.
  