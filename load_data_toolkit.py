import json
import numpy as np
import pandas as pd

"""
Converts JSON files to CSV through the use of DataFrames

Additionally provides support for loading JSON into Pandas DataFrames
"""


def load_json_to_dataframe(file_path, limit=-1):
    """
    Loads JSON file and returns a formatted DataFrame.
    :param file_path: path to JSON file ex: 'data/yelp_academic_dataset_business.json'
    :param limit: instances to read
    :return: DataFrame corresponding to the loaded JSON
    """
    # Code source: https://www.kaggle.com/ravijoe/loading-yelp-json-data-and-visualization
    data = []
    with open(file_path) as fl:
        for i, line in enumerate(fl):
            data.append(json.loads(line))
            if limit != -1 and i + 1 >= limit:
                break

    return pd.DataFrame(data)


if __name__ == '__main__':
    names = ['yelp_academic_dataset_business',
             'yelp_academic_dataset_review',
             'yelp_academic_dataset_user',
             'yelp_academic_dataset_covid_features']

    for file_name in names:
        df = load_json_to_dataframe('data/' + file_name + '.json')
        df.to_csv('data/' + file_name + '.csv')
