from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
from random import randint
from tensorflow import keras
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def RandomForest(dataset, output, random):
      features = np.array(dataset)
      NUM_WORDS = 10000
      (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
      train_dataset = dataset.sample(frac=0.6667,random_state=random)
      test_dataset = dataset.drop(train_dataset.index)

      train_stats = train_dataset.describe()
      train_stats.pop(output)
      train_stats = train_stats.transpose()
      train_labels = train_dataset.pop(output)
      test_labels = test_dataset.pop(output)
      
      def norm(x):
            return ((x - train_stats['mean']) / train_stats['std'])
      normed_train_data = norm(train_dataset)
      normed_test_data = norm(test_dataset)
      test_features = normed_test_data
      train_features = normed_train_data
      rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
      
      rf.fit(train_features, train_labels)
      predictions = rf.predict(test_features)
      errors = abs(predictions - test_labels)
      
      mape = 100 * (errors / test_labels)
      accuracy = 100 - np.mean(mape)

      new_test_predictions = []
      for i in predictions:
            new_test_predictions.append(round(i))

      results = pd.DataFrame()
      
      results['Actual Value'] = test_labels
      results['Prediction'] = new_test_predictions
      return results, accuracy


def main(raw_dataset, indicatorPredictionName):
      raw_dataset[indicatorPredictionName] = raw_dataset[indicatorPredictionName].astype(float)
      dataset = raw_dataset.copy()
      dataset.pop('countryName')
      dataset.pop('year')

      origin = dataset.pop('regionName')
      for region in raw_dataset['regionName'].unique():
            dataset[region] = (origin == region)*1.0
      dataset.tail()

      random = randint(0,1000)
      result, accuracy = RandomForest(dataset, indicatorPredictionName, random)
      return result, accuracy