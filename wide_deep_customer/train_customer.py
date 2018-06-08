# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#Code: Mark Bolton - Customer prediction model
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys

import tensorflow as tf

_CSV_COLUMNS = [
    'size', 'customer', 'years_trading', 'years_customer', 'turn_over', 'bop',
    'credit_rating', 'employees', 'corporate_family', 'country',
    'market', 'period', 'abc', 'ly_spend', 'budget_spend', 'curr_period_fc', 'last_period', 'curr_period_ly',
    'weather_fc', 'make_budget_lp', 'make_budget_ply', 'make_budget'
]

_CSV_COLUMN_DEFAULTS = [[''], [''], [0], [0], [0], [0],
                        [''], [0], [''], [''],
                        [''], [0], [''], [0], [0], [0], [0],[0],
                        [''],[''],[''],['']]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default=(r"c:\\_dev\\models\\official\\wide_deep_customer\\model"),
    help='Base directory for the model.')

parser.add_argument(
    '--model_type', type=str, default='wide_deep',
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--train_epochs', type=int, default=40, help='Number of training epochs.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=2,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=40, help='Number of examples per batch.')

parser.add_argument(
    '--train_data', type=str, default=(r"c:\\_dev\\models\\official\\wide_deep_customer\\data\\cust.data"),
    help='Path to the training data.')

parser.add_argument(
    '--test_data', type=str, default=(r"c:\\_dev\\models\\official\\wide_deep_customer\\data\\cust.test"),
    help='Path to the test data.')

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns

  size = tf.feature_column.categorical_column_with_vocabulary_list(
      'size', [
          'VS', 'S', 'M', 'XL', 'XXL'])


  customer = tf.feature_column.numeric_column('customer')
  years_trading = tf.feature_column.numeric_column('years_trading')
  years_customer = tf.feature_column.numeric_column('years_customer')
  turn_over = tf.feature_column.numeric_column('turn_over')
  bop = tf.feature_column.numeric_column('bop')
  credit_rating = tf.feature_column.categorical_column_with_vocabulary_list('credit_rating', [
      'aaa', 'aa', 'a', 'bbb', 'bb', 'b', 'ccc', 'cc', 'c', 'ddd', 'dd', 'd'])
  employees = tf.feature_column.numeric_column('employees')
  corporate_family = tf.feature_column.categorical_column_with_vocabulary_list('corporate_family', [
      'y', 'n'])
  country = tf.feature_column.categorical_column_with_vocabulary_list('country', [
      'uk', 'norway', 'sweden', 'finland', 'denmark'])
  market = tf.feature_column.categorical_column_with_vocabulary_list('market', [
      'pitched', 'flat'])
  period = tf.feature_column.numeric_column('period')
  abc = tf.feature_column.categorical_column_with_vocabulary_list('abc', [
      'a', 'b', 'c'])
  ly_spend = tf.feature_column.numeric_column('ly_spend')
  budget_spend = tf.feature_column.numeric_column('budget_spend')
  curr_period_fc = tf.feature_column.numeric_column('curr_period_fc')
  last_period = tf.feature_column.numeric_column('last_period')
  curr_period_ly = tf.feature_column.numeric_column('curr_period_ly')
  weather_fc = tf.feature_column.categorical_column_with_vocabulary_list('weather_fc', [
      'l-rain', 'rain', 'h-rain', 'wind', 's-wind', 'frost', 'l-snow', 'snow', 'h-snow',
      'sun', 'hot'])
  make_budget_lp = tf.feature_column.categorical_column_with_vocabulary_list('make_budget_lp', [
      'y', 'n'])
  make_budget_ply = tf.feature_column.categorical_column_with_vocabulary_list('make_budget_ply', [
      'y', 'n'])
  make_budget = tf.feature_column.categorical_column_with_vocabulary_list('make_budget', [
      'y', 'n'])

  # Wide columns and deep columns.
  base_columns = [
      size, credit_rating, corporate_family, market, country, abc, ly_spend, budget_spend, weather_fc,
      make_budget_lp, make_budget_ply,
  ]

  crossed_columns = [
      tf.feature_column.crossed_column(
          ['size', 'weather_fc'], hash_bucket_size=1000),
  ]

  wide_columns = base_columns + crossed_columns

  deep_columns = [
      years_trading,
      years_customer,
      turn_over,
      employees,
      period,
      ly_spend,
      budget_spend,
      curr_period_fc,
      last_period,
      curr_period_ly,
      tf.feature_column.indicator_column(abc),
      tf.feature_column.indicator_column(corporate_family),
      tf.feature_column.indicator_column(country),
      tf.feature_column.indicator_column(market),
      tf.feature_column.indicator_column(credit_rating),
      tf.feature_column.indicator_column(size),
      tf.feature_column.indicator_column(weather_fc),
      tf.feature_column.indicator_column(make_budget_lp),
      tf.feature_column.indicator_column(make_budget_ply)
  ]

  return wide_columns, deep_columns
def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)
def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    #print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    #print("mb.........dict....", features)
    labels = features.pop('make_budget')
    return features, tf.equal(labels, 'y')

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels


def main(unused_argv):
  # Clean up the model directory if present
  record = 1
  shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
  model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

  # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
  for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    model.train(input_fn=lambda: input_fn(
        FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

    results = model.evaluate(input_fn=lambda: input_fn(
        FLAGS.test_data, 1, False, FLAGS.batch_size))

    # Display evaluation metrics
    #print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
    #print('-' * 60)
    #for key in sorted(results):
      #print('%s: %s' % (key, results[key]))
    #print(FLAGS.epochs_per_eval)
    record = record + 1
    #print("record = ", record)
  #Export Trained Model for Serving
  wideColumns, DeepColumns = build_model_columns()
  feature_columns = DeepColumns
  feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
  export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
  servable_model_dir = (r"c:\\_dev\\models\\official\\wide_deep_customer\\")
  servable_model_path = model.export_savedmodel(servable_model_dir, export_input_fn)
  print("Done Exporting at Path - %s", servable_model_path )

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_PREDICT_COLUMNS_DEFAULTS)
    features = dict(zip(_PREDICT_COLUMNS, columns))
    print("mb.....csv...", features)

    return features

def predict_input_fn(data_file):
    assert tf.gfile.Exists(data_file), ('%s not found. Please make sure the path is correct.' % data_file)

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.batch(1) # => This is very important to get the rank correct
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features
