import tensorflow as tf
import os
import numpy as np

exported_path = (r"c:\\_dev\\models\\official\\wide_deep_customer\\saved_model")
predictionoutputfile = (r"c:\\_dev\\models\\official\\wide_deep_customer\\result.csv")
predictioninputfile = 'input.txt'


def main():
	with tf.Session() as sess:
		# load the saved model
		tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)
		
		# get the predictor , refer tf.contrib.predictor
		predictor = tf.contrib.predictor.from_saved_model(exported_path)
		prediction_OutFile = open(predictionoutputfile, 'w')
		
		#Write Header for CSV file
		prediction_OutFile.write("size, customer, years_trading, years_customer, turn_over, bop, credit_rating, employees, corporate_family, country, 	market, period, abc, ly_spend, budget_spend, curr_period_fc, last_period, curr_period_ly, weather_fc, make_budget_lp, make_budget_ply, make_budget, probability")
		prediction_OutFile.write('\n')
		
		# Read file and create feature_dict for each record
		with open(predictioninputfile) as inf:
			# Skip header
			next(inf)
			for line in inf:

				# Read data, using python, into our features
				size, customer, years_trading, years_customer, turn_over, bop, credit_rating, employees, corporate_family, country, market, period, abc, ly_spend, budget_spend, curr_period_fc, last_period, curr_period_ly, weather_fc, make_budget_lp, make_budget_ply, make_budget = line.strip().split(",")


				# Create a feature_dict for train.example - Get Feature Columns using
				feature_dict = {
					'size': _bytes_feature(value=size.encode()),
					'customer':  _float_feature(value=int(customer)),
					'years_trading': _float_feature(value=int(years_trading)),
					'years_customer': _float_feature(value=int(years_customer)),
					'turn_over': _float_feature(value=int(turn_over)),
					'bop': _float_feature(value=int(bop)),
					'credit_rating': _bytes_feature(value=credit_rating.encode()),
					'employees': _float_feature(value=int(employees)),
					'corporate_family': _bytes_feature(value=corporate_family.encode()),
					'country': _bytes_feature(value=country.encode()),
					'market': _bytes_feature(value=market.encode()),
					'period': _float_feature(value=int(period)),
					'abc': _bytes_feature(value=abc.encode()),
					'ly_spend': _float_feature(value=int(ly_spend)),
					'budget_spend': _float_feature(value=int(budget_spend)),
					'curr_period_fc': _float_feature(value=int(curr_period_fc)),
					'last_period': _float_feature(value=int(last_period)),
					'curr_period_ly': _float_feature(value=int(curr_period_ly)),
					'weather_fc': _bytes_feature(value=weather_fc.encode()),
					'make_budget_lp': _bytes_feature(value=make_budget_lp.encode()),
					'make_budget_ply': _bytes_feature(make_budget_ply.encode()),
				}

				# Prepare model input
				
				model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))
				
				model_input = model_input.SerializeToString()
				output_dict = predictor({"inputs": [model_input]})
				
				print(" prediction Label is ", output_dict['classes'])
				print('Probability : ' + str(output_dict['scores']))

				# Positive label = 1
				prediction_OutFile.write(size + "," + str(customer) + "," + str(years_trading) + "," + str(years_customer) + "," + str(turn_over) + "," + str(bop) + "," + credit_rating + "," + str(employees) + "," + corporate_family + "," + country + "," + market + "," + period + "," + abc + "," + str(ly_spend) + "," + str(budget_spend) + "," + str(curr_period_fc) + "," + str(last_period) + "," + str(curr_period_ly) + "," + weather_fc + "," + make_budget_lp + "," + make_budget_ply + ",")
				label_index = np.argmax(output_dict['scores'])
				prediction_OutFile.write(str(label_index))
				prediction_OutFile.write(',')
				prediction_OutFile.write(str(output_dict['scores'][0][label_index]))
				prediction_OutFile.write('\n')

	prediction_OutFile.close()


def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == "__main__":
	main()
