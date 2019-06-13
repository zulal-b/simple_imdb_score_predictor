##
#	Author:	Zulal Bingol (21301083)
##

import sys
import csv
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsRegressor

#Parameters
total_features = 21
regulate = 1

#Domain flags
color_flag = 1
genre_flag = 9
plot_flag = 13
language_flag = 15
country_flag = 16
contentR_flag = 17
year_flag = 19

# Domains and Decoders
color_domain = []
genre_domain = []
plot_domain = []
language_domain = []
country_domain = []
contentR_domain = []
year_domain = []

color_dec = []
genre_dec = []
plot_dec = []
lan_dec = []
country_dec = []
contentR_dec = []
year_dec = []

# Global Lists
feature_space = []
categorical = []
extended_categorical = []


def addToDomain(feature, flag):
    tokens = feature.split('|')
    token_arr = np.array(tokens)
    
    if flag == color_flag:
        for i in range (0, token_arr.shape[0]):
            token_arr[i] = token_arr[i].rstrip()
            token_arr[i] = token_arr[i].lstrip() 
            if token_arr[i] not in color_domain:
                color_domain.append(token_arr[i])
    	
    elif flag == genre_flag:
        for i in range (0, token_arr.shape[0]):
            token_arr[i] = token_arr[i].rstrip()
            token_arr[i] = token_arr[i].lstrip()
            if token_arr[i] not in genre_domain:
                genre_domain.append(token_arr[i])

    elif flag == plot_flag:
        for i in range (0, token_arr.shape[0]):
            token_arr[i] = token_arr[i].rstrip()
            token_arr[i] = token_arr[i].lstrip()
            if token_arr[i] not in plot_domain:
                plot_domain.append(token_arr[i])

    elif flag == language_flag:
        for i in range (0, token_arr.shape[0]):
            token_arr[i] = token_arr[i].rstrip()
            token_arr[i] = token_arr[i].lstrip()
            if token_arr[i] not in language_domain:
                language_domain.append(token_arr[i])

    elif flag == country_flag:
        for i in range (0, token_arr.shape[0]):
            token_arr[i] = token_arr[i].rstrip()
            token_arr[i] = token_arr[i].lstrip()
            if token_arr[i] not in country_domain:
                country_domain.append(token_arr[i])

    elif flag == contentR_flag:
        for i in range (0, token_arr.shape[0]):
            token_arr[i] = token_arr[i].rstrip()
            token_arr[i] = token_arr[i].lstrip()
            if token_arr[i] not in contentR_domain:
                contentR_domain.append(token_arr[i])

    elif flag == year_flag:
        for i in range (0, token_arr.shape[0]):
            token_arr[i] = token_arr[i].rstrip()
            token_arr[i] = token_arr[i].lstrip()
            if token_arr[i] not in year_domain:
                year_domain.append(token_arr[i])

    else:
        print "addToDomain : SOMETHING IS VERY WRONG IN HERE"
        
    return token_arr

def correct_unextendedCat (sample_set, flag, decoder, domain):

	feature_values = []
	current_list = []
	for i in range(0, len(sample_set)):
		current_list = sample_set[i][flag]
		score_sum = 0
		if 'Unknown' in current_list and regulate == 1:
			score_sum = np.amax(decoder)
		else:
			for l_iter in range(0, len(current_list)):
				ind = domain.index(current_list[l_iter])
				score_sum = score_sum + decoder[ind]
		feature_values.append(score_sum)

	return	feature_values

def correct_numerical_unregulated (feature_values):

	feature_values = np.array(feature_values)
	feature_values = pd.DataFrame(feature_values)
	#print(feature_values.isnull().sum())
	feature_values.fillna(feature_values.mean(), inplace=True)
	feature_values = feature_values.as_matrix()

	return feature_values

def fill_missing_numerical(feature_values, true_scores, predictor):

	k = 0
	new_feature_values = []
	for i in range(0, len(feature_values)):
		temp_value = feature_values[i]
		if np.isnan(feature_values[i]) == True or np.isfinite(feature_values[i]) == False:
			temp_value = predictor.predict(true_scores[k])
			if k+1 < len(true_scores):
				k = k+1
		new_feature_values.append(temp_value)

	new_feature_values = np.array(new_feature_values).reshape(-1, 1)

	return new_feature_values

def calc_numerical_regulated (feature_values, true_scores):

	knn = KNeighborsRegressor()

	labels = []
	known_values = []
	u_indices = []

	for i in range(0, len(feature_values)):
		if np.isnan(feature_values[i]) == True or np.isfinite(feature_values[i]) == False:
			u_indices.append(i)
		else:
			labels.append(true_scores[i])
			known_values.append(feature_values[i])

	labels = np.array(labels).reshape(-1, 1)
	known_values = np.array(known_values).reshape(-1, 1)
	predictor = knn.fit(known_values, labels.ravel())
	
	new_feature_values = []
	for i in range(0, len(feature_values)):
		temp_value = feature_values[i]
		if i in u_indices:
			temp_value = predictor.predict(true_scores[i])
		new_feature_values.append(temp_value)

	new_feature_values = np.array(new_feature_values).reshape(-1, 1)

	return new_feature_values, predictor

def extendedDomainCalc_unregulated (domain, dataset, flag):
	decoder = np.ndarray(shape=(len(dataset), len(domain)), dtype=float, order='C')
	l = []
	for i in range(0, len(dataset)):
		temp = dataset[i][flag]
		l.append(temp[0])

	l = np.array(l)
	l = pd.get_dummies(l)

	k = 0
	for i in range(0, len(domain)):
		temp = l.as_matrix(columns=[domain[i]])
		if np.isnan(temp).any() == True or np.all(np.isfinite(temp)) == False:		# in unregulate mode, training nan values are treated as 0
			temp = np.zeros((len(dataset), 1))
		decoder[:, k] = temp[:, 0]
		k = k + 1

	return decoder 

def extendedDomainCalc_regulated (domain, dataset, tr_true_scores, flag):	# Finds the expected labels of Unknown values by naive bayesian classifier

	decoder = np.ndarray(shape=(len(dataset), len(domain) - 1), dtype=float, order='C')
	gnb = GaussianNB()
	labels = []
	scores = []
	u_indices = []

	for i in range(0, len(dataset)):
		temp_list = dataset[i][flag]
		if 'Unknown' not in temp_list:
			labels.append(temp_list[0])
			scores.append(tr_true_scores[i])
		else:
			u_indices.append(i)

	labels = np.array(labels).reshape(-1, 1)
	scores = np.array(scores).reshape(-1, 1)
	predictor = gnb.fit(scores, labels.ravel())
	
	l = []
	temp_list = []
	k = 0
	for i in range(0, len(dataset)):
		temp_list = dataset[i][flag]
		if i in u_indices:
			temp_list = predictor.predict(tr_true_scores[k])
			if k+1 < len(tr_true_scores):
				k = k+1
		l.append(temp_list[0])

	l = np.array(l)
	l = pd.get_dummies(l)

	k = 0
	for i in range(1, len(domain)):
		temp = []
		temp = l.as_matrix(columns=[domain[i]])
		if np.any(np.isnan(temp)) == True or np.all(np.isfinite(temp)) == False:
			temp = np.zeros((len(dataset), 1))
		decoder[:, k] = temp[:, 0]
		k = k + 1

	if np.any(np.isnan(decoder)) == True or np.all(np.isfinite(decoder)) == False:
		print "ERROR: decoder"
		print np.any(np.isnan(decoder)), np.all(np.isfinite(decoder))
		print np.sum(np.isnan(decoder)), np.sum(np.isfinite(decoder))
		print decoder.shape
		print decoder
		sys.exit()

	return decoder, predictor

def fill_missing_categorical(dataset, true_scores, predictor, flag, domain):

	decoder = np.ndarray(shape=(len(dataset), len(domain) - 1), dtype=float, order='C')

	l = []
	temp_list = []
	k = 0
	for i in range(0, len(dataset)):
		temp_list = dataset[i][flag]
		if 'Unknown' in temp_list:
			temp_list = predictor.predict(true_scores[k])
			if k+1 < len(true_scores):
				k = k+1
		l.append(temp_list[0])

	l = np.array(l)
	l = pd.get_dummies(l)

	k = 0
	for i in range(1, len(domain)):
		temp = []
		temp = l.as_matrix(columns=[domain[i]])
		if np.any(np.isnan(temp)) == True or np.all(np.isfinite(temp)) == False:
			temp = np.zeros((len(dataset), 1))
		decoder[:, k] = temp[:, 0]
		k = k + 1

	return decoder

def applyRegression_test(tr_feature_values, tr_true_scores, dataset_size, sample_size, selected_num_features, test_feature_values, true_scores_testing):
    
    regressor = DecisionTreeRegressor()
    regressor.fit(tr_feature_values, tr_true_scores)
    
    predicted_test = regressor.predict(test_feature_values) 
    for i in range(0, len(test_feature_values)):
        predicted_test[i] = round(predicted_test[i], 1)
    score = mean_squared_error(true_scores_testing, predicted_test)
    
    return score

def applyRegression_train(tr_feature_values, tr_true_scores, dataset_size, sample_size, selected_num_features):
    
    regressor = DecisionTreeRegressor()
    regressor.fit(tr_feature_values, tr_true_scores)
    
    predicted_tr = regressor.predict(tr_feature_values) 
    for i in range(0, len(tr_feature_values)):
        predicted_tr[i] = round(predicted_tr[i], 1)
    score_tr = mean_squared_error(tr_true_scores, predicted_tr)
    
    return score_tr, regressor


# Processing categorical features
def domainAnalysis(train_set, tr_true_scores, cat_size):

	# COLOR DOMAIN
	global color_dec
	color_dec = [0 for i in range (0, len(color_domain))]
	for i in range(0, len(train_set)):
		c = train_set[i][color_flag]
		temp_ind = color_domain.index(c[0])
		color_dec[temp_ind] = color_dec[temp_ind] + tr_true_scores[i]
	color_dec = [(color_dec[i] / len(train_set)) for i in range (0, len(color_dec))] 

	# GENRE DOMAIN
	global genre_dec
	for i in range(0, len(genre_domain)):
		genre_dec.append(0)

	for i in range(0, len(train_set)):
		genre_size = len(train_set[i][genre_flag])
		imdb_s = tr_true_scores[i]
		current_genre_list = train_set[i][genre_flag]
		g_weight = imdb_s / genre_size
		if 'Unknown' not in current_genre_list:
			for j in range(0, genre_size):
				ind = genre_domain.index(current_genre_list[j])
				genre_dec[ind] = genre_dec[ind] + g_weight
		else:
			ind = genre_domain.index('Unknown')
			genre_dec[ind] = genre_dec[ind] + imdb_s   # POSSIBLE IMPROVEMENT

    #PLOT DOMAIN
	global plot_dec
	for i in range(0, len(plot_domain)):
		plot_dec.append(0)

	for i in range(0, len(train_set)):
		plot_size = len(train_set[i][plot_flag])
		imdb_s = tr_true_scores[i]
		current_plot_list = train_set[i][plot_flag]
		c_weight = imdb_s / plot_size
		if 'Unknown' not in current_plot_list:
			for j in range(0, plot_size):
				ind = plot_domain.index(current_plot_list[j])
				plot_dec[ind] = plot_dec[ind] + c_weight
		else:
			ind = plot_domain.index('Unknown')
			plot_dec[ind] = plot_dec[ind] + imdb_s   # POSSIBLE IMPROVEMENT
    
    #LANGUAGE DOMAIN
	global lan_dec
	if regulate == 1:
		lan_dec, lan_predictor = extendedDomainCalc_regulated (language_domain, train_set, tr_true_scores, language_flag)
	else:
		lan_dec = extendedDomainCalc_unregulated (language_domain, train_set, language_flag)

	# COUNTRY DOMAIN
	global country_dec
	if regulate == 1:
		country_dec, country_predictor = extendedDomainCalc_regulated(country_domain, train_set, tr_true_scores, country_flag)
	else:
		country_dec = extendedDomainCalc_unregulated (country_domain, train_set, country_flag)
	

    #CONTENT_R DOMAIN
	global contentR_dec
	if regulate == 1:
		contentR_dec, contentR_predictor = extendedDomainCalc_regulated(contentR_domain, train_set, tr_true_scores, contentR_flag)
	else:
		contentR_dec = extendedDomainCalc_unregulated (contentR_domain, train_set, contentR_flag)
    
    #YEAR DOMAIN:
	global year_dec
	if regulate == 1:
		year_dec, year_predictor = extendedDomainCalc_regulated(year_domain, train_set, tr_true_scores, year_flag)
	else:
		year_dec = extendedDomainCalc_unregulated (year_domain, train_set, year_flag)

	return


def train_model(sample_set, true_scores, dataset_size, sample_size, feature_size):

	extended_categorical.extend([language_flag, year_flag, contentR_flag, country_flag])
	categorical.extend([color_flag, genre_flag, plot_flag, language_flag, country_flag, contentR_flag, year_flag]) # indices of categorical features

	domainAnalysis(sample_set, true_scores, len(categorical)) # processes categorical features 

	regressors = []
	selected_features = []
	stepped_errors = []
	extended_size = 0
	whole_features = np.ndarray(shape=(sample_size, feature_size), dtype=float, order='C')
	whole_size = 0
	
	notyet = [] 
   	selected_num_features = 0
	for f_iter in range(0, feature_size):
		#print "ITER = ", f_iter
		error_list = []
		indices = []
		temp_regressors = []

		for j in range(0, total_features):
			temp_extension = 0
			if j not in selected_features and j not in categorical:
				temp_extension = 1
				feature_values = []
				#print "current FEATURE ", j
				feature_values = [sample_set[i][j] for i in range(0, sample_size)]

				if regulate == 1:
					feature_values, knn_predictor = calc_numerical_regulated (feature_values, true_scores)
				else:
					feature_values = correct_numerical_unregulated (feature_values)

				if (f_iter == 0):
					temp_whole_features = feature_values	
				else:
					temp_whole_features = np.ndarray(shape=(sample_size, whole_size + 1), dtype=float, order='C')
					for t in range(0, whole_size + 1):
						if t != whole_size:
							temp_whole_features[:, t] = whole_features[:, t]
						else:
							temp_whole_features[:, t] = feature_values[:, 0]

				if np.any(np.isnan(temp_whole_features)) == True or np.all(np.isfinite(temp_whole_features)) == False:
					print "ERROR: numerical", j
					print np.sum(np.isnan(temp_whole_features)), np.sum(np.isfinite(temp_whole_features))
					print temp_whole_features
					sys.exit()

				loop_error , loop_regressor = applyRegression_train(temp_whole_features, true_scores, dataset_size, sample_size, whole_size + 1)
				indices.append(j)
				error_list.append(loop_error)
				temp_regressors.append(loop_regressor)

			elif j not in selected_features and j in categorical and j not in notyet:

				if j not in extended_categorical:
					feature_values = []
					if j == color_flag:
						for i in range (0, sample_size):
							temp_color = sample_set[i][color_flag]
							if 'Unknown' in temp_color and regulate == 1:
								max_index = color_dec.index(np.amax(color_dec))
								temp_index = max_index
							else:
								temp_index = color_domain.index(temp_color[0])
							feature_values.append(color_dec[temp_index])

					elif j == plot_flag:
						feature_values = correct_unextendedCat(sample_set, plot_flag, plot_dec, plot_domain)

					elif j == genre_flag:
						feature_values = correct_unextendedCat(sample_set, genre_flag, genre_dec, genre_domain)

					temp_extension = 1
					feature_values = np.array(feature_values).reshape(sample_size, 1)
				
					if (f_iter == 0):
						temp_whole_features = feature_values
					else:
						temp_whole_features = np.ndarray(shape=(sample_size, whole_size + 1), dtype=float, order='C')
						for t in range(0, whole_size + 1):
							if t != whole_size:
								temp_whole_features[:, t] = whole_features[:, t]
							else:
								temp_whole_features[:, t] = feature_values[:, 0]
				
				else:
					if j == language_flag:
						if regulate == 1:
							temp_extension = len(language_domain) - 1
							feature_values, predictor = extendedDomainCalc_regulated (language_domain, sample_set, true_scores, language_flag)
						else:
							temp_extension = len(language_domain)
							feature_values = extendedDomainCalc_unregulated (language_domain, sample_set, language_flag)

					elif j == year_flag:
						if regulate == 1:
							temp_extension = len(year_domain) - 1
							feature_values, predictor = extendedDomainCalc_regulated (year_domain, sample_set, true_scores, year_flag)
						else:
							temp_extension = len(year_domain)
							feature_values = extendedDomainCalc_unregulated (year_domain, sample_set, year_flag)

					elif j == contentR_flag:
						if regulate == 1:
							temp_extension = len(contentR_domain) - 1
							feature_values, predictor = extendedDomainCalc_regulated (contentR_domain, sample_set, true_scores, contentR_flag)
						else:
							temp_extension = len(contentR_domain)
							feature_values = extendedDomainCalc_unregulated (contentR_domain, sample_set, contentR_flag)

					elif j == country_flag:
						if regulate == 1:
							temp_extension = len(country_domain) - 1
							feature_values, predictor = extendedDomainCalc_regulated (country_domain, sample_set, true_scores, country_flag)
						else:
							temp_extension = len(country_domain)
							feature_values = extendedDomainCalc_unregulated (country_domain, sample_set, country_flag)

					else:
						print "CONTROL"


					if (f_iter == 0):
						temp_whole_features = feature_values
					else:
						temp_whole_features = np.ndarray(shape=(sample_size, whole_size + temp_extension), dtype=float, order='C')
						k = 0
						for t in range(0, whole_size + temp_extension):
							if t < whole_size:
								temp_whole_features[:, t] = whole_features[:, t]
							else:
								temp_whole_features[:, t] = feature_values[:, k]
								k = k + 1

				if np.any(np.isnan(temp_whole_features)) == True or np.all(np.isfinite(temp_whole_features)) == False:
					print "ERROR: categorical", j
					print np.any(np.isnan(temp_whole_features)), np.all(np.isfinite(temp_whole_features))
					print temp_whole_features
					sys.exit()

				loop_error , loop_regressor = applyRegression_train(temp_whole_features, true_scores, dataset_size, sample_size, whole_size + 1)
				indices.append(j)
				error_list.append(loop_error)
				temp_regressors.append(loop_regressor)

		min_error = min(error_list)
		min_index = error_list.index(min_error)
		selected_feature_index = indices[min_index] # gives j
		#print "selected_feature_index", selected_feature_index
		selected_features.append(selected_feature_index)
		stepped_errors.append(min_error)

		if selected_feature_index not in categorical:
			selected_feature_values = [sample_set[i][selected_feature_index] for i in range(0, sample_size)]
			
			if regulate == 1:
				selected_feature_values, predictor = calc_numerical_regulated (selected_feature_values, true_scores)
			else:
				selected_feature_values = correct_numerical_unregulated (selected_feature_values)
		
		else:
			selected_feature_values = []
			if selected_feature_index == color_flag:
				for i in range (0, sample_size):
					temp_color = sample_set[i][color_flag]
					if 'Unknown' in temp_color and regulate == 1:
						temp_index = color_dec.index(np.amax(color_dec))
					else:
						temp_index = color_domain.index(temp_color[0])
					selected_feature_values.append(color_dec[temp_index])
				selected_feature_values = np.array(selected_feature_values).reshape(-1, 1)

			elif selected_feature_index == plot_flag:
				selected_feature_values = correct_unextendedCat(sample_set, plot_flag, plot_dec, plot_domain)
				selected_feature_values = np.array(selected_feature_values).reshape(-1, 1)

			elif selected_feature_index == language_flag:
				if regulate == 1:
					extended_size = len(language_domain)-1
				else:
					extended_size = len(language_domain)
				selected_feature_values = lan_dec

			elif selected_feature_index == genre_flag:
				selected_feature_values = correct_unextendedCat(sample_set, genre_flag, genre_dec, genre_domain)
				selected_feature_values = np.array(selected_feature_values).reshape(-1, 1)

			elif selected_feature_index == year_flag:
				if regulate == 1:
					extended_size = len(year_domain)-1
				else:
					extended_size = len(year_domain)
				selected_feature_values = year_dec

			elif selected_feature_index == contentR_flag:
				if regulate == 1:
					extended_size = len(contentR_domain)-1
				else:
					extended_size = len(contentR_domain)
				selected_feature_values = contentR_dec

			elif selected_feature_index == country_flag:
				if regulate == 1:
					extended_size = len(country_domain)-1
				else:
					extended_size = len(country_domain)
				selected_feature_values = country_dec

			else:
				print "CONTROL"

		# Constructing whole feature set after adding the selected feature
		if selected_feature_index in extended_categorical:
			whole_size = whole_size + extended_size
			temp_whole = np.ndarray(shape=(sample_size, whole_size), dtype=float, order='C')
			
			if f_iter == 0:
				for i in range(0, selected_feature_values.shape[1]):
					temp_whole[:, i] = selected_feature_values[:, i]
			else:
				for i in range (0, whole_features.shape[1]):
					temp_whole[:, i] = whole_features[:, i]
				k = 0
				for i in range (whole_features.shape[1], whole_size):
					temp_whole[:, i] = selected_feature_values[:, k]
					k = k + 1
			
			whole_features = np.ndarray(shape=(sample_size, whole_size), dtype=float, order='C')
			whole_features = temp_whole
		
		else:
			whole_size = whole_size + 1
			temp_whole = np.ndarray(shape=(sample_size, whole_size), dtype=float, order='C')

			if f_iter == 0:
				temp_whole[:, 0] = selected_feature_values[:, 0]
			else:
				for i in range (0, whole_features.shape[1]):
					temp_whole[:, i] = whole_features[:, i]
				
				for i in range(0, sample_size):
					temp_whole[i][whole_size-1] = selected_feature_values[i]
			
			whole_features = np.ndarray(shape=(sample_size, whole_size), dtype=float, order='C')
			whole_features = temp_whole

		if np.any(np.isnan(whole_features)) == True or np.all(np.isfinite(whole_features)) == False:
			print "ERROR selected_feature_index", selected_feature_index
			print np.any(np.isnan(whole_features)), np.all(np.isfinite(whole_features))
			print np.sum(np.isnan(whole_features)), np.sum(np.isfinite(whole_features) == False)
			print whole_features
			print whole_features.shape
			sys.exit()

		selected_num_features = f_iter + 1

	feature_names = [feature_space[selected_features[i]] for i in range(0, feature_size)]
	#export_graphviz(loop_regressor, out_file='tree.dot', feature_names=feature_names)

	print "FINAL", selected_features
	print stepped_errors

	return selected_features, whole_features, loop_regressor, stepped_errors


def main():

    dataset_size = 0
    true_imdb_scores = [] 
    all_data = []
    feature_space.extend(['title', 'color', 'director', 'actor_1', 'actor_2', 'actor_3', 'num_critics_for_reviews', 'duration', 'gross', 'genres', 'num_voted_users', 'cast_total_fb_likes', 'face_num_poster', 'plot_keywords', 'num_users_for_reviews', 'language', 'country', 'content_rating', 'budget', 'year', 'aspect_ratio'])
    #movie = namedtuple('movie', 'm_title m_color m_director m_actor_1 m_actor_2 m_actor_3 m_num_critics_for_reviews m_duration m_gross m_genres m_num_voted_users m_cast_total_fb_likes m_face_num_poster m_plot_keywords m_num_users_for_reviews m_language m_country m_content_rating m_budget m_year m_aspect_ratio')
    
    genre_domain.append('Unknown')
    color_domain.append('Unknown')
    language_domain.append('Unknown')
    country_domain.append('Unknown')
    contentR_domain.append('Unknown')
    year_domain.append('Unknown')
    plot_domain.append('Unknown')

    training_scores = []
    test_scores = []

    #Reading the dataset, adjusting values and forming the input sample
    with open('movie_metadata.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if dataset_size != 0:
                m_title = float('NaN') if row[27] == '' else float(row[27])
                m_color = ['Unknown'] if row[0] == '' else addToDomain(row[0], color_flag)
                m_director = float('NaN') if row[4] == '' else float(row[4])
                m_actor_1 = float('NaN') if row[7] == '' else float(row[7])
                m_actor_2 = float('NaN') if row[24] == '' else float(row[24])
                m_actor_3 = float('NaN') if row[5] == '' else float(row[5])
                #print row[11]
                m_num_critics_for_reviews = float('NaN') if row[2] == '' else float(row[2])
                m_duration = float('NaN') if row[3] == '' else float(row[3])
                m_gross = float('NaN') if row[8] == '' else float(row[8])
                m_genres = ['Unknown'] if row[9] == '' else addToDomain(row[9], genre_flag)
                m_num_voted_users = float('NaN') if row[12] == '' else float(row[12])
                m_cast_total_fb_likes = float('NaN') if row[13] == '' else float(row[13])
                m_face_num_poster = float('NaN') if row[15] == '' else float(row[15])
                m_plot_keywords = ['Unknown'] if row[16] == '' else addToDomain(row[16], plot_flag)
                m_num_users_for_reviews = float('NaN') if row[18] == '' else float(row[18])
                m_language = ['Unknown'] if row[19] == '' else addToDomain(row[19], language_flag)
                m_country = ['Unknown'] if row[20] == '' else addToDomain(row[20], country_flag)
                m_content_rating = ['Unknown'] if row[21] == '' else addToDomain(row[21], contentR_flag)
                m_budget = float('NaN') if row[22] == '' else float(row[22])
                m_year = ['Unknown'] if row[23] == '' else addToDomain(row[23], year_flag)
                m_aspect_ratio = float('NaN') if row[26] == '' else float(row[26])
                m_imdb_score = float('NaN') if row[25] == '' else float(row[25])
                #temp_movie = movie(m_title, m_color, m_director, m_actor_1, m_actor_2, m_actor_3, m_num_critics_for_reviews, m_duration, m_gross, m_genres, m_num_voted_users, m_cast_total_fb_likes, m_face_num_poster, m_plot_keywords, m_num_users_for_reviews, m_language, m_country, m_content_rating, m_budget, m_year, m_aspect_ratio)
                temp_movie = tuple((m_title, m_color, m_director, m_actor_1, m_actor_2, m_actor_3, m_num_critics_for_reviews, m_duration, m_gross, m_genres, m_num_voted_users, m_cast_total_fb_likes, m_face_num_poster, m_plot_keywords, m_num_users_for_reviews, m_language, m_country, m_content_rating, m_budget, m_year, m_aspect_ratio))
                all_data.append(temp_movie)
                true_imdb_scores.append(m_imdb_score)
            dataset_size = dataset_size + 1
        
        # Determining the training set and testing set
        dataset_size = dataset_size - 1
        sample_size = dataset_size / 2 
        tr_ind = random.sample(range(0, dataset_size), sample_size)
        training_sample = []
        testing_sample = []
        true_scores_training = []
        true_scores_testing = []
        for i in range (0, dataset_size):
            if i in tr_ind:
                training_sample.append(all_data[i])
                true_scores_training.append(true_imdb_scores[i])
            else:
                testing_sample.append(all_data[i])
                true_scores_testing.append(true_imdb_scores[i])
        
        for feature_size in range (1, total_features+1):
	        # Training
			selected_features, trained, lastRegressor, stepped_errors = train_model(training_sample, true_scores_training, dataset_size, sample_size, feature_size)
			next_addition_index = 0

	        # Processing test set
			test_features = np.ndarray(shape=(sample_size+1, trained.shape[1]), dtype=float, order='C')
			for j in range(0 , feature_size):
				feature = selected_features[j]

				if feature not in categorical:
					#print "FEATURE ", feature
					selected_feature_values = [testing_sample[i][feature] for i in range(0, sample_size+1)]

					if regulate == 1:
						dummy_features = [training_sample[i][feature] for i in range(0, sample_size)]
						dummy_features, knn_predictor = calc_numerical_regulated (dummy_features, true_scores_training)
						selected_feature_values = fill_missing_numerical(selected_feature_values, true_scores_training, knn_predictor)
					else:
						selected_feature_values = correct_numerical_unregulated (selected_feature_values)

					test_features[:, next_addition_index] = selected_feature_values[:, 0]
					next_addition_index = next_addition_index + 1

				elif feature in categorical and feature not in extended_categorical:
					#print "feature ", feature
					if feature == color_flag:
						selected_feature_values = []
						for i in range (0, sample_size+1):
							temp_color = testing_sample[i][color_flag]
							if 'Unknown' in temp_color and regulate == 1:
								max_index = color_dec.index(np.amax(color_dec))
								temp_index = max_index
							else:
								temp_index = color_domain.index(temp_color[0])
							selected_feature_values.append(color_dec[temp_index])
					
					elif feature == plot_flag:
						selected_feature_values = correct_unextendedCat(testing_sample, plot_flag, plot_dec, plot_domain)

					elif feature == genre_flag:
						selected_feature_values = correct_unextendedCat(testing_sample, genre_flag, genre_dec, genre_domain)
						
					selected_feature_values = np.array(selected_feature_values).reshape(-1, 1)
					test_features[:, next_addition_index] = selected_feature_values[:, 0]
					next_addition_index = next_addition_index + 1

				elif feature in extended_categorical:
					#print "feature_exd ", feature
					if feature == language_flag:
						if regulate == 1:
							dummy_features, lan_predictor = extendedDomainCalc_regulated (language_domain, training_sample, true_scores_training, language_flag)
							selected_feature_values = fill_missing_categorical(testing_sample, true_scores_training, lan_predictor, language_flag, language_domain)
						else:
							selected_feature_values = extendedDomainCalc_unregulated (language_domain, testing_sample, language_flag)

					elif feature == year_flag:
						if regulate == 1:
							dummy_features, year_predictor = extendedDomainCalc_regulated (year_domain, training_sample, true_scores_training, year_flag)
							selected_feature_values = fill_missing_categorical(testing_sample, true_scores_training, year_predictor, year_flag, year_domain)
						else:
							selected_feature_values = extendedDomainCalc_unregulated (year_domain, testing_sample, year_flag)

					elif feature == contentR_flag:
						if regulate == 1:
							dummy_features, contentR_predictor = extendedDomainCalc_regulated (contentR_domain, training_sample, true_scores_training, contentR_flag)
							selected_feature_values = fill_missing_categorical(testing_sample, true_scores_training, contentR_predictor, contentR_flag, contentR_domain)
						else:
							selected_feature_values = extendedDomainCalc_unregulated (contentR_domain, testing_sample, contentR_flag)

					elif feature == country_flag:
						if regulate == 1:
							dummy_features, country_predictor = extendedDomainCalc_regulated (country_domain, training_sample, true_scores_training, country_flag)
							selected_feature_values = fill_missing_categorical(testing_sample, true_scores_training, country_predictor, country_flag, country_domain)
						else:
							selected_feature_values = extendedDomainCalc_unregulated (country_domain, testing_sample, country_flag)

					for i in range (0, selected_feature_values.shape[1]):
						test_features[:, next_addition_index] = selected_feature_values[:, i]
						next_addition_index = next_addition_index + 1

				else:
					print "CONTROL"

			print trained.shape
			print test_features.shape
			if np.any(np.isnan(test_features)) == True or np.all(np.isfinite(test_features)) == False:
				print "ERROR: NaN in testing"
				print np.sum(np.isnan(test_features)), np.sum(np.isfinite(test_features))
				print test_features
				sys.exit()
			if np.any(np.isnan(trained)) == True or np.all(np.isfinite(trained)) == False:
				print "ERROR: NaN in trained"
				print np.sum(np.isnan(trained)), np.sum(np.isfinite(trained))
				print trained
				sys.exit()

			test_score = applyRegression_test(trained, true_scores_training, dataset_size, sample_size, feature_size, test_features, true_scores_testing)
			print "Features: ", feature_size, "TEST_SCORE ", test_score
			test_scores.append(test_score)
			training_scores = stepped_errors
			print "TEST_SCORE", test_scores
			print "Final Tree Depth", lastRegressor.tree_.max_depth
			print "******************************************************"

	x_axis = [i for i in range (1, total_features+1)]
	plt.subplot(2, 1, 1)
	plt.plot(x_axis, training_scores, 'o-')
	plt.title('Mean Squared Errors: Training vs Testing')
	plt.ylabel('Training')

	plt.subplot(2, 1, 2)
	plt.plot(x_axis, test_scores, '.-')
	plt.xlabel('Number of Features')
	plt.ylabel('Testing')
 
	plt.savefig("reg1_noMaxL")
	plt.show()

	return


if __name__ == "__main__":
    main()


