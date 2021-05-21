# Import modules
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold, GridSearchCV, cross_validate

from transformers import BertModel, BertConfig, BertTokenizer
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import ElectraModel, ElectraConfig, ElectraTokenizer
from transformers import XLNetModel, XLNetTokenizer
import torch

import re
import joblib



# Define class
class ModelExperiment(object):
	def __init__(self, embedding_path: str = './data/latent_var_embeddings', embedding_mode: str = 'latent_var', model_name: str = 'ICA', data_file: str = 'data/COVID19_Dataset-CM-ZB-complete with sources.xls', append_test_fold: bool = False, kernel : str = 'linear', inner_cv_nsplits = 5, outer_cv_nsplits = 10, RANDOM_STATE = 42):#latex: bool = True, RANDOM_STATE = 42):
		self.RANDOM_STATE = RANDOM_STATE
		self.embedding_path = embedding_path # either './data/transformer_embeddings' or './data/latent_var_embeddings'
		self.embedding_mode = embedding_mode # either 'latent_var' or 'transformer'
		self.model_name = model_name # one of transformer models, or 'ICA', 'IVA-mean', 'IVA-max', 'LDA', 'DL', or 'NMF'
		self.embeddings = None
		
		# only used if embedding_mode=='transformer'
		self.tokenizer_module = None
		self.model_module = None
		if re.match('^bert', model_name):
			self.tokenizer_module = BertTokenizer
			self.model_module = BertModel
		elif re.match('^roberta', model_name):
			self.tokenizer_module = RobertaTokenizer
			self.model_module = RobertaModel
		elif re.match('^google/electra', model_name):
			self.tokenizer_module = ElectraTokenizer
			self.model_module = ElectraModel
		self.model_definition = { # NOTE: if embedding_mode == 'latent_var', the tokenizer & model module values will be None
		"tokenizer_module": self.tokenizer_module,
		"model_module": self.model_module,
		"model_name": model_name
		}

		# Load data
		self.data_file = data_file
		self.raw_df = pd.read_excel(data_file)
		self.labels = self.raw_df["Is_Unreliable"].to_numpy()

		# For experiment
		self.outer_cv_nsplits = outer_cv_nsplits
		self.inner_cv_nsplits = inner_cv_nsplits
		self.append_test_fold = append_test_fold # True or False: whether to add a column to the data file that labels the rows with their test fold
		self.kernel = kernel # one of 'rbf', 'linear', 'poly', or 'sigmoid'
		self.svr = None
		self.param_grid = None
		self.inner_cv = None
		self.outer_cv = None
		self.clf = None # GridSearchCV fitted clf
		self.estimators = None # list of models by fold
		# self.result_df = pd.DataFrame()
		self.nested_scores = None
		# self.latex = latex

	def main(self):
		# load or create embeddings (saved as class attribute)
		self.get_embeddings()

		# run experiment & save results as class attribute
		self.experiment(self.embeddings, self.labels)

		# save trained models
		self.save_estimators()

		# # save results to csv file
		# results_df = self.get_results_df(self.nested_scores)
		# results_df.to_csv(f'{self.model_name}_{self.kernel}_results.csv')

		# # if selected, print LaTeX formatted results
		# if self.latex:
		# 	latex_results = self.get_results_latex(self.nested_scores)
		# 	print(latex_results)

		# if selected, save csv of original data w new column marking test fold for each row
		if self.append_test_fold:
			self.mark_test_fold(self.embeddings)

	# Get (or create) word embeddings
	def get_embeddings(self):
		if self.embedding_mode == 'transformer':
			shortname = self.model_definition['model_name'].split("/")[-1]
			output_path = self.embedding_path
			output_file = f"{output_path}/{shortname}.npy"
			Path(output_path).mkdir(parents=True, exist_ok=True)
			
			if Path(output_file).is_file():
				print(f"Embeddings {output_file} already exist. Skipping.")
				self.embeddings = np.load(output_file)
			else:
				self.embeddings = self.generate_transformer_embeddings(self.raw_df["Tweet"])
				np.save(output_file, self.embeddings)
		elif self.embedding_mode == 'latent_var':
			try:
				self.embeddings = np.load(f'{self.embedding_path}/tweet_embed_{self.model_name}.npy')
			except:
				print(f'The file {self.embedding_path}/tweet_embed_{self.model_name} does not exist.')

	# Generate transformer embeddings
	def generate_transformer_embeddings(self, text_list):
		tokenizer = self.model_definition["tokenizer_module"].from_pretrained(self.model_definition["model_name"])
		model = self.model_definition["model_module"].from_pretrained(self.model_definition["model_name"])
		
		embeddings = []
		for text in text_list:
			inputs = tokenizer(text, return_tensors="pt")
			outputs = model(**inputs)
			last_hidden_states = outputs.last_hidden_state[0]
			mean = torch.mean(last_hidden_states, 0)
			embeddings.append(mean.detach().numpy())
		
		return np.array(embeddings)

	# Define experiment function
	def experiment(self, X, y):
		self.svr = svm.SVC(kernel=self.kernel)
		self.param_grid = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001]}

		self.inner_cv = KFold(n_splits=self.inner_cv_nsplits, shuffle=True, random_state=self.RANDOM_STATE)
		self.outer_cv = KFold(n_splits=self.outer_cv_nsplits, shuffle=True, random_state=self.RANDOM_STATE)

		clf = GridSearchCV(estimator=self.svr, param_grid=self.param_grid, cv=self.inner_cv, refit=True)
		self.nested_scores = cross_validate(clf, X=X, y=y, cv=self.outer_cv, scoring=['accuracy', 'precision', 'recall', 'f1'], n_jobs=-1, return_estimator=True)
		self.clf = clf

		#return self.nested_scores.copy() # dictionary, where each entry is a fold's results

	# # Get results as pandas data frame
	# def get_results_df(self, scores):
	# 	result = dict()
	# 	result['Method'] = self.model_name
	# 	for score_name, score_array in scores.items():
	# 		if score_name != 'estimator':
	# 			result[score_name] = np.mean(score_array)
				
	# 	self.result_df = self.result_df.append(result, ignore_index=True)

	# 	return self.result_df.copy()

	# # Format results as LaTeX
	# def get_results_latex(self, scores):
	# 	result_df = self.get_results_df(scores)
	# 	formatted_df = result_df.copy().round(3)
	# 	formatted_df = formatted_df.rename(columns={"test_accuracy": "Accuracy", "test_f1": "F1", "test_precision": "Precision", "test_recall": "Recall"})

	# 	#formatted_df["Method"] = formatted_df["File"].map(file_method_mapping)
	# 	formatted_df = formatted_df[["Method", "Accuracy", "Recall", "Precision", "F1"]]
	# 	formatted_df = formatted_df.set_index("Method")

	# 	self.result_latex = formatted_df.to_latex()

	# 	return self.result_latex

	# Add column noting test fold number for each row in dataset (outer CV)
	def mark_test_fold(self, X):
		fold_labels = np.zeros(self.labels.shape[0]) # 1-dimensional array
		test_folds = [test_idx for train_idx, test_idx in self.outer_cv.split(X)]

		for fold_num, test_set in enumerate(test_folds, start=1): # folds 1-10 instead of 0-9
			fold_labels[test_set] = fold_num

		vals = fold_labels.tolist()
		self.raw_df['Test_Fold'] = vals
		fname = self.data_file.split('.')[0] + '_wTestFold.csv'
		self.raw_df.to_csv(fname)

	# Save the fitted estimators for each fold in outer CV
	def save_estimators(self):
		self.estimators = self.nested_scores['estimator']
		clean_name = self.model_name.split('/')[-1]
		for i, est in enumerate(self.estimators, start=1): # folds 1-10 instead of 0-9
			joblib.dump(est, f'./fold_estimators/SVM-{self.kernel}_{clean_name}_Fold{i}.joblib')


if __name__ == '__main__':
	method_name_mapping = {
		'electra-base-discriminator': "ELECTRA-Base",
		'roberta-base': "RoBERTa-Base",
		'electra-large-discriminator': "ELECTRA-Large",
		'bert-base-cased': "BERT-Base-Cased",
		'roberta-large': "RoBERTa-Large",
		'bert-base-uncased': "BERT-Base-Uncased",
		'bert-large-cased': "BERT-Large-Cased",
		'bert-large-uncased': "BERT-Large-Uncased",
		'NMF': 'NMF',
		'LDA': 'LDA',
		'DL': 'DL',
		'ICA': 'ICA',
		'IVA-mean': 'IVA-mean',
		'IVA-max': 'IVA-max'
		}

	# Get results as pandas data frame
	def get_results_df(model_scores):
		results_df = pd.DataFrame()
		for mname, scores in model_scores.items():
			result = dict()
			clean_name = mname.split('/')[-1]
			result['ModelName'] = clean_name
			
			for score_name, score_array in scores.items():
				if score_name != 'estimator':
					result[score_name] = np.mean(score_array)
				
			results_df = results_df.append(result, ignore_index=True)

		return results_df.copy()

	# Format results as LaTeX
	def get_results_latex(results_df):
		formatted_df = results_df.copy().round(3)
		formatted_df = formatted_df.rename(columns={"test_accuracy": "Accuracy", "test_f1": "F1", "test_precision": "Precision", "test_recall": "Recall"})

		formatted_df["Method"] = formatted_df["ModelName"].map(method_name_mapping)
		formatted_df = formatted_df[["Method", "Accuracy", "Recall", "Precision", "F1"]]
		formatted_df = formatted_df.set_index("Method")

		result_latex = formatted_df.to_latex()

		return result_latex

	models = ['ICA', 'NMF', 'DL', 'IVA-mean', 'IVA-max', 'LDA', 'bert-large-uncased', 'bert-large-cased', 'bert-base-uncased', 'bert-base-cased', 'roberta-base', 'roberta-large', 'google/electra-base-discriminator', 'google/electra-large-discriminator']
	kernels = ['rbf', 'linear', 'poly', 'sigmoid']
	for kernel in kernels:
		complete_scores = dict()
		for mname in models:
			if mname in models[:6]:
				embedding_mode = 'latent_var'
				embedding_path = './data/latent_var_embeddings'
			else:
				embedding_mode = 'transformer'
				embedding_path = './data/transformer_embeddings'

			exp = ModelExperiment(kernel=kernel, model_name=mname, embedding_mode=embedding_mode, embedding_path=embedding_path)
			exp.main()
			complete_scores[mname] = exp.nested_scores

		results = get_results_df(complete_scores)
		results.to_csv(f'SVM-{kernel}_results.csv')
		latex_results = get_results_latex(results)
		print(latex_results)
		with open(f'SVM-{kernel}_results_latex.txt', 'wt') as f:
			f.write(latex_results)

