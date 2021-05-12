# Create word embeddings using other code files

# Import modules
import numpy as np
from contextmatrix import ContextMatrix, get_text_vectors
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import DictionaryLearning
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from datetime import datetime

print('Finished importing modules.\n')
start_time = datetime.now()


# Set random state
rs = 714

# Load data
# train = pd.read_csv('./data/competition_data/TrainLabels.csv').drop('id', axis = 1)
# validate = pd.read_csv('./data/competition_data/validation.csv').drop('id', axis = 1)
# test = pd.read_csv('./data/competition_data/english_test_with_labels - Sheet1.csv').drop('id', axis = 1)
# print('Loaded tweets.\n')

# tweets = pd.concat([train, validate, test], ignore_index = True)
# print('There are {} tweets.\n'.format(tweets.shape[0]))


# # Create word-context matrix
# wcm_t0 = datetime.now()
# cm = ContextMatrix(window_size = 15,
#                    lowercase = True,
#                    lemmatize = True,
#                    pmi = True,
#                    laplace_smoothing = 2) # shifted by 2
# print('Instantiated ContextMatrix class.\n')

# # Fit vocabulary using full set of tweets & output word-context matrix
# wcm = cm.fit_transform(tweets['tweet'])
# print(f'Completed fit_transform method in {(datetime.now() - wcm_t0).total_seconds()} seconds.\n')
# np.save('wcm.npy', wcm)
# print('Created word-word co-occurrence matrix of shape {}.\n'.format(wcm.shape))

# # Check for NaN's
# if not np.isnan(wcm).any():
#     print('There are no NaN values in the word-context matrix.\n')

wcm = np.load('wcm.npy')
print('Loaded word-context array.')

# Standard scaling of word context matrix (DL, ICA)
#scaler = StandardScaler()
# scale word context matrix to be non-negative (NMF, LDA)
scaler = StandardScaler()
#X_std = scaler.fit_transform(wcm)
#print('Standardized word-context matrix.\n')


# Get word embeddings

# Instantiate matrix factorization class
mf = DictionaryLearning(n_components=250, random_state=rs)
#embeddings = mf.fit_transform(X_std)

print('Instantiated scaler & matrix factorization algo.')

pipe = Pipeline(steps=[('scaler', scaler), ('matfac', mf)], verbose=True)
pipe_t0 = datetime.now()
embeddings = pipe.fit_transform(wcm)

print(f'Created DL word embeddings of shape {embeddings.shape} in {(datetime.now() - pipe_t0).total_seconds()} seconds.\n')
np.save('DL_embeddings.npy', embeddings)


# Get tweet embeddings from word embeddings

# get vocab dict for union of vocabulary
vocabulary_dict = cm.vocabulary

# load our set of tweets for modeling
tweets2 = pd.read_csv('./data/COVID19_Dataset-text_labels_only.csv')
print('Loaded new set of tweets for modeling.\n')

# get tweet vectors
get_vectors_t0 = datetime.now()
X = get_text_vectors(embeddings, vocabulary_dict, tweets2['Tweet'])
np.save('tweet_embeddings_DL.npy', X)
print(f'Embedded new set of tweets for modeling & saved tweet embeddings in {(datetime.now() - get_vectors_t0).total_seconds()} seconds.\n')


# BINARY CLASSIFICATION

# list of embeddings to iterate over
#embeddings = [X]

# target y
target = np.array(tweets2['Is_Unreliable'])


# Binary classification: five-fold CV

# SVC hyperparams to optimize
kernel = ['rbf', 'linear', 'poly', 'sigmoid']
C = [0.001, 0.01, 0.1, 1, 10]

tune_num = int(tweets2.shape[0]/5) # 20% of the data for tuning
tune_num

# Compute the folds
num_folds = 5
kf = KFold(n_splits = num_folds, shuffle = True, random_state = 1)
splits = kf.split(X) # use any set of embeddings to get train/test indices splits

training_sets = []
testing_sets = []
for train_idx, test_idx in splits:
    training_sets.append(train_idx)
    testing_sets.append(test_idx)

# Construct tuning sets from training sets (20% of data ~ 1 fold) &
# write over training sets (60% of data ~ 3 folds)
tuning_sets = []
for i in range(len(training_sets)):
    train_set = training_sets[i]
    np.random.seed(i)
    tune_idx = np.random.choice(train_set,
                                size = tune_num,
                                replace = False)
    tuning_sets.append(tune_idx)
    new_train_set = train_set[~np.in1d(train_set, tune_idx)]
    training_sets[i] = new_train_set

print('Constructed train, tune, & test sets.\n')


# Define function to create results dictionary & save models in opt_models dictionary
#
# Inputs: numpy array
# Outputs: dictionary containing model performance stats

def get_results(np_array):
    # Initialize dict to store all model stats
    performance = dict()
    
    # Initialize dict to store optimal models from each fold
    opt_models = dict()

    # Loop over folds
    for i in range(num_folds):
        key1 = 'Fold {}'.format(i+1) # key for the performance dict

        train_idx = training_sets[i]
        test_idx = testing_sets[i]
        tune_idx = tuning_sets[i]

        y_train = target[train_idx]
        y_test = target[test_idx]
        y_tune = target[tune_idx]

        X_train = np_array[train_idx]
        X_test = np_array[test_idx]
        X_tune = np_array[tune_idx]

        # Training & tuning
        models = [] # store list of models in order to retrieve optimal model
        tune_auc = [] # tune based on AUC
        model_dict = dict() # to store model params & performance metric values

        for ker in kernel:
            for el in C:
                # Training
                svc_rs = rs + i
                svc = SVC(C = el, kernel = ker, probability = True, random_state=svc_rs)
                svc.fit(X_train, y_train)
                models.append(svc)

                # Tuning
                tune_predict_proba = svc.predict_proba(X_tune)[:,1] # check on this subscripting
                auc = roc_auc_score(y_tune, tune_predict_proba)
                tune_auc.append(auc)

        # Get optimal model based on hyperparameter tuning
        opt_model = models[tune_auc.index(max(tune_auc))] # tune based on AUC
        opt_model_params = opt_model.get_params()
        model_dict['params'] = opt_model_params # store optimal values for model hyperparameters
        opt_models[key1] = opt_model

        # Save training scores
        train_scores = dict() # to store all training scores
        train_predict = opt_model.predict(X_train)
        train_predict_proba = opt_model.predict_proba(X_train)[:,1] # check on this subscripting
        train_scores['auc'] = roc_auc_score(y_train, train_predict_proba)
        train_scores['accuracy'] = accuracy_score(y_train, train_predict)
        train_scores['recall_macro'] = recall_score(y_train, train_predict, average = 'macro')
        train_scores['precision_macro'] = precision_score(y_train, train_predict, average = 'macro')
        train_scores['f1_macro'] = f1_score(y_train, train_predict, average = 'macro')

        # Save training scores dictionary to model dictionary
        model_dict['training'] = train_scores

        # Save tuning scores
        tune_scores = dict() # to store all tuning scores
        tune_predict = opt_model.predict(X_tune)
        tune_predict_proba = opt_model.predict_proba(X_tune)[:,1] # check on this subscripting
        tune_scores['auc'] = roc_auc_score(y_tune, tune_predict_proba)
        tune_scores['accuracy'] = accuracy_score(y_tune, tune_predict)
        tune_scores['recall_macro'] = recall_score(y_tune, tune_predict, average = 'macro')
        tune_scores['precision_macro'] = precision_score(y_tune, tune_predict, average = 'macro')
        tune_scores['f1_macro'] = f1_score(y_tune, tune_predict, average = 'macro')

        # Save tuning scores dictionary to model dictionary
        model_dict['tuning'] = tune_scores

        # Testing
        test_scores = dict() # to store all testing scores
        test_predict = opt_model.predict(X_test)
        test_predict_proba = opt_model.predict_proba(X_test)[:,1]
        test_scores['auc'] = roc_auc_score(y_test, test_predict_proba)
        test_scores['accuracy'] = accuracy_score(y_test, test_predict)
        test_scores['recall_macro'] = recall_score(y_test, test_predict, average = 'macro')
        test_scores['precision_macro'] = precision_score(y_test, test_predict, average = 'macro')
        test_scores['f1_macro'] = f1_score(y_test, test_predict, average = 'macro')

        # Save test scores dictionary to model dictionary
        model_dict['testing'] = test_scores

        # Save model dictionary to overall dictionary
        performance[key1] = model_dict
    
    return performance, opt_models


# get results for IVA embeddings
LDA_results, LDA_models = get_results(X)
LDA_results

print('Trained, tuned, & tested model.\n')


# Define function to create results df from nested dictionary
def create_df(input_dict):
    df = pd.DataFrame(input_dict)
    df = df.transpose()
    
    df_params = df['params'].apply(pd.Series)
    
    df_training = df['training'].apply(pd.Series)
    df_training.columns = ['train_' + str(col) for col in df_training.columns]
    
    df_tuning = df['tuning'].apply(pd.Series)
    df_tuning.columns = ['tune_' + str(col) for col in df_tuning.columns]
    
    df_testing = df['testing'].apply(pd.Series)
    df_testing.columns = ['test_' + str(col) for col in df_testing.columns]
    
    final_df = pd.concat([df_training, df_tuning, df_testing, df_params], axis = 1).reset_index()
    final_df = final_df.rename({'index': 'fold_num'}, axis = 1)
    
    return final_df


# Define function to get means for test results from dataframe of full results
def get_test_means(df):
    filter_cols = [col for col in df if col.startswith('test_')]
    df_test = df[filter_cols]
    df_test_mean = pd.DataFrame(df_test.mean(axis = 0)).transpose()
    
    return df_test_mean


# Save results
LDA_full = create_df(LDA_results)
LDA_full.to_csv('DL_svm_full_results.csv')

print('Saved full results.\n')

LDA_test_mean = get_test_means(LDA_full)
LDA_test_mean.to_csv('DL_svm_testmean_results.csv')

print('Saved test performance averages.\n')

print(f'Total run time was {(datetime.now() - start_time).total_seconds()} seconds')


