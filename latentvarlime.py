# Transformers / LIME integration
import pandas as pd
import numpy as np
import nltk
from sklearn.svm import SVC
from sklearn.base import TransformerMixin
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from lime import submodular_pick
import re
import pickle
import time
import joblib
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
import sys
print('loaded imports')


# to convert nltk_pos tags to wordnet-compatible PoS tags
def convert_pos_wordnet(tag):
    tag_abbr = tag[0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }

    if tag_abbr in tag_dict:
        return tag_dict[tag_abbr]


class Text2Embed(TransformerMixin):
    """ Description:
        Transformer that takes in a list of strings, constructs word embeddings
        using BERT, and then provides the text embeddings of a (new) list of texts
        depending on which words in the "vocab" occur in the (new) strings.
    """

    # initialize class & private variables
    def __init__(self, model_name):

        self.corpus = None
        #self.X = None
        self.text_embeddings = np.load(f'./data/latent_var_embeddings/tweet_embed_{model_name}.npy')
        self.word_embeddings = np.load(f'./data/word_embeddings/word_embed_{model_name}.npy')

        # Load vocabulary list
        with open('./data/word_embeddings/tweet_vocab_list', 'rb') as f:
            self.tweet_vocab_list = pickle.load(f)

        # Create vocabulary dictionary
        self.vocabulary_dict = dict()
        for i in range(len(self.tweet_vocab_list)):
            word = self.tweet_vocab_list[i]
            self.vocabulary_dict[word] = i

        # to convert contractions picked up by word_tokenize() into full words
        self.contractions = {
            "n't": 'not',
            "'ve": 'have',
            "'s": 'is',  # note that this will include possessive nouns
            'gonna': 'going to',
            'gotta': 'got to',
            "'d": 'would',
            "'ll": 'will',
            "'re": 'are',
            "'m": 'am',
            'wanna': 'want to'
        }

    def get_text_vectors(self, word_embeddings, # numpy array
                         word_index_dict, # dictionary mapping words to index in array
                         text_list, # list of strings to derive embeddings for
                         remove_stopwords = True,
                         lowercase = True,
                         lemmatize = True,
                         add_start_end_tokens = True):
        
        lemmatizer = WordNetLemmatizer()
        
        for k in range(len(text_list)):
            text = text_list[k]
            text = re.sub(r'[_~`@$%^&*[\]+=\|}{\"\'<>/]+', '', text)
            text_vec = np.zeros(word_embeddings.shape[1])
            words = word_tokenize(text)
            tracker = 0 # to track whether we've encountered a word for which we have an embedding (in each tweet)
            
            if remove_stopwords:
                clean_words = []
                for word in words:
                    if word.lower() not in set(stopwords.words('english')):
                        clean_words.append(word)
                words = clean_words

            if lowercase:
                clean_words = []
                for word in words:
                    clean_words.append(word.lower())

                words = clean_words

            if lemmatize:
                clean_words = []
                for word in words:
                    PoS_tag = pos_tag([word])[0][1]

                    # to change contractions to full word form
                    if word in self.contractions:
                        word = self.contractions[word]

                    if PoS_tag[0].upper() in 'JNVR':
                        word = lemmatizer.lemmatize(word, convert_pos_wordnet(PoS_tag))
                    else:
                        word = lemmatizer.lemmatize(word)

                    clean_words.append(word)

                words = clean_words

            if add_start_end_tokens:
                words = ['<START>'] + words + ['<END>']
            
            for i in range(len(words)):
                word = words[i]
                if word in word_index_dict:
                    word_embed_vec = word_embeddings[word_index_dict[word],:]
                    if tracker == 0:
                        text_matrix = word_embed_vec
                    else:
                        text_matrix = np.vstack((text_matrix, word_embed_vec))
                        
                    # only increment if we have come across a word in the embeddings dictionary
                    tracker += 1
                        
            for j in range(len(text_vec)):
                text_vec[j] = text_matrix[:,j].mean()
                
            if k == 0:
                full_matrix = text_vec
            else:
                full_matrix = np.vstack((full_matrix, text_vec))
                
        return full_matrix

    def fit(self, corpus, y=None):

        """ Do nothing because BERT will be used in transform method. Return self.

            Params:
                corpus: list of strings

            Returns: self
        """

        return self

    def transform(self, new_corpus=None, y=None):

        """ Get text embeddings for given corpus, using predefined term dictionary and word embeddings.

            Returns: text embeddings (shape: num texts by embedding dimensions)
        """

        full_matrix = self.get_text_vectors(self.word_embeddings, self.vocabulary_dict, new_corpus)

        self.text_embeddings = full_matrix

        return self.text_embeddings.copy()



# Define TransformerLIME class to do the rest of the work
class LatentVarLIME(object):
    def __init__(self, model_name: str = 'ICA', data_file: str = 'data/COVID19_Dataset-CM-ZB-complete with sources_wTestFold.csv', embedding_path: str = 'data/latent_var_embeddings/', fold_num = 1, kernels = ['rbf', 'linear', 'poly', 'sigmoid']):
        self.fold_num = fold_num
        self.model_predictions = None
        self.kernels = kernels
        self.model_name = model_name
        
        # Load data
        self.data_file = data_file
        self.raw_df = pd.read_csv(data_file)
        self.labels = self.raw_df["Is_Unreliable"].to_numpy()
        self.embedder = Text2Embed(model_name)
        self.embedder.fit(self.raw_df['Tweet'])

        # Load embeddings
        self.tweet_embeddings = np.load(f'./data/latent_var_embeddings/tweet_embed_{model_name}.npy')

        # LIME explanations
        self.lime_explanations = dict()
        self.lime_list_explanations = dict()
        self.lime_times = dict()

    def main(self):
        self.lime_explain()
        self.save_results()

    def lime_explain(self):
        model_predictions = dict()
        for kernel in self.kernels: # iterate over kernels
            model_predictions[kernel] = [] # to save model predictions by SVM kernel model
            self.lime_explanations[kernel] = []
            self.lime_list_explanations[kernel] = []
            self.lime_times[kernel] = []

            # load trained classification model
            svc = joblib.load(f'./fold_estimators/SVM-{kernel}_{self.model_name}_Fold{self.fold_num}.joblib')
            
            # create pipeline
            c = make_pipeline(self.embedder, svc)

            # instantiate LIME explainer
            explainer = LimeTextExplainer(class_names = ['Reliable', 'Unreliable'])
            
            # subset data by fold i test set
            subset_text = self.raw_df[self.raw_df['Test_Fold'] == self.fold_num]
            index_vals = list(subset_text.index)
            subset_text = subset_text['Tweet'] # pd.Series
            subset_embeddings = self.tweet_embeddings[index_vals,:]

            # get model predictions
            predictions = svc.predict(subset_embeddings)
            # [model_predictions[kernel] += [(idx, pred)] for idx, pred in zip(index_vals, predictions.tolist())]
            for idx, pred in zip(index_vals, predictions.tolist()):
                model_predictions[kernel] += [(idx, pred)]
            
            # initialize empty lists
            lime_expl = [] # capture lime explanations for each fold
            lime_expl_as_list = [] # capture lime explanations (as lists) for each fold
            lime_time = [] # capture lime processing time for each fold

            for i in range(len(predictions)):
                if (i+1) % 3 == 0:
                    print(f'Working on tweet {i+1} of {len(predictions)}.')

                idx = index_vals[i]
                
                # compute lime explanation
                tweet = subset_text[idx]
                # y_true = targets[idx]
                # y_predict = predictions[idx]
                num_words = len(re.split("\W+", tweet))
                startt = time.process_time() # to track how long it takes for LIME to form the explanation
                exp = explainer.explain_instance(tweet, c.predict_proba, num_features = num_words)
                endt = time.process_time()
                dur = endt - startt

                # save explanations
                lime_time.append(dur)
                lime_expl.append((idx, exp))
                lime_expl_as_list.append((idx, exp.as_list()))

            self.lime_times[kernel] += lime_time
            self.lime_explanations[kernel] += lime_expl
            self.lime_list_explanations[kernel] += lime_expl_as_list
            
            # print average LIME explanation time for each fold x kernel combination
            print(f'Average LIME computation time: {np.mean(lime_time)}\n')

        self.model_predictions = model_predictions

    def save_results(self):

        for kernel in self.kernels:
            # save explanations
            with open(f'{self.model_name}_{kernel}_explanations_fold{self.fold_num}.pkl', 'wb') as f:
                pickle.dump(self.lime_explanations[kernel], f)

            # save explanations as lists
            with open(f'{self.model_name}_{kernel}_explanations_aslist_fold{self.fold_num}.pkl', 'wb') as f:
                pickle.dump(self.lime_list_explanations[kernel], f)

            # save LIME explanation time
            with open(f'{self.model_name}_{kernel}_lime_time_fold{self.fold_num}.pkl', 'wb') as f:
                pickle.dump(self.lime_times[kernel], f)

            # save model predictions
            with open(f'{self.model_name}_{kernel}_predictions_fold{self.fold_num}.pkl', 'wb') as f:
                pickle.dump(self.model_predictions[kernel], f)



if __name__ == '__main__':
    args = sys.argv
    fold_num = int(args[1])
    lvLIME = LatentVarLIME(fold_num = fold_num, kernels = ['linear'])
    lvLIME.main()


