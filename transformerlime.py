# Transformers / LIME integration
import pandas as pd
import numpy as np
import nltk
import torch
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import ElectraModel, ElectraConfig, ElectraTokenizer
from transformers import XLNetModel, XLNetTokenizer
import matplotlib.pyplot as plt
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
from multiprocessing import Process, Manager
print('loaded imports')

def lime_results(lime_expl, lime_expl_list, lime_time_list, idx, text_data, pipe, lime_explainer):
    # compute lime explanation
    tweet = text_data['Tweet'][idx]
    # y_true = targets[idx]
    # y_predict = predictions[idx]
    num_words = len(re.split("\W+", tweet))
    startt = time.process_time() # to track how long it takes for LIME to form the explanation
    exp = lime_explainer.explain_instance(tweet, pipe.predict_proba, num_features = num_words)
    endt = time.process_time()
    dur = endt - startt

    # save explanations
    lime_time_list.append(dur)
    lime_expl.append((idx, exp))
    lime_expl_list.append((idx, exp.as_list()))


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
        self.text_embeddings = None
        
        # Load pretrained model/tokenizer
        self.model_name = model_name
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

        # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        # self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        # self.model = model_class.from_pretrained(pretrained_weights)

    def fit(self, corpus, y=None):

        """ Do nothing because BERT will be used in transform method. Return self.

            Params:
                corpus: list of strings

            Returns: self
        """

        return self

    def transform(self, new_corpus=None, y=None):

        """ Get text embeddings for given corpus, using specified transformer model.

            Returns: text embeddings (shape: num texts by embedding dimensions)
        """

        # Load pretrained model/tokenizer
        tokenizer = self.model_definition["tokenizer_module"].from_pretrained(self.model_definition["model_name"])
        model = self.model_definition["model_module"].from_pretrained(self.model_definition["model_name"])
        
        embeddings = []
        for text in new_corpus:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state[0]
            mean = torch.mean(last_hidden_states, 0)
            embeddings.append(mean.detach().numpy())

        self.text_embeddings = np.array(embeddings)
        
        # # Load pretrained model/tokenizer
        # tokenizer = self.tokenizer
        # model = self.model
        
        # for k in range(len(new_corpus)):
        #     text = new_corpus[k]
        #     #text = re.sub(r'[_~`@$%^&*[\]+=\|}{\"\'<>/]+', '', text)
        #     text_vec = np.zeros(768) # num features in bert base
            
        #     tokenized = tokenizer.encode(text.lower(), add_special_tokens=True)
        #     #tokenized = np.array(tokenized)
            
        #     # max length of tweet tokens is 83 (from Saswat's code); pad all vectors
        #     maxi = 83
        #     #padded = list()
        #     #padded.append(np.array(tokenized + [0]*(maxi - len(tokenized))))
        #     padded = np.array(tokenized + [0]*(maxi - len(tokenized)))
            
        #     segment_ids = [1]*len(padded)
            
        #     # create tensors
        #     tokens_tensor = torch.tensor([padded])
        #     segments_tensor = torch.tensor([segment_ids])
            
        #     with torch.no_grad():
        #         last_hidden_states = model(tokens_tensor, segments_tensor)[0] # pull out only the last hidden state
            
        #     last_hidden_states = last_hidden_states.numpy() # dim: tweets x words x features (where tweets = 1)
            
        #     word_embeddings = last_hidden_states[0,:,:] # dim: words x features (where features = 768)
            
        #     for j in range(768):
        #         text_vec[j] = word_embeddings[:, j].mean() # should be of dimension 1 x 768

        #     if k == 0:
        #         full_matrix = text_vec
        #     else:
        #         full_matrix = np.vstack((full_matrix, text_vec))

        # self.text_embeddings = full_matrix

        return self.text_embeddings.copy()



# Define TransformerLIME class to do the rest of the work
class TransformerLIME(object):
    def __init__(self, model_name: str, data_file: str = 'data/COVID19_Dataset-CM-ZB-complete with sources.xls', embedding_path: str = 'data/transformer_embeddings/', n_cv_folds = 10, kernels = ['rbf', 'linear', 'poly', 'sigmoid']):
        self.n_cv_folds = n_cv_folds
        self.model_predictions = None
        self.kernels = kernels
        
        # Load data
        self.data_file = data_file
        self.raw_df = pd.read_excel(data_file)
        self.labels = self.raw_df["Is_Unreliable"].to_numpy()
        self.model_name = model_name
        self.clean_name = model_name.split('/')[-1]

        self.embedder = Text2Embed(model_name)
        self.embedder.fit(self.raw_df['Tweet'])

        # Load embeddings
        embedding_file = embedding_path + self.clean_name + '.npy'
        self.tweet_embeddings = np.load(embedding_file)

        # LIME explanations
        self.lime_explanations = dict()#[0]*self.tweet_embeddings.shape[0]
        self.lime_list_explanations = dict()#[0]*self.tweet_embeddings.shape[0]
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
            
            for i in range(self.n_cv_folds): # iterate over model for each cv fold
                fold_num = i+1
                
                # load trained classification model
                svc = joblib.load(f'./fold_estimators/SVM-{kernel}_{self.clean_name}_Fold{fold_num}.joblib')
                
                # create pipeline
                c = make_pipeline(self.embedder, svc)

                # instantiate LIME explainer
                explainer = LimeTextExplainer(class_names = ['Reliable', 'Unreliable'])
                
                # subset data by fold i test set
                subset_text = self.raw_df[self.raw_df['Test_Fold'] == fold_num]
                index_vals = list(subset_text.index)
                subset_text = subset_text['Tweet']
                #subset_text = subset_text.reset_index(drop=True)['Tweet']
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

                with Manager() as manager: # use multiprocessing for LIME explanations
                    lime_expl = manager.list()  # <-- can be shared between processes.
                    lime_expl_as_list = manager.list()  # <-- can be shared between processes.
                    lime_time = manager.list()  # <-- can be shared between processes.
                    processes = []

                    for i in range(len(predictions)):
                        idx = index_vals[i]
                        p = Process(target=lime_results, args=(lime_expl,lime_expl_as_list,lime_time,idx,subset_text[idx],c, explainer)) # pass lists in
                        p.start()
                        processes.append(p)

                    for p in processes:
                        p.join()

                    lime_expl = list(lime_expl)
                    lime_expl_as_list = list(lime_expl_as_list)
                    lime_time = list(lime_time)
                self.lime_times[kernel] += lime_time
                self.lime_explanations[kernel] += lime_expl
                self.lime_list_explanations[kernel] += lime_expl_as_list
                
                # print average LIME explanation time for each fold x kernel combination
                print(f'Average LIME computation time: {np.mean(lime_time)}\n')


        self.model_predictions = model_predictions

    def save_results(self):

        for kernel in self.kernels:
            # save explanations
            with open(f'{self.clean_name}_{kernel}_explanations.pkl', 'wb') as f:
                pickle.dump(self.lime_explanations[kernel], f)

            # save explanations as lists
            with open(f'{self.clean_name}_{kernel}_explanations_aslist.pkl', 'wb') as f:
                pickle.dump(self.lime_list_explanations[kernel], f)

            # save LIME explanation time
            with open(f'{self.clean_name}_{kernel}_lime_time.pkl', 'wb') as f:
                pickle.dump(self.lime_times[kernel], f)



if __name__ == '__main__':
    tfLIME = TransformerLIME(model_name = 'bert-base-uncased', kernels = ['rbf', 'linear'])
    tfLIME.main()


