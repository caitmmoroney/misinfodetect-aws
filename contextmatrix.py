# Create Word-Context Co-Occurrence Matrix

# import modules
import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import TweetTokenizer
from sklearn.base import TransformerMixin
import re
from nltk.stem import WordNetLemmatizer

contractions = { 
    "ain't": "are not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "he will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}


# dictionary that matches nltk_pos tags to wordnet-compatible PoS tags
tag_dict = {
    'J': wordnet.ADJ,
    'N': wordnet.NOUN,
    'V': wordnet.VERB,
    'R': wordnet.ADV
}

# to convert nltk_pos tags to wordnet-compatible PoS tags
def convert_pos_wordnet(tag):
    tag_abbr = tag[0].upper()
                
    if tag_abbr in tag_dict:
        return tag_dict[tag_abbr]


class ContextMatrix(TransformerMixin):
    
    # initialize class & private variables
    def __init__(self,
                 window_size = 4,
                 remove_stopwords: bool = True,
                 add_start_end_tokens: bool = True,
                 lowercase: bool = False,
                 lemmatize: bool = False,
                 pmi: bool = False,
                 spmi_k = 1,
                 laplace_smoothing = 0,
                 pmi_positive: bool = False,
                 sppmi_k = 1):
        
        """ Params:
                window_size: size of +/- context window (default = 4)
                remove_stopwords: boolean, whether or not to remove NLTK English stopwords
                add_start_end_tokens: boolean, whether or not to append <START> and <END> to the
                beginning/end of each document in the corpus (default = True)
                lowercase: boolean, whether or not to convert words to all lowercase
                lemmatize: boolean, whether or not to lemmatize input text
                pmi: boolean, whether or not to compute pointwise mutual information
                pmi_positive: boolean, whether or not to compute positive PMI
        """
        self.window_size = window_size
        self.remove_stopwords = remove_stopwords
        self.add_start_end_tokens = add_start_end_tokens
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.pmi = pmi
        self.spmi_k = spmi_k
        self.laplace_smoothing = laplace_smoothing
        self.pmi_positive = pmi_positive
        self.sppmi_k = sppmi_k
        self.corpus = None
        self.clean_corpus = None
        self.vocabulary = None
        self.X = None
        self.doc_terms_lists = None
        self.lemmatizer = WordNetLemmatizer()
        self.detokenizer = TreebankWordDetokenizer()
        self.tweet_tokenizer = TweetTokenizer()
        self.new_vocab = None
        self.new_vocab_dict = None
    
    def fit(self, corpus: list, y = None):
        
        """ Learn the dictionary of all unique tokens for given corpus.
        
            Params:
                corpus: list of strings
            
            Returns: self
        """
        self.corpus = corpus
        
        term_dict = dict()
        k = 0
        corpus_words = []
        clean_corpus = []
        doc_terms_lists = []
        #detokenizer = TreebankWordDetokenizer()
        #lemmatizer = WordNetLemmatizer()
        
        for text in corpus:
            #text = re.sub(r'[_~`@$%^&*[\]+=\|}{\"\'<>/]+', '', text)
            
            # expand contractions
            for key in contractions.keys():
                text = re.sub(key, contractions[key], text, flags=re.IGNORECASE)
            
            #words = word_tokenize(text)
            words = self.tweet_tokenizer.tokenize(text)
            
            if self.remove_stopwords:
                clean_words = []
                for word in words:
                    if word.lower() not in set(stopwords.words('english')):
                        clean_words.append(word)
                words = clean_words
                
            if self.lowercase:
                clean_words = []
                for word in words:
                    clean_words.append(word.lower())
                
                words = clean_words
                
            if self.lemmatize:
                clean_words = []
                for word in words:
                    PoS_tag = pos_tag([word])[0][1]
                    
                    # to change contractions to full word form
                    #if word in contractions:
                    #    word = contractions[word]

                    if PoS_tag[0].upper() in 'JNVR':
                        word = self.lemmatizer.lemmatize(word, convert_pos_wordnet(PoS_tag))
                    else:
                        word = self.lemmatizer.lemmatize(word)

                    clean_words.append(word)
                    
                words = clean_words
            
            # detokenize trick taken from this StackOverflow post:
            # https://stackoverflow.com/questions/21948019/python-untokenize-a-sentence
            # and NLTK treebank documentation:
            # https://www.nltk.org/_modules/nltk/tokenize/treebank.html
            text = self.detokenizer.detokenize(words)
            clean_corpus.append(text)
            
            [corpus_words.append(word) for word in words]
            
            if self.add_start_end_tokens:
                words = ['<START>'] + words + ['<END>']
            
            doc_terms_lists.append(words)
            
        self.clean_corpus = clean_corpus
        
        self.doc_terms_lists = doc_terms_lists
        
        corpus_words = list(set(corpus_words))
        
        if self.add_start_end_tokens:
            corpus_words = ['<START>'] + corpus_words + ['<END>']
        
        corpus_words = sorted(corpus_words)
        
        for el in corpus_words:
            term_dict[el] = k
            k += 1
            
        self.vocabulary = term_dict
        
        return self
        
    def transform(self, new_corpus = None, y = None):
        
        """ Compute the co-occurrence matrix for given corpus and window_size, using term dictionary
            obtained with fit method.
        
            Returns: term-context co-occurrence matrix (shape: target terms by context terms) with
            raw counts
        """
        self.new_vocab = [] # to store new vocabulary
        doc_terms_list_new = [] # to store texts as lists of words
        
        num_original_terms = len(self.vocabulary)
        window = self.window_size

        #if type(new_corpus) != list:
            #print('The new corpus should be of type list.')
            #new_corpus = self.corpus
        
        for text in new_corpus:
            #text = re.sub(r'[_~`@$%^&*[\]+=\|}{\"\'<>/]+', '', text)
            
            # expand contractions
            for key in contractions.keys():
                text = re.sub(key, contractions[key], text, flags=re.IGNORECASE)
            
            words = self.tweet_tokenizer.tokenize(text)
            
            if self.remove_stopwords:
                clean_words = []
                for word in words:
                    if word.lower() not in set(stopwords.words('english')):
                        clean_words.append(word)
                words = clean_words
                
            if self.lowercase:
                clean_words = []
                for word in words:
                    clean_words.append(word.lower())
                
                words = clean_words
                
            if self.lemmatize:
                clean_words = []
                for word in words:
                    PoS_tag = pos_tag([word])[0][1]
                    
                    # to change contractions to full word form
                    #if word in contractions:
                    #    word = contractions[word]

                    if PoS_tag[0].upper() in 'JNVR':
                        word = self.lemmatizer.lemmatize(word, convert_pos_wordnet(PoS_tag))
                    else:
                        word = self.lemmatizer.lemmatize(word)

                    clean_words.append(word)
                    
                words = clean_words
                
            if self.add_start_end_tokens:
                words = ['<START>'] + words + ['<END>']
            
            # add words to new vocab
            self.new_vocab += words

            # add list of words to new list of documents' terms
            doc_terms_list_new.append(words)

        # OUTSIDE OF LOOP OVER TEXTS #######################################
        self.new_vocab = list(set(self.new_vocab))
        
        if self.add_start_end_tokens:
            self.new_vocab = ['<START>'] + self.new_vocab + ['<END>']
        
        self.new_vocab = sorted(self.new_vocab)
        num_new_terms = len(self.new_vocab)
        # create a dict for new vocab
        self.new_vocab_dict = dict()
        m = 0
        for el in self.new_vocab:
            self.new_vocab_dict[el] = m
            m += 1

        # initialize word-context co-occurrence matrix of shape (num target words = len original vocab) X (num context words = len new vocab)
        X = np.full((num_original_terms, num_new_terms), self.laplace_smoothing) # this is NOT a square matrix anymore
        
        # NEW LOOP OVER TEXTS ##############################################
        for k in range(len(doc_terms_list_new)): # loop over list of texts
            words = doc_terms_list_new[k] # get list of words for the kth text
            
            for i in range(len(words)): # loop over list of words
                target = words[i]
                
                # check to see if target word is in the original dictionary; if not, skip
                if target in self.vocabulary:
                    
                    # grab index from dictionary
                    target_dict_index = self.vocabulary[target]
                    
                    # find left-most and right-most window indices for each target word
                    left_end_index = max(i - window, 0)
                    right_end_index = min(i + window, len(words) - 1)
                    
                    # loop over all words within window
                    # NOTE: this will include the target word; make sure to skip over it
                    for j in range(left_end_index, right_end_index + 1):
                        
                        # skip "context word" where the "context word" index is equal to the
                        # target word index
                        if j != i:
                            context_word = words[j]
                            
                            # check to see if context word is in the new fitted dictionary; if
                            # not, skip
                            if context_word in self.new_vocab:
                                X[target_dict_index, self.new_vocab_dict[context_word]] += 1 # add 1 for each observed target-context pair
        
        # if pmi = True, compute pmi matrix from word-context raw frequencies
        # more concise code taken from this StackOverflow post:
        # https://stackoverflow.com/questions/58701337/how-to-construct-ppmi-matrix-from-a-text-corpus
        if self.pmi:
            denom = X.sum()
            col_sums = X.sum(axis = 0)
            row_sums = X.sum(axis = 1)
            
            expected = np.outer(row_sums, col_sums)/denom
            
            X = X/expected
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                
                    if X[i,j] > 0:
                        
                        
                        X[i,j] = np.log(X[i,j]) - np.log(self.spmi_k)
                        
                        if self.pmi_positive:
                            X[i,j] = max(X[i,j] - np.log(self.sppmi_k), 0)
        
        # note that X is a dense matrix
        self.X = X

        return X

    def fit_transform(self, corpus, y=None):
        self.corpus = corpus
        window = self.window_size
        
        term_dict = dict()
        k = 0
        corpus_words = []
        clean_corpus = []
        doc_terms_lists = []
        #detokenizer = TreebankWordDetokenizer()
        #lemmatizer = WordNetLemmatizer()
        
        # FIRST LOOP OVER CORPUS ##############################################
        for text in corpus:
            #text = re.sub(r'[_~`@$%^&*[\]+=\|}{\"\'<>/]+', '', text)
            
            # expand contractions
            for key in contractions.keys():
                text = re.sub(key, contractions[key], text, flags=re.IGNORECASE)
            
            #words = word_tokenize(text)
            words = self.tweet_tokenizer.tokenize(text)
            
            if self.remove_stopwords:
                clean_words = []
                for word in words:
                    if word.lower() not in set(stopwords.words('english')):
                        clean_words.append(word)
                words = clean_words
                
            if self.lowercase:
                clean_words = []
                for word in words:
                    clean_words.append(word.lower())
                
                words = clean_words
                
            if self.lemmatize:
                clean_words = []
                for word in words:
                    PoS_tag = pos_tag([word])[0][1]
                    
                    # to change contractions to full word form
                    #if word in contractions:
                    #    word = contractions[word]

                    if PoS_tag[0].upper() in 'JNVR':
                        word = self.lemmatizer.lemmatize(word, convert_pos_wordnet(PoS_tag))
                    else:
                        word = self.lemmatizer.lemmatize(word)

                    clean_words.append(word)
                    
                words = clean_words
            
            # detokenize trick taken from this StackOverflow post:
            # https://stackoverflow.com/questions/21948019/python-untokenize-a-sentence
            # and NLTK treebank documentation:
            # https://www.nltk.org/_modules/nltk/tokenize/treebank.html
            text = self.detokenizer.detokenize(words)
            clean_corpus.append(text)
            
            [corpus_words.append(word) for word in words]
            
            if self.add_start_end_tokens:
                words = ['<START>'] + words + ['<END>']
            
            doc_terms_lists.append(words)
            
        # OUTSIDE LOOP OVER CORPUS ##############################################
        self.clean_corpus = clean_corpus
        
        self.doc_terms_lists = doc_terms_lists
        
        corpus_words = list(set(corpus_words))
        
        if self.add_start_end_tokens:
            corpus_words = ['<START>'] + corpus_words + ['<END>']
        
        corpus_words = sorted(corpus_words)
        
        for el in corpus_words:
            term_dict[el] = k
            k += 1
            
        self.vocabulary = term_dict

        num_terms = len(corpus_words)

        # initialize word-context co-occurrence matrix of shape (num target words = len vocab) X (num context words = len vocab)
        X = np.full((num_terms, num_terms), self.laplace_smoothing) # this is a square matrix

        # NEW LOOP OVER TEXTS ##############################################
        for k in range(len(self.doc_terms_lists)): # loop over list of texts
            words = self.doc_terms_lists[k] # get list of words for the kth text
            
            for i in range(len(words)): # loop over list of words
                target = words[i]
                
                # check to see if target word is in the original dictionary; if not, skip
                if target in self.vocabulary:
                    
                    # grab index from dictionary
                    target_dict_index = self.vocabulary[target]
                    
                    # find left-most and right-most window indices for each target word
                    left_end_index = max(i - window, 0)
                    right_end_index = min(i + window, len(words) - 1)
                    
                    # loop over all words within window
                    # NOTE: this will include the target word; make sure to skip over it
                    for j in range(left_end_index, right_end_index + 1):
                        
                        # skip "context word" where the "context word" index is equal to the
                        # target word index
                        if j != i:
                            context_word = words[j]
                            
                            # check to see if context word is in the fitted dictionary; if
                            # not, skip
                            if context_word in self.vocabulary:
                                X[target_dict_index, self.vocabulary[context_word]] += 1 # add 1 for each observed target-context pair
        
        # if pmi = True, compute pmi matrix from word-context raw frequencies
        # more concise code taken from this StackOverflow post:
        # https://stackoverflow.com/questions/58701337/how-to-construct-ppmi-matrix-from-a-text-corpus
        if self.pmi:
            denom = X.sum()
            col_sums = X.sum(axis = 0)
            row_sums = X.sum(axis = 1)
            
            expected = np.outer(row_sums, col_sums)/denom
            
            X = X/expected
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                
                    if X[i,j] > 0:
                        
                        
                        X[i,j] = np.log(X[i,j]) - np.log(self.spmi_k)
                        
                        if self.pmi_positive:
                            X[i,j] = max(X[i,j] - np.log(self.sppmi_k), 0)
        
        # note that X is a dense matrix
        self.X = X

        return X




# define class to get string embeddings from word embeddings

def get_text_vectors(word_embeddings, # numpy array
                     word_index_dict, # dictionary mapping words to index in array
                     text_list, # list of strings to derive embeddings for
                     remove_stopwords: bool = True,
                     add_start_end_tokens: bool = True,
                     lowercase: bool = False,
                     lemmatize: bool = False,
                     ):
    
    lemmatizer = WordNetLemmatizer()
    tokenizer = TweetTokenizer()
    
    for k in range(len(text_list)):
        text = text_list[k]

        # expand contractions
        for key in contractions.keys():
            text = re.sub(key, contractions[key], text, flags=re.IGNORECASE)
        
        #text = re.sub(r'[_~`@$%^&*[\]+=\|}{\"\'<>/]+', '', text)
        text_vec = np.zeros(word_embeddings.shape[1]) # initialize text vector to zeros in case no words in the tweet are in the vocab dict
        words = tokenizer.tokenize(text)
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

                if PoS_tag[0].upper() in 'JNVR':
                    word = lemmatizer.lemmatize(word, convert_pos_wordnet(PoS_tag))
                else:
                    word = lemmatizer.lemmatize(word)

                clean_words.append(word)

            words = clean_words

        if add_start_end_tokens:
            words = ['<START>'] + words + ['<END>']
        
        for i in range(len(words)): # loop through the words in the tweet
            word = words[i]
            if word in word_index_dict: # if the word is in the vocab dict...
                word_embed_vec = word_embeddings[word_index_dict[word],:] # extract the corresponding word embedding
                if tracker == 0: # if this is the first word in the tweet that's in the vocab dict that we've come across...
                    text_matrix = word_embed_vec # the text matrix is the word embedding
                else: # if this is not the first word in the tweet that's in the vocab dict...
                    text_matrix = np.vstack((text_matrix, word_embed_vec)) # stack the text matrix on the new word embedding
                    
                # only increment if we have come across a word in the embeddings dictionary
                tracker += 1
            
        if tracker != 0: # if at least one token in the tweet is in the vocab dict...
            if tracker == 1: # if there was only 1 word in the tweet that was in the vocab dict...
                text_vec = text_matrix # set the text vec equal to that word vec
            elif tracker > 1: # if there was more than 1 word in the tweet that was in the vocab dict...
                text_vec = np.mean(text_matrix, axis=0) # set text vector to mean of word embeddings (average over all rows)
                    
        #for j in range(len(text_vec)):
            #text_vec[j] = text_matrix[:,j].mean()
            
        if k == 0:
            full_matrix = text_vec
        else:
            full_matrix = np.vstack((full_matrix, text_vec))
            
    return full_matrix
