# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 14:56:10 2021

@author: kosti
"""

import xml.etree.ElementTree as et
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.util import ngrams
import pickle
import os

def create_svm_model(source, array, percentile,     ):
    files_to_use = []
    texts = []
    review_length = []
    targets = []
    category = []
    polarity_score = []
      
    for i in array:                     # Match nums in array with files
        i = f"part{i}.xml"
        files_to_use.append(i)
        
    for file in os.listdir(source):     
        tokens = []              
        bigrams = []
        trigrams = []
        postags = []
        num_adj_adv = []
        sentiment_intensity = []
        tfidf_scores = []
    
        if file in files_to_use:        # Separate the files to be used
            # Read xml
            xtree = et.parse(source + "/" + file)
            root = xtree.getroot()
    
            def get_reviews(root):
                # texts = []
                for review in root:
                    mid_str = ""
                    for sentences in review:
                        for sentence in sentences:
                            text = sentence.find("text").text         # Sentences
                            mid_str += text + " "
                    texts.append(mid_str)
                return texts
            get_reviews(root)
            
            def get_review_length(root):
                for review in root:
                    mid_sen_lentgh = 0
                    for sentences in review:
                        for sentence in sentences:
                            text = sentence.find("text").text         # Sentences
                            text_tokens = nltk.word_tokenize(text)    # Tokens
                            num_words = len(text_tokens)
                            mid_sen_lentgh += num_words
                    review_length.append(mid_sen_lentgh)
                return review_length
                
            get_review_length(root)
            
            def get_targets(root):
                for review in root:
                    for sentences in review:
                        for sentence in sentences:
                            for opinions in sentence:
                                mid_targets = ""
                                for opinion in opinions:
                                    single_target = opinion.attrib.get("target")
                                    mid_targets += single_target + " "
                        targets.append(mid_targets)
                return targets
            get_targets(root)
            
            def get_category(root):
                for review in root:
                    for sentences in review:
                        for sentence in sentences:
                            for opinions in sentence:
                                mid_category = ''
                                for opinion in opinions:
                                    single_category = opinion.attrib.get("category")
                                    mid_category += single_category + " "
                        category.append(mid_category)
                return category
        
            get_category(root)
          
            def get_polarity(root):
                for review in root:
                    opinion_score = 0
                    for sentences in review:
                        for sentence in sentences:
                            for opinions in sentence:
                                for opinion in opinions:
                                    polarity = opinion.attrib.get("polarity")
                                    if polarity == "positive":
                                        opinion_score +=1
                                    elif polarity == "negative":
                                        opinion_score -=1
                    if opinion_score > 0:
                        polarity_score.append(1)
                    elif opinion_score < 0:
                        polarity_score.append(-1)
                    else:
                        polarity_score.append(0)
                return polarity_score 
            get_polarity(root)
       
    def get_tokens(text):
        
        for i in text:
            mid_tokens = nltk.word_tokenize(i)    # Tokens
            line = ""
            for token in mid_tokens:
                line += token + " "
            tokens.append(line)  
        return tokens
    
    get_tokens(texts)
      
    def get_bigrams(text):  
        for line in text:
            token = nltk.word_tokenize(line)
            mid_bigrams = list(ngrams(token, 2)) 
            line_bigs = ""
            for bigram in mid_bigrams:
                txt_big = ""
                for word in bigram:
                    txt_big += word + " "
                line_bigs += txt_big + ", "
            bigrams.append(line_bigs)
        return bigrams
      
    get_bigrams(texts)
    
    def get_trigrams(text):
        for line in text:
            token = nltk.word_tokenize(line)
            mid_trigrams = list(ngrams(token, 3)) 
            line_trigs = ""
            for trigram in mid_trigrams:
                txt_trig = ""
                for word in trigram:
                    txt_trig += word + " "
                line_trigs += txt_trig + ", "
            trigrams.append(line_trigs)
        return trigrams
    
    get_trigrams(texts)
    
    def get_pos_tags(text):
        for i in text:
            mid_tags = ""
            tokens = nltk.word_tokenize(i)
            text_tags = nltk.pos_tag(tokens)
            
            for tag in text_tags:
                x = tag[1]
                mid_tags += x + " "
            postags.append(mid_tags)
        return postags
    
    get_pos_tags(texts)
    
    def get_adj_adv(text):
        for i in text:
            mid_tags = []
            tokens = nltk.word_tokenize(i)
            text_tags = nltk.pos_tag(tokens)
            for tag in text_tags:
                ud = {'RB','JJ'}
                x = tag[1]
                if x in ud:
                    mid_tags.append(x)
            num_adj_adv.append(len(mid_tags))
        return num_adj_adv
    get_adj_adv(texts)
        
    def get_sentiment_intensity(text):
        sia = SentimentIntensityAnalyzer()
        for i in text:
            score = sia.polarity_scores(i)
            sentiment_intensity.append(score['compound'])
        return sentiment_intensity
    get_sentiment_intensity(texts)
    
    def get_TFIDF(text):
        tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
        tfIdfVectorizer=TfidfVectorizer(lowercase=True, stop_words='english', 
                                        ngram_range=(1, 1), tokenizer=tokenizer.tokenize)
        tfIdf = tfIdfVectorizer.fit_transform(text)
        mid_tfidf = []
        for i in tfIdf:
            mid_tfidf.append(str(i))
        for x in mid_tfidf:
            line = x.split()
            tfidf_scores.append(line[2])
        return tfidf_scores
    
    get_TFIDF(texts)
    
    def create_dataframe(root, train_csv):
        # text = get_reviews(root)
        
        dict = {'Tokens':  tokens, 
                'Bigrams': bigrams, 
                'Trigrams': trigrams, 
                'Tags': postags,
                'Number of words': review_length,
                'Number of adverbs and adjectives': num_adj_adv,
                'Targets': targets,
                'Categories': category,
                'Sentiment intensity': sentiment_intensity,
                'TF-IDF': tfidf_scores,
                'Polarity': polarity_score}
        
        df = pd.DataFrame(dict)
        # print(df)
        # Convert categorical data to one hot encoding
        categorical_cols = ['Tokens', 'Bigrams', 'Trigrams', 
                            'Tags', 'Targets', 'Categories'] 
        one_hot = pd.get_dummies(df[categorical_cols])

        # If we are creating the test dataframe, need to combine test one hot encodings with train one hot encodings
        if train_csv != None:
            train_one_hot = pd.read_csv(train_csv)
            try:
                # Drop the non categorical data that were not used in training
                train_non_categorical = pd.read_csv('train_non_categorical.csv')
                df = df.drop(columns = train_non_categorical.columns)
            except:
                pass
            final_train, one_hot = one_hot.align(train_one_hot, join='right', axis=1)

        data = df.join(one_hot)
        data = data.drop(columns = categorical_cols)
        # Return polarity column back to the end of the dataframe
        cols = list(data.columns.values) # Make a list of all of the columns in the df
        cols.pop(cols.index('Polarity')) # Remove polarity from list
        data = data[cols+['Polarity']]   # Create new dataframe with new order
        return data, one_hot
    
    def save_train_csv(data, features_selected, train_csv):
        non_categorical_cols = ['Number of words', 'Number of adverbs and adjectives',
                                'Sentiment intensity', 'TF-IDF']
        non_categorical_df = pd.DataFrame()
        one_hot = pd.DataFrame()

        # Add to the dataframe the features that were selected by SelectPercentile in training
        for i in range(len(data.columns)):
            if i in features_selected:
                one_hot = pd.concat([one_hot, data[data.columns[i]]], axis=1)

        # Drop from the dataframe the non categorical data
        for col in non_categorical_cols:
            if col not in one_hot.columns.values:
                non_categorical_df[col] = data[col].values
            else:
                one_hot = one_hot.drop(col, axis=1)

        # The resulting dataframe contains the selected one hot encodings
        one_hot.to_csv('train_one_hot.csv', index=False)
        # This dataframe contains the non categorical data that were not selected by SelectPercentile in training
        non_categorical_df.to_csv('train_non_categorical.csv', index=False)
        
    def train_svm_model(root, percentile, train_csv):
        data, one_hot = create_dataframe(root, train_csv)  
        data = data.dropna()         # Removes NaN values
        temp_data = data.to_numpy()  # Converts the dataframe into a numpy array
        X = temp_data[:,:-1]         # Gets the features
        y = temp_data[:,-1]          # Gets the output values
        y = y.astype('int')          # Defines y as integer

        # Splits the model into training and data sets, by a ratio of 90/10
        #X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.1)
        if percentile != 100:
            X_fit = SelectPercentile(mutual_info_classif, percentile=percentile).fit(X, y)
            features_selected = X_fit.get_support(indices=True)
            X = X_fit.transform(X)
            save_train_csv(data, features_selected, train_csv)
        else:
            one_hot.to_csv('train_one_hot.csv', index=False)
            non_categorical_df = pd.DataFrame()
            non_categorical_df.to_csv('train_non_categorical.csv', index=False)

        svm = SVC(kernel='linear', C=20)
        model = svm.fit(X,y)

        filename = 'model.sav'
        pickle.dump(model, open(filename, 'wb'))
    
    if train_csv is None:
        train_svm_model(root, percentile, train_csv)
    else:
        dataframe = create_dataframe(root, train_csv)
        return dataframe


if __name__ == '__main__':
    array = [1, 2, 3, 4, 5]
    folder = "./output"
        
    create_svm_model(folder, array, percentile=100, train_csv=None)

