import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans, KMeans
import operator
from gensim.models.word2vec import Word2Vec
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import collections
from sklearn.metrics import accuracy_score
import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


def restore_tokens_from_text(text):
    """
        由于Excel里读取出来的数据都是string类型,这里我们把文本转换成list.
    """
    regex_to_list = r"[\[\',\]\(\)）（]"
    space_separated_string = re.sub(regex_to_list, '', text)
    word_list = space_separated_string.split(' ')
    word_list.append('UNKNOWN')
    return word_list


def restore_features_from_text(text):
    regex_to_list = r"[\[\',\]\(\)）（]"
    space_separated_string = re.sub(regex_to_list, '', text)
    word_list = space_separated_string.split(' ')
    return word_list

def restore_floats_from_text(text, embedding_dim = 100):
    regex_to_list = r"[\[\',\]\(\)）（] "
    try:
        space_separated_string = re.sub(regex_to_list, '', text)
    except:
        print(text)
        return np.zeros(embedding_dim)
    space_separated_string = space_separated_string.replace('[', '')
    space_separated_string = space_separated_string.replace(']', '')
    # if space_separated_string[-1] == ']':
    #     space_separated_string = space_separated_string[:-1]
    # if space_separated_string[0] == '[':
    #     space_separated_string = space_separated_string[1:]
    word_list = space_separated_string.split(' ')
    try:
        [float(w) for w in word_list if w != ""]
    except:
        print(text)
    return [float(w) for w in word_list if w != ""]

def convert_tokens_to_sentence(tokens):
    return ' '.join(tokens)


def load_data(file_path):
    """
        1. 把数据从Excel里读取出来.
        2. 把merged_tokens从字符串转换成tokens分词的列表.
        3. 把分词合并成以空格分割的语句,以便后面进行tf-idf计算.
        4. 选取class, tokens, sentence两列组成新的DataFrame返回给调用函数.
    :param file_path:
    :return:
    """

    df_original = pd.read_excel(file_path, names=['class', 'descriptions', 'baidu_tokens', 'jieba_tokens', 'merged_tokens'])
    df_original['tokens'] = df_original['merged_tokens'].apply(restore_tokens_from_text)
    df_original['sentence'] = df_original['tokens'].apply(convert_tokens_to_sentence)
    return df_original[['class', 'tokens', 'sentence']]


def load_transformed_data(file_path, dims=100):
    df_original = pd.read_excel(file_path, names=['class', 'tokens', 'sentence', 'top_tokens', 'features'])
    df_original['tokens'] = df_original['tokens'].apply(restore_tokens_from_text)
    df_original['top_tokens'] = df_original['top_tokens'].apply(restore_features_from_text)
    df_original['features'] = df_original['features'].apply(lambda x: restore_floats_from_text(x, dims))
    return df_original[['top_tokens', 'features', 'class']]

def get_word_list_from_dataframe(df, col_name):
    """
        遍历DataFrame的一列,把所有的token存放到一个list里面.
    """
    w_list = []
    for idx, row in df.iterrows():
        w_list += restore_tokens_from_text(row[col_name])
    return w_list


def w2v_file_name_from_parameters(col, vect_dims=100, window=5, min_count=1, seed=11):
    """
        根据参数组合来获取word2vec模型保存的文件名.
    """
    col = col.replace('_', '')
    return 'col-{0}_dim-{1}_window-{2}_mincount-{3}_seed-{4}.w2v'.format(col, str(vect_dims), str(window), str(min_count), str(seed))


def generate_word_embedding_from_df(df, col='tokens', vect_dims=100, window=5, min_count=1, workers=4, seed=11):
    saved_model_name = w2v_file_name_from_parameters(col, vect_dims, window, min_count, seed)
    if os.path.exists(saved_model_name):
        return Word2Vec.load(saved_model_name)
    else:
        all_token_lists = df[col]
        # size: The number of dimensions of the embedding, e.g. the length of the dense vector to represent each token(word)
        # sg: THe training algorithm, either CBOW(0) or skip gram(1).
        # window: The maximum distance between a target word and words around the target word.
        # min_count: The minimum count of words to consider when training the model; words with an occurence less than this count will be ignored.
        w2v_model = Word2Vec(sentences=all_token_lists, size=vect_dims, sg=1, window=window, min_count=min_count, seed=seed, workers=workers)
        w2v_model.save(saved_model_name)
        return w2v_model


def get_top_tokens_in_doc(df, Xtr, features, row_id, top_n=25):
    row = np.squeeze(Xtr[row_id].toarray())
    tokens = df.loc[row_id]['tokens']
    token_length = len(tokens)
#     print('Token length: ', str(token_length))
    token_values = {}
    for i in range(token_length):
        # Get tfidf score for each token
        token_name = tokens[i]
        try:
            if token_name in vectorizer.vocabulary_:
                token_index = vectorizer.vocabulary_[token_name]
                token_value = row[token_index]
            else:
                token_value = 0
        except:
            print("Exception: ", str(row_id))
        token_values[token_name] = token_value
    # Sort the tokens by tfidf values
    sorted_tokens = sorted(token_values.items(), key=operator.itemgetter(1), reverse=True)
#     print(sorted_tokens)
    # Get the most weighted tokens
    top_tokens = []
    padding_count = 0
#     print("Sorted tokens length: ", str(len(sorted_tokens)))
    if len(sorted_tokens) < top_n:
        padding_count = top_n - len(sorted_tokens)
        for i in range(len(sorted_tokens)):
            top_tokens.append(sorted_tokens[i][0])
    else:
        for i in range(top_n):
            top_tokens.append(sorted_tokens[i][0])
    for i in range(padding_count):
        top_tokens.append('UNKNOWN')
    return top_tokens


def convert_tokens_to_features(tokens, embedding_dim=100):
    default_embedding = np.zeros(embedding_dim, dtype=int).tolist()
    features = []
    for t in tokens:
        if t in w2v_model:
            features += w2v_model[t].tolist()
        else:
            features += default_embedding
    return features


class TokensPicker(BaseEstimator, TransformerMixin):

    def __init__(self, embed_dim=10, top_n=10, window=5, min_count=1):
        self.embedding_dim = embed_dim
        self.top_n_token = top_n
        self.window = window
        self.min_count = min_count

    def generate_word_embedding_from_df(self, df, col='tokens', vect_dims=100, window=5, min_count=1, workers=4, seed=11):
        saved_model_name = w2v_file_name_from_parameters(col, vect_dims, window, min_count, seed)
        if os.path.exists(saved_model_name):
            return Word2Vec.load(saved_model_name)
        else:
            all_token_lists = df[col]
            # size: The number of dimensions of the embedding, e.g. the length of the dense vector to represent each token(word)
            # sg: THe training algorithm, either CBOW(0) or skip gram(1).
            # window: The maximum distance between a target word and words around the target word.
            # min_count: The minimum count of words to consider when training the model; words with an occurence less than this count will be ignored.
            w2v_model = Word2Vec(sentences=all_token_lists, size=vect_dims, sg=1, window=window, min_count=min_count,
                                 seed=seed, workers=workers)
            w2v_model.save(saved_model_name)
            return w2v_model

    def fit(self, df, y=None):
        self.w2v_model = self.generate_word_embedding_from_df(df,
                                                    col='tokens',
                                                    vect_dims=self.embedding_dim,
                                                    window=self.window
                                                    )
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(df['sentence'].tolist())
        return self

    def get_top_tokens_in_doc(self, df, vectors, row_id, top_n=25):
        row = np.squeeze(vectors[row_id].toarray())
        tokens = df.loc[row_id]['tokens']
        token_length = len(tokens)
        #     print('Token length: ', str(token_length))
        token_values = {}
        for i in range(token_length):
            # Get tfidf score for each token
            token_name = tokens[i]
            try:
                if token_name in self.vectorizer.vocabulary_:
                    token_index = self.vectorizer.vocabulary_[token_name]
                    token_value = row[token_index]
                else:
                    token_value = 0
            except:
                print("Exception: ", str(row_id))
            token_values[token_name] = token_value
        # Sort the tokens by tfidf values
        sorted_tokens = sorted(token_values.items(), key=operator.itemgetter(1), reverse=True)
        #     print(sorted_tokens)
        # Get the most weighted tokens
        top_tokens = []
        padding_count = 0
        #     print("Sorted tokens length: ", str(len(sorted_tokens)))
        if len(sorted_tokens) < top_n:
            padding_count = top_n - len(sorted_tokens)
            for i in range(len(sorted_tokens)):
                top_tokens.append(sorted_tokens[i][0])
        else:
            for i in range(top_n):
                top_tokens.append(sorted_tokens[i][0])
        for i in range(padding_count):
            top_tokens.append('UNKNOWN')
        return top_tokens

    def convert_tokens_to_features_with_weights(self, df, vectors, row_id):
        row = np.squeeze(vectors[row_id].toarray())
        tokens = df.loc[row_id]['top_tokens']
        token_length = len(tokens)
        token_values = {}
        features = None
        for i in range(token_length):
            # Get tfidf score for each token
            t = tokens[i]
            try:
                if t in self.vectorizer.vocabulary_:
                    token_index = self.vectorizer.vocabulary_[t]
                    token_value = row[token_index] * 1000
                    if t in self.w2v_model:
                        current_features = np.asarray(self.w2v_model[t]) * token_value
                        if features is None:
                            features = current_features
                        else:
                            features = np.vstack((features, current_features))
            except:
                print("Exception: ", str(row_id))
        try:
            np.average(features, axis=0)
        except:
            return np.zeros(self.embedding_dim)
        return np.average(features, axis=0)

    def convert_tokens_to_features_1(self, tokens):
        """
        Convert word embeddings to sentence embedding by calculating the average of each dim.
        :param tokens:
        :return:
        """
        features = None
        for t in tokens:
            if t in self.w2v_model:
                if features is None:
                    features = np.asarray(self.w2v_model[t])
                else:
                    features = np.vstack((features, np.asarray(self.w2v_model[t])))
        return np.average(features, axis=0)


    def convert_tokens_to_features(self, tokens):
        default_embedding = np.zeros(self.embedding_dim, dtype=int).tolist()
        features = []
        for t in tokens:
            if t in self.w2v_model:
                features += self.w2v_model[t].tolist()
            else:
                features += default_embedding
        return features

    def transform(self, df, y=None):
        df['top_tokens'] = df.apply(lambda x: self.get_top_tokens_in_doc(df,
                                                                       vectors=self.vectors,
                                                                       row_id=x.name,
                                                                       top_n=self.top_n_token),
                                    axis=1
                                  )
        df['features'] = df.apply(lambda x: self.convert_tokens_to_features_with_weights(df, vectors=self.vectors, row_id=x.name),
                                  axis=1)
        writer_train = pd.ExcelWriter('transformed_topn-{0}_embeddingdim-{1}_window-{2}.xlsx'.format(self.top_n_token, self.embedding_dim, self.window))
        df.to_excel(writer_train, sheet_name='Sheet1')
        writer_train.save()
        return df


class DataFrameSlector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        # 根据传入的列明对DataFrame进行过滤
        # 然后返回sklearn可以处理的数组数据
        return df[self.attribute_names].values


if __name__ == '__main__':
    training_path = '/home/alvin/!Final_Project/training_with_tokens.xlsx'
    testing_path = '/home/alvin/!Final_Project/testing_with_tokens.xlsx'

    # embedding_dim = 10
    # top_n_token = 10

    dims = [768, 512, 256]
    top_ns = [10, 20, 30]
    windows = [10, 8, 5]

    cv_results = []
    for d in dims:
        for t in top_ns:
            for w in windows:
                result = {'d': d, 't': t, 'w': w}
                file_name = 'transformed_topn-{0}_embeddingdim-{1}_window-{2}.xlsx'.format(t, d, w)
                print(file_name)
                df_train = load_data(training_path)

                feature_pipeline = Pipeline([
                    ('Tokens_Picker_Pipeine', TokensPicker(embed_dim=d, top_n=t, window=w))
                ])
                x_train_transformed = feature_pipeline.fit_transform(df_train)
                xgb_final = xgb.XGBClassifier(objective='auc_score', seed=11, learning_rate=0.1, n_estimators=215,
                                              max_depth=3,
                                              min_child_weight=6, gamma=0,
                                              subsample=0.6, colsample_bytree=0.6, reg_alpha=0.05)


                df_traned = load_transformed_data(file_name, d)
                # df_traned = x_train_transformed
                _cv_results = cross_validate(xgb_final, np.asarray(df_traned['features'].tolist()),
                                             df_traned['class'].tolist(),
                                             return_train_score=False, scoring='accuracy', cv=5)
                print(df_traned.loc[0]['features'])
                print(_cv_results)

                print('Average Score: ', np.asarray(_cv_results['test_score']))
                print(np.average(_cv_results['test_score']))
                result['score'] = np.average(_cv_results['test_score'])
                result['fit_time'] = np.average(_cv_results['fit_time'])
                result['score_time'] = np.average(_cv_results['score_time'])
                cv_results.append(result)

    df_results = pd.DataFrame.from_dict(cv_results)
    df_results.to_csv('results_with_weight.csv')
    #
    #
    # result = {'d': 128, 't': 20, 'w': 10}
    # df_train = load_data(training_path)
    #
    # feature_pipeline = Pipeline([
    #     ('Tokens_Picker_Pipeine', TokensPicker(embed_dim=128, top_n=20, window=10))
    # ])
    # x_train_transformed = feature_pipeline.fit_transform(df_train)
    # xgb_final = xgb.XGBClassifier(objective='auc_score', seed=11, learning_rate=0.1, n_estimators=215,
    #                               max_depth=3,
    #                               min_child_weight=6, gamma=0,
    #                               subsample=0.6, colsample_bytree=0.6, reg_alpha=0.05)
    #
    # file_name = 'transformed_topn-{0}_embeddingdim-{1}_window-{2}.xlsx'.format(10, 50, 5)
    # df_traned = load_transformed_data(file_name)
    # _cv_results = cross_validate(xgb_final, np.asarray(df_traned['features'].tolist()),
    #                              df_traned['class'].tolist(),
    #                              return_train_score=False, scoring='accuracy', cv=5)
    # print(df_traned.loc[0]['features'])
    # print(_cv_results)
    #
    # print('Average Score: ', np.asarray(_cv_results['test_score']))
    # print(np.average(_cv_results['test_score']))
    # result['score'] = np.average(_cv_results['test_score'])
    # result['fit_time'] = np.average(_cv_results['fit_time'])
    # result['score_time'] = np.average(_cv_results['score_time'])
    # cv_results.append(result)
    # df_results = pd.DataFrame.from_dict(cv_results)
    # df_results.to_csv('results_with_weight.csv')
    print('Done...')
