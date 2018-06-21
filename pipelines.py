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
from sklearn.preprocessing import StandardScaler
import collections
from sklearn.metrics import accuracy_score


def restore_tokens_from_text(text):
    """
        由于Excel里读取出来的数据都是string类型,这里我们把文本转换成list.
    """
    regex_to_list = r"[\[\',\]\(\)）（]"
    space_separated_string = re.sub(regex_to_list, '', text)
    word_list = space_separated_string.split(' ')
    word_list.append('UNKNOWN')
    return word_list


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


if __name__ == '__main__':
    training_path = '/home/alvin/!Final_Project/training_with_tokens.xlsx'
    testing_path = '/home/alvin/!Final_Project/testing_with_tokens.xlsx'

    embedding_dim = 10
    top_n_token = 10

    print('Load data...')
    df_train = load_data(training_path)
    print('Build word2vec model...')
    w2v_model = generate_word_embedding_from_df(df_train,
                                                col='tokens',
                                                vect_dims=embedding_dim
                                                )
    print('Build TF-IDF vector...')
    vectorizer = TfidfVectorizer()
    vector = vectorizer.fit_transform(df_train['sentence'].tolist())

    features = vectorizer.get_feature_names()

    print('Get top n tokens for each item...')
    df_train['top_tokens'] = df_train.apply(
                                                lambda x: get_top_tokens_in_doc(df_train,
                                                                          Xtr=vector,
                                                                          features=features,
                                                                          row_id=x.name,
                                                                          top_n=top_n_token),
                                                axis=1
                                            )

    print('Create final features from top n tokens for each item...')
    df_train['features'] = df_train['top_tokens'].apply(lambda x: convert_tokens_to_features(x, embedding_dim))

    print('Build KMeans model...')
    km = KMeans(n_clusters=11, init='k-means++', max_iter=100, n_init=1)
    km.fit(df_train['features'].tolist())

    print('Start to predict the results...')
    df_train['predicted'] = km.predict(df_train['features'].tolist())

    print('Save results to excel...')
    df_train[['class', 'predicted', 'top_tokens']].to_csv('results.csv')

    print('Accuracy score: ', accuracy_score(df_train['class'].tolist(), df_train['predicted'].tolist()))

    print('Done...')
