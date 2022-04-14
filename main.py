
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


def remove_chars_from_text(text, chars):
    return "".join([ch for ch in text if ch not in chars])

def rand_forest(Xxx_train,y_train,Xxx_test,y_test):
    random_forest = RandomForestClassifier(criterion='gini')
    random_forest.fit(Xxx_train, y_train)
    pred = random_forest.predict(Xxx_test)
    print(classification_report(y_test, pred))
    print('f1 score: ', f1_score(y_test, pred, average='macro'))
    print('accuracy: ', accuracy_score(y_test, pred))

def princ_comp_method(data,colors):     # TruncatedSVD
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
    :param data:
    :param colors:
    :return:
    """

    data = csr_matrix(data) #это надо так как матрица разрежена
    model = TruncatedSVD(n_components = 2) #создаем объект
    x_reduced_2 = model.fit(data)

    #print(x_reduced_2.explained_variance_ratio_)
    #print(x_reduced_2.explained_variance_ratio_.sum())
    #print(x_reduced_2.singular_values_)
    #print(x_reduced_2.explained_variance_)
    print('Первая компонента:',x_reduced_2.components_[0])
    print(len(x_reduced_2.components_[0]))
    print('Вторая компонента:', x_reduced_2.components_[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(x_reduced_2.components_[0], x_reduced_2.components_[1],c=colors,s=10)
    legend = ax.legend( ['Не ИАД'])
    ax.add_artist(legend)
    plt.show()

def T_SNE(data,lbls):

    model=TSNE(n_components=2,learning_rate='auto',init='random')
    x_reduced_2= model.fit_transform(data)
    print('Первая компонента:', x_reduced_2[:,0])
    print(len(x_reduced_2[:,0]))
    print('Вторая компонента:', x_reduced_2[:,1])
    fig=plt.figure()
    ax=Axes3D(fig)
    ax.scatter(x_reduced_2[:,0],x_reduced_2[:,1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

stemmer = SnowballStemmer("russian")
data1 = pd.read_csv('D:\\pyprojects\\PAD\\IAD_raw.csv')
data2 = pd.read_csv('D:\\pyprojects\\PAD\\IAD_flags_n_words.csv')
texts = data1['annotation']

# Важно: в texts есть значения nan, из-за которых в цикле не получится приводить
# в цикле все буквы к одному регистру: https://pythonru.com/biblioteki/not-a-number-vse-o-nan-pd-5
# удалять nan не стоит, потому что тогда придется удалять соответствующие строки в lbls
# лучше заменить nan на ''

texts = texts.fillna('')
"""for i in range(0, len(texts)):
    texts[i] = texts[i].lower()

spec_chars = string.punctuation + '\n\xa0«»\t—…'

for i in range(0, len(texts)):
    texts[i] = remove_chars_from_text(texts[i], spec_chars)

for i in range(0, len(texts)):
    texts[i] = remove_chars_from_text(texts[i], string.digits)"""

lbls = data2['flag']
data = pd.concat([lbls, texts], axis=1)
russian_stopwords = stopwords.words("russian")

"""
# Стеммер
from nltk.stem.porter import PorterStemmer
import re
porter_stemmer = PorterStemmer()
porter_stemmer = porter_stemmer.stem

# Лемматайзер
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer = lemmatizer.lemmatize
"""
text_test = []
text_train = []

def tokenizer(str_input, result):
    """

    Шо делает: делит строку str_input на отдельные слова, после чего выделяет в них корни и сует в result
    """
    res = []
    words = re.sub(r"[^А-Яа-я]", " ", str_input).lower().split()
    for word in words:
        # здесь условия чтобы сразу отсеивать слова из одной буквы по типу I
        # в примере ниже I'm разбивается на I и m и соответственно не попадают в результат
        # условие можно и убрать, тут как считаете нужным делайте
        if len(word) > 1:
            res.append(stemmer.stem(word))
    result.append(' '.join(res))

x_train, x_test, y_train, y_test = train_test_split(texts, lbls, test_size=0.3)
XXX= []
for text in x_test:
    tokenizer(text, text_test)

for text in x_train:
    tokenizer(text, text_train)
for text in texts:
    tokenizer(text, XXX)

vector = TfidfVectorizer(stop_words=russian_stopwords)
vector.fit(text_train)  # фитим векторайзер на обучающей выборке
Xxx_train = vector.transform(text_train)# трансформируем обучающие данные
Xxx_test = vector.transform(text_test)  # трансформируем тестовые данные
print('Кол-во текстов в тесте', Xxx_test.toarray().shape[0])
print('Кол-во признаков в тесте', Xxx_test.toarray().shape[1])

#rand_forest(Xxx_train,y_train,Xxx_test,y_test)
XXX=vector.transform(XXX)
colors=[]

for i in range(len(lbls)):
    if str(lbls[i])=='False':
        colors.append('b')
    else:
       colors.append('r')

princ_comp_method(XXX.transpose(),colors)
#T_SNE(XXX,lbls)