from data_loader import Data
import numpy as np
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import spacy
from sklearn.preprocessing import MinMaxScaler #fixed import
from copy import deepcopy
import argparse
from sklearn.metrics import accuracy_score
import numpy as np
import gensim.downloader as api



models_type = [
    LogisticRegression(),
    MultinomialNB(),
    SVC(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
]

parser = argparse.ArgumentParser(description="NLP hw2")
parser.add_argument(
    '--train_rate',
    type=float,
    default=0.9,
    help="the rate of train_data set used"
)
parser.add_argument(
    '--pad_num',
    type=int,
    default=15,
    help="padding num of a sentence"
)


glove_model = api.load("glove-wiki-gigaword-50")

def get_sentence_vector(sentence,args):
    words = sentence.split()
    vectors = []
    for i in range(min(args.pad_num,len(words))):
        try:
            vectors.append(glove_model[words[i]])
        except KeyError:
            vectors.append(np.zeros_like(glove_model['hello']))
    ans = np.mean(np.array(vectors),axis=0)
    return ans

def ensemble_predict(models, vectors):
    predictions = np.array([model.predict(vectors) for model in models])
    ensemble_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x+1))-1, axis=0, arr=predictions)
    return ensemble_predictions


def train(train_features:list,train_labels:list,args,batch_num):
    # Create the ensemble model using VotingClassifier
    models_list = []
    for idx, train_feature in enumerate(train_features):
        train_feature = np.array(train_feature)
        train_len = int(batch_num*args.train_rate)
        train_feature = train_feature[:train_len]
        train_label = train_labels[idx][:train_len]
        models = deepcopy(models_type)
        for model in models:
            model.fit(train_feature,train_label)
        models_list.append(models)
    return models_list

def validate(models_list,train_features,train_labels,args,batch_num):
    for idx, train_feature in enumerate(train_features):
        train_len = int(batch_num*args.train_rate)
        valid_feature = train_feature[train_len:]
        valid_label = train_labels[idx][train_len:]
        for models in models_list:
            for model in models:
                pred = model.predict(valid_feature)
                precision = accuracy_score(pred,valid_label)
                print("model:{}, data: {}, precision: {} ".format(model, idx,precision))

def predict(models_list, test_features):
    predictions = []
    for idx,models in enumerate(models_list):
        test_feature = test_features[idx]
        for model in models:
            predictions.append(model.predict(test_feature))
    predictions = np.array(predictions)
    ensemble_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x+1))-1, axis=0, arr=predictions)
    return ensemble_predictions

def predict_filter(preds, data:Data):
    text = data.texts
    assert len(preds) == len(text)
    for i in range(len(preds)):
        if text[i] == "None":
            preds[i] = 0

def get_feature(data:Data,args):
    context = [get_sentence_vector(i,args) for i in train_data.texts]
    related = [get_sentence_vector(i,args) for i in train_data.relateds]
    ans = []    
    for i in range(len(context)):
        ans.append(np.concatenate((context[i],related[i])))
    return ans

if __name__ == "__main__":
    train_data = Data('train.txt')
    test_data = Data('test.txt',False)
    train_data.extract_dataset()
    test_data.extract_dataset()
    args = parser.parse_args()

    train_features = []
    train_labels = []
    test_features = []
    
    #用original+Tfidf 训transfrom
    # vectorizer = TfidfVectorizer()
    # original_feature = vectorizer.fit_transform(train_data.old_texts)
    # train_feature = vectorizer.transform(train_data.texts).toarray()
    # test_feature = vectorizer.transform(test_data.texts).toarray()
    # train_features.append(train_feature)
    # test_features.append(test_feature)
    # train_labels.append(train_data.labels)
    

    #用预训练词向量+词向量平均进行训练
    # vector_model = spacy.load("en_core_web_sm")
    # train_feature = [vector_model(i).vector for i in train_data.texts]
    # test_feature = [vector_model(i).vector for i in test_data.texts]
    # scaler = MinMaxScaler()
    # train_feature = scaler.fit_transform(train_feature)
    # test_feature = scaler.transform(test_feature)
    # train_features.append(train_feature)
    # test_features.append(test_feature)
    # train_labels.append(train_data.labels)

    #用glove预训练词向量来加载
    
    train_feature = get_feature(train_data,args) 
    test_feature = get_feature(test_data,args)
    scaler = MinMaxScaler()
    train_feature = scaler.fit_transform(train_feature)
    test_feature = scaler.transform(test_feature)
    train_features.append(train_feature)
    test_features.append(test_feature)
    train_labels.append(train_data.labels)

    models_list = train(train_features, train_labels, args,3602)
    validate(models_list,train_features,train_labels,args,3602)
    preds = predict(models_list,test_features)
    # predict_filter(preds,test_data)

    preds_train = predict(models_list,train_features)
    # predict_filter(preds_train,train_data)

    with open('201300032.txt','w') as file:
        for i in range(len(preds)):
            file.write(f"{preds[i]}\n")

    with open('train_result.txt', 'w') as file:
        for i in range(len(test_data.texts)):
            file.write(f"Text: {train_data.texts[i]}\n")
            file.write(f"Target: {train_data.targets[i]}\n")
            file.write(f"Real: {train_data.labels[i]}\n")
            file.write(f"Prediction: {preds_train[i]}\n")
            file.write("\n")

