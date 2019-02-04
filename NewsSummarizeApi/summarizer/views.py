from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
# from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
# from . models import Summary
from . serializers import SummarySerializer

from rest_framework import views

from nltk.corpus import wordnet
import re

from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from nltk.tokenize import RegexpTokenizer
from nltk import sent_tokenize

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
# Create your views here.


# class SummarizeView(APIView):
#
#     # def get(self,request):
#     #     todo=Todo.objects.all()
#     #     serializer=TodoSerializer(todo,many=True)
#     #     return Response(serializer.data)
#     def post(self, request):
#         Summary.objects.create(
#             title=request.data.get('title'),
#             summarized_text=request.data.get('text'))
#         return HttpResponse(status=201)


class SummarizeView(views.APIView):
    model = RandomForestClassifier(n_estimators=300, max_depth=150, n_jobs=1)
    vect = TfidfVectorizer(stop_words='english', min_df=2)

    def post(self, request):
        averageTime = request.data.get('average_time_in_seconds')
        words = request.data.get('personal_interests')
        text = request.data.get('text')

        summary = self.summary_generator(text, averageTime)
        # print("title:"+title)

        features = self.corpus_generator(words)
        sentences = self.sentence_extracter(features, text)
        # generalSummarySentences = sent_tokenize(summary)
        personalized_summary = self.summary_merge(summary, sentences, text)
        self.ModelTrainer()
        category = self.ContentClassifier(text)
        yourdata= [{"title":"hkuh", "general_summary": summary, "personalized_summary": personalized_summary, "category": category}]
        results = SummarySerializer(yourdata, many=True).data
        return Response(results)

    # ============================================

    def corpus_generator(self,*features):
        synonyms = []
        for feature in features[0]:
            feature = feature.strip()
            synonyms.append(feature)
            feature = str(feature)
            synsets = wordnet.synsets(feature)

            for synset in synsets:
                for hypo in synset.hyponyms():
                    for lem in hypo.lemmas():
                        synonyms.append(lem.name())
        synonyms = (set(synonyms))
        return synonyms

    def sentence_extracter(self,features, text):
        sent = []
        for hyp in features:
            ext = re.findall(r"([^.]*?" + hyp + "[^.]*\.)", str(text))
            if (ext != []):
                sent.extend(ext)
        return sent

    def summary_ratio(self,averageTime, text):
        tokenizer = RegexpTokenizer("[\w']+")
        word = tokenizer.tokenize(text)
        count = len(word)
        if not averageTime:
            averageTime = 60
        rat = (float)((200 * averageTime / 60) / count)
        return rat

    def summary_generator(self, text, averageTime):
        # title, text = get_only_text(url)
        rat = (float)(self.summary_ratio(averageTime, text))

        sum = summarize(str(text), ratio=rat)
        summary = sum.rstrip().replace("\n", " ")
        # return title,summary
        return summary

    def summary_merge(self, generalSummary, sentences, text):
        generalSummarySentences = sent_tokenize(generalSummary)
        generalSummarySentsWithoutQuotos = [item.replace('"', '') for item in generalSummarySentences]
        generalSummarySentsWithoutQuotos = [item.replace('/', '') for item in generalSummarySentences]
        # remove quotes ans slashes from text
        textSentences = sent_tokenize(text)
        textSentsWithoutQuotes = textSentences

        for sent in sentences:
            sentWithoutQuotes = sent
            sentWithoutQuotes = sentWithoutQuotes.lstrip()
            sentWithoutQuotes = sentWithoutQuotes.rstrip()

            if not sentWithoutQuotes in generalSummarySentsWithoutQuotos:
                index = textSentsWithoutQuotes.index(sentWithoutQuotes)
                if index == 0:
                    generalSummarySentences.insert(0, sent)
                else:
                    for index in range(0):
                        previousSent = textSentences[index - 1]
                        if previousSent in generalSummarySentsWithoutQuotos:
                            previousIndex = generalSummarySentsWithoutQuotos.index(previousSent)
                            generalSummarySentences.insert(previousIndex + 1, sent)
                            break
                        else:
                            index = index - 1

        return generalSummarySentences


    def clean_str(self,string):
        """
        Tokenization/string cleaning for datasets.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"\'s", "", string)
        string = re.sub(r"\'ve", "", string)
        string = re.sub(r"n\'t", "", string)
        string = re.sub(r"\'re", "", string)
        string = re.sub(r"\'d", "", string)
        string = re.sub(r"\'ll", "", string)
        string = re.sub(r",", "", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", "", string)
        string = re.sub(r"\)", "", string)
        string = re.sub(r"\?", "", string)
        string = re.sub(r"'", "", string)
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"[0-9]\w+|[0-9]", "", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    # def vectorization(self):
    #     vect = TfidfVectorizer(stop_words='english', min_df=2)
    #     return vect

    def ModelTrainer(self):
        data = pd.read_csv('D:/Projects/Final year/NewsSummarizeApi/summarizer/dataset.csv', encoding="ISO-8859-1")
        x = data['news'].tolist()
        y = data['type'].tolist()

        for index, value in enumerate(x):
            x[index] = ' '.join([Word(word).lemmatize() for word in self.clean_str(value).split()])

        # vect = TfidfVectorizer(stop_words='english', min_df=2)
        Y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.20, random_state=42)

        print("train:", X_train)
        print("test :", X_test)

        X_train = self.vect.fit_transform(X_train)
        X_test = self.vect.transform(X_test)


        # model = RandomForestClassifier(n_estimators=300, max_depth=150, n_jobs=1)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print(y_pred[-1])
        c_mat = confusion_matrix(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        print("Confusion Matrix:\n", c_mat)
        print("\nKappa: ", kappa)
        print("\nAccuracy: ", acc)

    def ContentClassifier(self, content):
        x_new = ' '.join([Word(word).lemmatize() for word in self.clean_str(content).split()])
        X_new = self.vect.transform([x_new])
        new_y = self.model.predict(X_new)
        return new_y


    # averageTime = int(input("\n Enter average time of user in seconds: "))
    # word = input("Enter personal interests separated by comma ")
    # word = word.split(",")
    #
    # # word =input("enter word:")
    # # title,summary = summary_generator(url,averageTime)
    # text = input("Enter text: ")
    # summary = summary_generator(text, averageTime)
    # # print("title:"+title)
    #
    # features = corpus_generator(word)
    # print(corpus_generator(word))
    # sentences = sentence_extracter(features, text)
    # generalSummarySentences = sent_tokenize(summary)
    # print(generalSummarySentences)
    # print(summary_merge(summary, sentences, text))
