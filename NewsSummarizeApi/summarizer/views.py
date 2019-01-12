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

        yourdata= [{"title":"hkuh", "general_summary": summary, "personalized_summary": personalized_summary}]
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
