# import Serializers as Serializers
from rest_framework import serializers


class SummarySerializer(serializers.Serializer):

        title = serializers.CharField(max_length=100)
        general_summary = serializers.CharField(max_length=10000)
        personalized_summary = serializers.CharField(max_length=10000)
        category = serializers.CharField(max_length=10000)