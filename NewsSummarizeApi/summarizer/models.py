from django.db import models

# Create your models here.
# Create your models here.


class Summary(models.Model):
    title = models.CharField(max_length=50)
    summarized_text = models.CharField(max_length=10000)


def _str_(self):
    return self.summary

