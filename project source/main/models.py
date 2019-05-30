from django.db import models
import os
import re
from datetime import datetime
def upload_image_directory_path(instance, filename):
    filename, file_extension = os.path.splitext(filename)
    return 'plant/'+re.sub('[-:. ]','',str(datetime.today()))+file_extension


class Plant(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(null = True, blank = True)
    createdAt = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
    
class UploadFile(models.Model):
    name = models.CharField(max_length=30, null = True, blank = True)
    image = models.ImageField(upload_to = upload_image_directory_path)
    edited = models.ImageField(upload_to = upload_image_directory_path, null = True, blank = True)
    segmented = models.ImageField(upload_to = upload_image_directory_path, null = True, blank = True)
    def __str__(self):
        return self.name

class Disease(models.Model):
    name = models.CharField(max_length=100)
    plant = models.ForeignKey('Plant', on_delete=models.CASCADE , related_name="diseases")
    symptoms = models.TextField(null = True, blank = True)
    cause = models.TextField(null = True, blank = True)
    comments = models.TextField(null = True, blank = True)
    management = models.TextField(null = True, blank = True)
    image = models.CharField(max_length = 100, null = True, blank = True)

    def __str__(self):
        return self.name
