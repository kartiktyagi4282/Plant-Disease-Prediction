from django import forms
from .models import *

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = UploadFile
        fields = '__all__'