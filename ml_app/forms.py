from django import forms
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm


class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(label="Upload CSV File")
    target_column = forms.CharField(max_length=100, label="Target Column")
    model_choice = forms.ChoiceField(choices=[('Linear Regression', 'Linear Regression'), ('Decision Tree', 'Decision Tree'), ('Logistic Regression', 'Logistic Regression'),
                                              ('RandomForestClassifier', 'RandomForestClassifier')])


# ml_app/forms.py



class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
