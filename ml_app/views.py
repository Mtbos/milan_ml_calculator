# ml_app/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
from .forms import SignUpForm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from django.shortcuts import render
from .forms import CSVUploadForm
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import requires_csrf_token






@login_required
@csrf_protect
def upload_csv(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            target_column = form.cleaned_data['target_column']
            model_choice = form.cleaned_data['model_choice']
            
            
            # Reading CSV file
            data = pd.read_csv(csv_file)
             # Drop columns with all NaN values
             # Separate numeric and categorical columns
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

             # Encode categorical columns
            for col in categorical_cols:
                if data[col].nunique() < 10:  # Assuming < 10 unique values for Label Encoding
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])  # Apply Label Encoding
                else:
                    data = pd.get_dummies(data, columns=[col], drop_first=True)  # One-Hot Encoding

            # Prepare data
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize the model based on selection and calculate score
            if model_choice == 'Linear Regression':
                try:
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    score = r2_score(y_test, predictions)  # R² score for regression
                    score_type = "R² Score"
                except Exception as e:
                # Catch any other unexpected errors
                    return render(request, 'ml_app/error.html', {'error': f"An unexpected error occurred: {e}"})

                
            elif model_choice == 'Decision Tree':
                try:
                    model = DecisionTreeClassifier()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    score = accuracy_score(y_test, predictions)  # Accuracy score for classification
                    score_type = "Accuracy Score"
                except Exception as e:
                    
                # Catch any other unexpected errors
                    return render(request, 'ml_app/error.html', {'error': f"An unexpected error occurred: {e}"})

                
            elif model_choice == 'Logistic Regression':
                try:
                    model = LogisticRegression()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    score = r2_score(y_test, predictions)  # Accuracy score for classification  
                    score_type = "R² Score"
                except Exception as e:
                # Catch any other unexpected errors
                    return render(request, 'ml_app/error.html', {'error': f"An unexpected error occurred: {e}"})


            elif model_choice == 'RandomForestClassifier':
                try:
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    score = accuracy_score(y_test, predictions)
                    score_type = "Accuracy Score"
                except Exception as e:
                # Catch any other unexpected errors
                    return render(request, 'ml_app/error.html', {'error': f"An unexpected error occurred: {e}"})

                
            else:
                return redirect(request, 'ml_app/error', {'error': 'Invalid model selected.'})
            
            # Prepare the response
            result = {
                'predictions': predictions[:5],  # Show sample predictions
                'model': model_choice,
                'target_column': target_column,
                'score': score,
                'score_type': score_type
            }
            
            return render(request, 'ml_app/result.html', result)
    else:
        form = CSVUploadForm()
    
    return render(request, 'ml_app/upload.html', {'form': form})

# ml_app/views.py




def home(request):
    return render(request, 'ml_app/home.html')  # Public homepage

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
        #return render(request, 'ml_app/registration/login_error.html')# Log in the user after sign-up
        return render(request, 'ml_app/signup_error.html')  # Redirect to homepage or dashboard
    else:
        form = SignUpForm()
    return render(request, 'ml_app/signup.html', {'form': form, 'project_info': '''
                                                  THIS PROJECT HELP YOU FIND OUT THE SOLUTION FOR THE MACHINE LEARNING MODEL'''})

def logout_view(request):
    logout(request)
    return redirect('login') 


            