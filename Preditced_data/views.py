import os
import matplotlib
from sklearn.neighbors import NearestNeighbors

matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing
from io import BytesIO
import base64
from sklearn.preprocessing import LabelEncoder, StandardScaler

from django.conf import settings
from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404


def SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get("name")
        email = request.POST.get("email")
        pass1 = request.POST.get("password1")
        pass2 = request.POST.get("password2")
        if pass1 != pass2:
            return HttpResponse("Your password and confirm password didnt match")
        else:
            my_user = User.objects.create_user(uname, email, pass1)
            my_user.save()

            return redirect('login')

    return render(request, 'sign-up.html')


def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('pass')
        # print(email,password)
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            return HttpResponse("username or password is incorect!!!")
    return render(request, 'sign-in.html')


def LogoutPage(request):
    logout(request)
    return redirect('login')


def home(request):
    return render(request, 'home.html')


@login_required(login_url='login')
def document(request):
    return render(request, 'documentation.html')


@login_required(login_url='login')
def dashboard(request):
    return render(request, 'dashboard.html')


@login_required(login_url='login')
def ml_project_result(request):
    if request.method == 'POST':
        # Get the uploaded dataset file
        dataset = request.FILES['dataset']

        # Load the dataset using pandas
        df = pd.read_csv(dataset)
        df = df.dropna()
        # data = pd.read_csv('/content/travel.csv')
        # df = pd.read_csv(dataset)
        # context = {'data': df.to_html()}
        # return render(request, 'Details.html', context)

        # Change column datatypes if needed
        label_encoder = preprocessing.LabelEncoder()
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = label_encoder.fit_transform(df[column])

        split_column = request.POST['split_column']

        # Split the dataset into features (X) and target variable (y)
        X = df.drop(split_column, axis=1)
        y = df[split_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize classifiers
        classifiers = [
            LogisticRegression(),
            DecisionTreeClassifier(),
            SVC(),
            RandomForestClassifier(),
        ]

        # Store performance metrics
        classifiers_performance = {}

        # Calculate performance metrics for each classifier
        for classifier in classifiers:
            classifier_name = classifier.__class__.__name__
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            classifiers_performance[classifier_name] = [accuracy, precision, recall, f1]

        # Create a DataFrame from the performance dictionary
        performance_df = pd.DataFrame.from_dict(classifiers_performance, orient='index',
                                                columns=['Accuracy', 'Precision', 'Recall', 'f1'])

        # Generate the correlation heatmap plot
        correlation = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True)
        plt.title('Correlation Heatmap')
        heatmap_path = os.path.join(settings.STATIC_ROOT, 'heatmap.png')
        plt.savefig(heatmap_path)
        plt.close()

        # Save the best accuracy bar plot
        bar_plot_path = os.path.join(settings.MEDIA_ROOT, 'bar_plot.png')
        plt.figure(figsize=(10, 8))
        sns.barplot(data=performance_df, x=performance_df.index, y='Accuracy')
        plt.title('Best Accuracy')
        plt.xlabel('Classifier')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=55)
        plt.savefig(bar_plot_path)
        plt.close()

        # Save the scatter matrix plot
        # scatter_matrix_path = os.path.join(settings.MEDIA_ROOT, 'scatter_matrix.png')
        # plt.figure(figsize=(10, 10))
        dot_plot_path = os.path.join(settings.MEDIA_ROOT, 'dot_plot.png')
        plt.figure(figsize=(10, 8))
        sns.stripplot(data=df, jitter=True, alpha=0.5)
        plt.title('Dot Plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xticks(rotation=55)
        plt.savefig(dot_plot_path)
        plt.close()
        # Histogram plot
        histogram_plot_path = os.path.join(settings.MEDIA_ROOT, 'histogram_plot.png')
        plt.figure(figsize=(10, 8))
        plt.hist(df[split_column], bins=10)  # Replace 'Column' with the actual column name from your dataset
        plt.title('Histogram Plot')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.savefig(histogram_plot_path)
        plt.close()
        # pie chart
        # pie_chart_path = os.path.join(settings.MEDIA_ROOT, 'pie_chart.png')
        # plt.figure(figsize=(8, 8))
        # df[split_column].value_counts().plot(kind='pie', autopct='%1.1f%%')
        # plt.title('Pie Chart')
        # plt.ylabel('')
        # plt.savefig(pie_chart_path)
        # plt.close()
        # violin chart
        # violin_plot_path = os.path.join(settings.MEDIA_ROOT, 'violin_plot.png')
        # plt.figure(figsize=(10, 8))
        # sns.violinplot(data=df)
        # plt.title('Violin Plot')
        # plt.xlabel('Columns')
        # plt.ylabel('Values')
        # plt.xticks(rotation=45)
        # plt.savefig(violin_plot_path)
        # plt.close()
        # stacked chart
        stacked_bar_plot_path = os.path.join(settings.MEDIA_ROOT, 'stacked_bar_plot.png')
        plt.figure(figsize=(10, 8))
        colors = ['cyan', 'green', 'blue']  # Replace with your desired colors

        category_counts = df[split_column].value_counts()
        categories = category_counts.index
        counts = category_counts.values
        plt.bar(categories, counts)
        plt.bar(categories, counts, color=colors)
        plt.title('Stacked Bar Plot')
        plt.xlabel(split_column)
        plt.ylabel('Count')
        plt.xticks(rotation=55)
        plt.savefig(stacked_bar_plot_path)
        plt.close()
        # Convert the image files to base64 strings
        with open(heatmap_path, 'rb') as f:
            heatmap_data = base64.b64encode(f.read()).decode('utf-8')

        with open(bar_plot_path, 'rb') as f:
            bar_plot_data = base64.b64encode(f.read()).decode('utf-8')

        # with open(scatter_matrix_path, 'rb') as f:
        #     scatter_matrix_data = base64.b64encode(f.read()).decode('utf-8')
        with open(dot_plot_path, 'rb') as f:
            dot_plot_data = base64.b64encode(f.read()).decode('utf-8')

        with open(histogram_plot_path, 'rb') as f:
            histofram_plot_data = base64.b64encode(f.read()).decode('utf-8')

        with open(stacked_bar_plot_path, 'rb') as f:
            stacked_bar_data = base64.b64encode(f.read()).decode('utf-8')
        # Pass the necessary data to the HTML template
        context = {
            'performance_df': performance_df.to_html(),
            'heatmap_data': heatmap_data,
            'bar_plot_data': bar_plot_data,
            'dot_plot_data': dot_plot_data,
            'histofram_plot_data': histofram_plot_data,
            'stacked_bar_data': stacked_bar_data
        }

        return render(request, 'result.html', context)

    return render(request, 'result.html')


@login_required(login_url='login')
def input(request):
    return render(request, 'input.html')


# Step 1: Data Preprocessing
def preprocess_data(cellphone_data, cellphone_rating, cellphone_user):
    label_encoder = preprocessing.LabelEncoder()
    encoded_values = {}
    for column in cellphone_data.columns:
        if cellphone_data[column].dtype == 'object':
            cellphone_data[column] = label_encoder.fit_transform(cellphone_data[column])
            cellphone_data[column].fillna(cellphone_data[column].mode()[0], inplace=True)
            encoded_values[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    for column in cellphone_rating.columns:
        if cellphone_rating[column].dtype == 'object':
            cellphone_rating[column] = label_encoder.fit_transform(cellphone_rating[column])
            cellphone_rating[column].fillna(cellphone_rating[column].mode()[0], inplace=True)
            encoded_values[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    for column in cellphone_user.columns:
        if cellphone_user[column].dtype == 'object':
            cellphone_user[column] = label_encoder.fit_transform(cellphone_user[column])
            cellphone_user[column].fillna(cellphone_user[column].mode()[0], inplace=True)
            encoded_values[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    return cellphone_data, cellphone_rating, cellphone_user, encoded_values


@login_required(login_url='login')
def recommend_cellphones(request):
    if request.method == 'POST':
        cellphonedata_file = request.FILES['cellphonedata']
        cellphonerating_file = request.FILES['cellphonerating']
        cellphoneUser_file = request.FILES['cellphoneUser']

        cellphone_data = pd.read_csv(cellphonedata_file)

        cellphone_rating = pd.read_csv(cellphonerating_file)
        cellphone_user = pd.read_csv(cellphoneUser_file)

        cellphone_data, cellphone_rating, cellphone_user, encoded_values = preprocess_data(cellphone_data,
                                                                                           cellphone_rating,
                                                                                           cellphone_user)

        merged_data = pd.merge(cellphone_rating, cellphone_data, on='cellphone_id')
        merged_data = pd.merge(merged_data, cellphone_user, on='user_id')

        X_train, X_test, y_train, y_test = train_test_split(merged_data[['user_id', 'cellphone_id', 'rating']],
                                                            merged_data['rating'], test_size=0.2, random_state=42)

        # print("Shape of X_train:", X_train.shape)
        # print("Shape of X_test:", X_test.shape)

        model = NearestNeighbors(n_neighbors=5, algorithm='auto')
        model.fit(X_train[['user_id', 'rating']])

        new_user_id = 1001
        new_user_ratings = pd.DataFrame({'user_id': [new_user_id], 'rating': [5]})
        distances, indices = model.kneighbors(new_user_ratings)

        similar_users = X_train.iloc[indices[0]]['user_id']

        recommended_cellphones = merged_data.loc[merged_data['user_id'].isin(similar_users), 'brand'].astype(
             str).tolist()
        cellphone_id_to_model = dict(zip(merged_data['cellphone_id'].astype(str), merged_data['model'].astype(str)))
        recommended_cellphonesmodel = [cellphone_id_to_model[cellphone_id] for cellphone_id in recommended_cellphones]


        #print("Recommended Cellphonesmodel:", recommended_cellphonesmodel)  # Add this line for debugging

        # if len(recommended_cellphones) == 0:
        #     recommended_cellphones_str = "No recommendations available for the new user at the moment."
        # else:
        #     brand_column_name = 'brand'
        #     recommended_cellphone_names = [get_original_names(encoded_values, brand_column_name, int(cellphone_id)) for
        #                                    cellphone_id in recommended_cellphones]
        #
        #     # Join the list of cellphone names with a comma separator
        #     recommended_cellphones_str = ', '.join(recommended_cellphone_names)

        if len(recommended_cellphonesmodel) == 0:
            recommended_cellphonesmodel_str = "No recommendations available for the new user at the moment."
        else:

            # Join the list of model names with a comma separator

            model_name_column_name = 'model'
            recommended_cellphone_model_names = [
                get_original_names(encoded_values, model_name_column_name, int(cellphone_id)) for
                cellphone_id in recommended_cellphonesmodel]

            # Join the list of model names with a comma separator
            recommended_cellphonesmodel_str = ', '.join(recommended_cellphone_model_names)

        # Rest of your view function code

        # Correlation Heatmap
        correlation = merged_data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True)
        heatmap_image = plot_to_base64(plt)
        plt.close()

        # Individual Feature Plots
        features = ['brand', 'operating system', 'internal memory', 'RAM', 'performance', 'main camera',
                    'selfie camera', 'battery size', 'screen size', 'weight', 'price']
        feature_images = []
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=feature, y='rating', data=merged_data)
            plt.title(f'{feature.capitalize()} vs Rating')

            # Replace the encoded x-axis tick labels with the original names
            plt.xticks(ticks=plt.xticks()[0],
                       labels=[get_original_names(encoded_values, feature, int(tick)) for tick in plt.xticks()[0]])

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            feature_images.append((feature, image_base64))
            plt.close()

        context = {
            'recommended_cellphones_Model': list(zip( recommended_cellphone_model_names)),
            #'recommended_cellphones_Model': recommended_cellphonesmodel_str,

            'heatmap_image': heatmap_image,
            'feature_images': feature_images,
        }

        return render(request, 'recomendate.html', context)

    return render(request, 'input.html')


def get_original_names(encoded_values, column_name, encoded_value):
    if column_name in encoded_values:
        encoded_to_original = encoded_values[column_name]
        return next((name for name, value in encoded_to_original.items() if value == encoded_value), None)
    return encoded_value


def plot_to_base64(plot):
    buffer = BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return image_base64


@login_required(login_url='login')
def index(request):
    Document_file = request.FILES['Document']
    data = pd.read_csv(Document_file)
    context = {'data': data.to_html()}
    return render(request, 'Details.html', context)