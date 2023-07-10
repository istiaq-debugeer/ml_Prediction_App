import os
import matplotlib
from sklearn.neighbors import NearestNeighbors

matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from django.conf import settings
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing
import base64
from io import BytesIO
from django.http import HttpResponse



def home(request):
    return render(request, 'home.html')
def document(request):
    return render(request, 'documentation.html')
def dashboard(request):
    return render(request, 'dashboard.html')

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
        #Histogram plot
        histogram_plot_path = os.path.join(settings.MEDIA_ROOT, 'histogram_plot.png')
        plt.figure(figsize=(10, 8))
        plt.hist(df[split_column], bins=10)  # Replace 'Column' with the actual column name from your dataset
        plt.title('Histogram Plot')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.savefig(histogram_plot_path)
        plt.close()
        #pie chart
        # pie_chart_path = os.path.join(settings.MEDIA_ROOT, 'pie_chart.png')
        # plt.figure(figsize=(8, 8))
        # df[split_column].value_counts().plot(kind='pie', autopct='%1.1f%%')
        # plt.title('Pie Chart')
        # plt.ylabel('')
        # plt.savefig(pie_chart_path)
        # plt.close()
        #violin chart
        # violin_plot_path = os.path.join(settings.MEDIA_ROOT, 'violin_plot.png')
        # plt.figure(figsize=(10, 8))
        # sns.violinplot(data=df)
        # plt.title('Violin Plot')
        # plt.xlabel('Columns')
        # plt.ylabel('Values')
        # plt.xticks(rotation=45)
        # plt.savefig(violin_plot_path)
        # plt.close()
        #stacked chart
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
            'stacked_bar_data':stacked_bar_data
        }

        return render(request, 'result.html', context)

    return render(request, 'result.html')

def input(request):
    return render(request,'input.html')

def preprocess_data(cellphone_data, cellphone_rating, cellphone_user):
    label_encoder = preprocessing.LabelEncoder()
    for column in cellphone_data.columns:
        if cellphone_data[column].dtype == 'object':
            cellphone_data[column] = label_encoder.fit_transform(cellphone_data[column])

    for column in cellphone_rating.columns:
        if cellphone_rating[column].dtype == 'object':
            cellphone_rating[column] = label_encoder.fit_transform(cellphone_rating[column])

    for column in cellphone_user.columns:
        if cellphone_user[column].dtype == 'object':
            cellphone_user[column] = label_encoder.fit_transform(cellphone_user[column])

    return cellphone_data, cellphone_rating, cellphone_user



def recommend_cellphones(request):
    if request.method == 'POST':
        cellphonedata_file = request.FILES['cellphonedata']
        cellphonerating_file = request.FILES['cellphonerating']
        cellphoneUser_file = request.FILES['cellphoneUser']

        cellphone_data = pd.read_csv(cellphonedata_file)
        cellphone_rating = pd.read_csv(cellphonerating_file)
        cellphone_user = pd.read_csv(cellphoneUser_file)

        cellphone_data, cellphone_rating, cellphone_user = preprocess_data(cellphone_data, cellphone_rating,
                                                                           cellphone_user)

        merged_data = pd.merge(cellphone_rating, cellphone_data, on='cellphone_id')
        merged_data = pd.merge(merged_data, cellphone_user, on='user_id')

        X_train, X_test, y_train, y_test = train_test_split(merged_data[['user_id', 'cellphone_id', 'rating']],
                                                            merged_data['rating'], test_size=0.2, random_state=42)

        model = NearestNeighbors(n_neighbors=2, algorithm='auto')
        model.fit(X_train[['user_id', 'rating']])

        new_user_id = 1001
        new_user_ratings = pd.DataFrame({'user_id': [new_user_id], 'rating': [5]})
        distances, indices = model.kneighbors(new_user_ratings)

        similar_users = X_train.iloc[indices[0]]['user_id']
        recommended_cellphones = merged_data[merged_data['user_id'].isin(similar_users)]['cellphone_id'].unique()

        recommended_cellphones_str = ', '.join(str(cellphone) for cellphone in recommended_cellphones)

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
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=feature, y='rating', data=merged_data)
            image = plot_to_base64(plt)
            feature_images.append((feature, image))
            plt.close()

        context = {
            'recommended_cellphones': recommended_cellphones_str,
            'heatmap_image': heatmap_image,
            'feature_images': feature_images,
        }



        return render(request, 'recomendate.html', context)

    return render(request, 'input.html')


def plot_to_base64(plot):
    buffer = BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return image_base64



