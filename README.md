# Disaster Response Application Project

## Project Summary

#### Part 1: ETL Pipeline and Training ML Classification Algorithm

In part 1, raw text data provided by Figure Eight is extracted from messages sent during a disaster. These messages are cleaned and restuctured in an ETL pipeline and stored in an SQL database. The data is then read in from the SQL database, prepared and fit to a machine learning model. That model is then saved in a pickle file for use in part 2.

#### Part 2: Web App for Message Classification

In part 2, a web app is built to demonstrate the message classification capabilities of the ML model trained in part 1. New text messages can be entered through the web app interface and are visually classified according to disaster response category. The app also provides several summary visualizations of the original disaster messages.

Here we see a screenshot of the app functioning:

![Alt text](Photos/app1.png?raw=true "Title")

![Alt text](Photos/app2.png?raw=true "Title")

## Project File Descriptions

#### /data folder

1. process_data.py - Python script for ETL pipeline
2. disaster_categories.csv - raw data from Figure Eight
3. disaster_messages.csv - raw data from Figure Eight
4. DisasterResponse.db - SQL database created as part of ETL pipeline

#### /models folder

5. train_classifier.py - Python script for training classification model
6. classifier.pkl - After training, the model is stored in this pickle file for use with the web app

#### /app folder

7. run.py - Python script for running the web app

#### /templates folder

8. master.html - html for web app homepage
9. go.html - html for web app message classification function

## How to Use the App:
1. Clone the repository to your local machine.

Optional:

2. Run the following commands in the project's root directory to set up a new database and model.

    - To run the ETL pipeline that cleans the raw data and stores it in a new database DisasterResponse.db
    
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run the ML pipeline that trains a new classifier and saves it in classifier.pkl
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Required:

3. Run the following command in the app's directory to run your web app.

    `python run.py`

4. Go to http://0.0.0.0:3001/ or http://localhost:3001/ to interact with the web app. 

## Project Challenges

Text messages could be classified into one or more of 36 categories for disaster relief response. One of the main challenges in building this app was how to train the machine learning model for classification of categories with rare and/or severely consequential events such as people in need of food, or children missing. These factors play a major role in whether the classifier should prioritize recall or precision for various categories.

Another major factor in determining the type of errors we prefer the classifier to make is the availability of support resources for various categories of needs.

For this project, the model was optimized in terms of a weighted-f1 score evaluation. However, with more information about the available support resources a more situationally tailored model could be built.



## References and Acknowledgements

- Data used for training the classification model was provided by Figure Eight.
- The backend template for the web app is built with flask and was provided as part of the Udacity Data Scientist Nanodegree program.
