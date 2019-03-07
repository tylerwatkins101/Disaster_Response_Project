# Disaster Response Pipeline Project

### Project Summary
The project is broken into 2 parts:
    In part 1 raw data provdied by Figure Eight is extracted from messages sent during a disaster. These messages are passed through an ETL pipeline and stored in an SQL database. The data is then read in from the SQL database, cleaned, and prepared to fit a machine learning model. That model is then saved in a pickle file for use in part 2.
    
    In part 2 a webapp is built that uses the model trained in part 1 to categorize new text messages entered in through the app interface to decide whether the message indicates a need of response. The app also provides several visualizations of the original disaster messages.

### File Descriptions
/data
1. process_data.py
2. disaster_categories.csv
3. disaster_messages.csv
4. DisasterResponse.db

/models
5. train_classifier.py
6. classifier.pkl

/app
7. run.py
    /templates
    8. master.html
    9. go.html

### How to Use the App:
1. Clone the repository to your local machine.
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
