#### Connect Python to MongoDB Compass
* We import data from MongoDB to Python.
* We then analyze the Renttherunway Clothing Review data.


~~~ python
import pymongo

# Connect to MongoDB Compass.
client = pymongo.MongoClient('localhost', 27017)

# Get the Reviews database.
db = client['Reviews']

# Get the rentedclothing_reviews collection.
collection = db['rentedclothing_reviews']

# Show the first few documents in the rentedclothing_reviews collection.
for document in collection.find().limit(2):
    print(document)
~~~

Below is the JSON data output.

![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/JSON.PNG)


#### Convert JSON to dataframe

~~~ python
import pandas as pd

# Connect to MongoDB Compass.
client = pymongo.MongoClient('localhost', 27017)

# Get the Reviews database.
db = client['Reviews']

# Get the rentedclothing_reviews collection.
collection = db['rentedclothing_reviews']

# Get all documents in the rentedclothing_reviews collection.
rentedclothing_reviews = collection.find()


# Create a DataFrame from the cursor.
renttherunway_df = pd.DataFrame(rentedclothing_reviews)

~~~


Below is the dataframe output.

![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/dataframe.PNG)


#### Data Preparation
The chosen dataset is the Clothing fit data, specifically the RentTheRunway JSON dataset.

The dataset contains information about rented attires for certain occassions. It also includes the ratings given by the renters. The user feedback or user review is also included in the data as well as information about the user and the attire.

We import Pandas library and load the data Renttherunway JSON Data into pandas dataframe.

We examine the data in terms of its size (total variables/columns and observations), actual values, completeness and data types.

The actual checking of the dataset is a very important step in data analysis.




##### Descriptive Analytics
We produce descriptive statistics for each attribute or column for our initial analysis. We also produce some statistics and figures which will give us more insights about the data. Below are some notable insights from the renttherunway dataset.
* In our new dataset, we have 146,381 observations.
* There are 77,347 unique users. This implies that there are several repeat customers renting attires.
* The number of repeat cutomers, i.e., users who rented attires at least twice is 12,564.
* There are 68 tyes of attire being rented (under category varaiable), and the top rented attire is a dress.
* Wedding is the topmost reason for renting (under rented_for variable).
* The average age of renters is 34 years old.
* The average rating is 9.08 with rating range of 2-10. This is a high overall rating which implies high customer satisfaction.




##### Sentimental Analysis



##### Recommender System

