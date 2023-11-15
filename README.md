# Sentiment Analysis and Recommender System 
### Description of the Use Case
We analyze the Renttherunway Clothing Fit data. The dataset contains information about rented attires for certain occassions. It also includes the ratings given by the renters. The user feedback or user review is also included in the data as well as information about the user and the attire.

We will apply Sentiment Analysis to the clothing fit review data and determine the customer sentiments regarding the rented attire. We can also determine the level of satisfaction of customers, whether it is positive, negative or neutral out of their feedback reviews. We will also determine the consistensy of sentiments with respect to the ratuings given and check if the two customer feedback align.

We then build a Recommender System that will suggest the next most likely items or attire category the customer will avail in the future using the users historical preference.



#### Connect Python to MongoDB Compass
* We import data from MongoDB to Python.


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
* We examine the data in terms of its size (total variables/columns and observations), actual values, completeness and data types. The actual checking of the dataset is a very important step in data analysis.
* We check the different data types of the columns of the dataset. We then evaluate for missing data. After identifying the missing data, we remove all observations with missing data and proceed with analyzing the observations with complete data.
* We also do transformation of variables by creating a new variable review_year.

_____________________________________________

### Descriptive Analysis
We produce descriptive statistics for each attribute or column for our initial analysis. We also produce some statistics and figures which will give us more insights about the data. Below are some notable insights from the renttherunway dataset.
* In our new dataset, we have **146,381** observations.
* There are **77,347 unique customers**. This implies that there are several repeat customers renting attires.
* The number of **repeat cutomers** is **12,564**, i.e., the cutomers who rented attires at least twice.
* There are **68 types of attire** being rented (under category varaiable), and the **top rented attire** is a **dress**.
* **Wedding** is the topmost reason for renting (under rented_for variable).
* The average age of renters is **34 years old**.
* The **average rating is 9.08** with rating range of 2-10. This is a high overall rating which implies high customer satisfaction.

![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/descriptives.PNG)


**Below are some additional insights.**
We get the topmost rented attires as well as the top reasons for renting.

**Top 5 attires being rented**
1. Dress
2. Gown
3. Sheath
4. Shift
5. Jumpsuit

**Top 5 reasons for renting**
1. Wedding
2. Formal Affair
3. Party
4. Everyday
5. Work

![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/top5.PNG)


We get the Top 3 attires being rented on each event. 
Note that only the Top 5 reasons for renting were considered. Below are the summary.

* Wedding: dress, gown, sheath
* Formal Affair: gown, dress, sheath
* Party: dress, sheath, jumpsuit
* Everyday: dress, top, jacket
* Work: dress, sheath, top


~~~ python
# Input rented_for options: wedding,formal affair, party, everyday, work  (Note: The top 5 reason for renting were only considered)
event_ = input('Input event: ')

event = renttherunway_new[renttherunway_new['rented_for']==event_]
dress_event = event.groupby(['rented_for','category'])['user_id'].aggregate('count').sort_values(ascending=False).head(3)

print(dress_event) 
~~~



##### Sentimental Analysis



##### Recommender System

