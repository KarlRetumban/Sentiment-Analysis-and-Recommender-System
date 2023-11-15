# Sentiment Analysis and Recommender System 
### Description of the Use Case
We analyze the Renttherunway Clothing Fit data. The dataset contains information about rented attires for certain occassions. It also includes the ratings given by the renters. The user feedback or user review is also included in the data as well as information about the user and the attire.

We conduct exploratory data analysis providing descriptive statistics and data visualizatiuon so we can have a good idea of the data and identify patterns and relationships among the variables.

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


### Data Preparation
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


**We get the Top 3 attires being rented on each event.** 

Note that only the Top 5 reasons for renting were considered. Below are the summary.

* **Wedding**: dress, gown, sheath
* **Formal Affair**: gown, dress, sheath
* **Party**: dress, sheath, jumpsuit
* **Everyday**: dress, top, jacket
* **Work**: dress, sheath, top


~~~ python
# Input rented_for options: wedding,formal affair, party, everyday, work  (Note: The top 5 reason for renting were only considered)
event_ = input('Input event: ')

event = renttherunway_new[renttherunway_new['rented_for']==event_]
dress_event = event.groupby(['rented_for','category'])['user_id'].aggregate('count').sort_values(ascending=False).head(3)

print(dress_event) 
~~~


### Data Visualization

**Number of attire rents per year**
The number of rents per year has been steadiy increasing from below 10,000 rents in 2011-2013 up to over 50,000 rents in 2017.

![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/linechart_yearlyrents.PNG)

~~~ python
#Number of reviews per year
import matplotlib as plt

line = renttherunway_new.groupby(['review_year'])['user_id'].count().plot(kind='line')
line.set_xlabel('Year')
line.set_ylabel('Number of rents')
line.set_title('Number of rents per year')
~~~


**Yearly overall average rating**
* The overall average rating has been relatively high and is alway above 8.8 level. This implies good customer satisfaction.
* The lowest overall average rating happened in 2013, but has been increasing every year, except in 2017 with a slight dip.

![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/linechart_yearlyratings.PNG)

~~~ python
# Linechart of overall average rating plotted overtime
line = renttherunway_new.groupby(['review_year'])['rating'].mean().plot(kind='line')
line.set_xlabel('Year')
line.set_ylabel('Average Rating')
line.set_title('Average Rating per year')
line.set_ylim([8.5, 9.5])
~~~


**Top 5 rented attire: Yearly Overall Average Rating**
* Throughout the years, the gown is the attire that has the highest yearly average rating except in 2018 when it was surpassed by sheath.
* The shift has the lowest yearly average rating in most years. It started as the attire with the highest rating in 2011, but continously declined thoughout the years.

![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/linechart_yearlyratings_attires.PNG)


~~~ python
# Linechart: Yearly average rating per attire
import seaborn as sns

plot = sns.lineplot(data=attire_top5, x="review_year", y="rating", hue="category", style="category", ci=None)
sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
plot.set_xlabel('Year')
plot.set_ylabel('Average Rating')
plot.set_title('Top 5 rented attire: Average Rating')
plot.set_ylim([8.2, 10.2])
~~~



#### Sentimental Analysis
We analyze the customers feedback or review using Sentiment Analysis. Sentiment Analysis is a natural language processing technique that analyzes and identifies the "mood" or sentiment in a text. With Sentiment Analysis, we can determine the kind of customer satisfaction, whether it is positive, negative or neutral, out of their feedback reviews. We also produced a worldcloud in order to visualize the customer feedback and see the dominant words used by the users in writing their feedback reviews.

We subset the data. We only get 1000 rows for runtime purposes. (edit and rerun the whole data)


~~~ python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()

# This will extract the sentiment score on each record
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["review_text"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["review_text"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["review_text"]]
data["Compound"] = [sentiments.polarity_scores(i)["compound"] for i in data["review_text"]]
data = data[["user_id","category","rented_for","fit","body_type","age","size","rating","review_text","review_summary","review_year",
             "Positive", "Negative", "Neutral","Compound"]]


# We tag the sentiment scores whether it is positive, neutral or negative using the Polarity Scores thresholds below.
# This is a recoding function for sentiment tagging
def OverallSentiment(scores):
    if scores <= -0.05:
        return "Negative"
    elif -0.05 < scores < 0.05:
        return "Neutral"
    elif  scores >= 0.05:
        return "Positive"
    
# Apply the recoding function on the Polarity Score Compound
data["Overall_Sentiment"] = data["Compound"].apply(OverallSentiment)

~~~


![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/polarityscores.PNG)



#### We produce charts using the Sentiments Taggings.
This will give us insights about the overall sentiments across various variables. We produce barchart over time, on fit feedback and rating.


**Overall sentiments per year**
* Across the years, from 2011 to 2017, the overall sentiment of customers has been positive.
* This implies that an overwhelming majority of customer have positive feedback from the attire they rented.
* There were also some negative and neutral customer feedback, but on all years, customers have an overall positive or satisfied feedback to the rented products.


![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/sa_yearlysentiments.PNG)



**Overall sentiment across ratings**
We determine whether the overall sentiment agrees with the rating given by customers. This will also serve as an assessment to the reliability of our Sentiment Analysis. We provide a barchart of the count of sentiments versus the ratings.

* Across the ratings, from 2 to 10, the overall sentiment of customers has been positive.
* For the ratings 6 to 10, it is expected to have a Positive orverall sentiments. But for ratings 2 to 4, the sentiment is still majority positive.
* There is a bit of conflict in the customer ratings and customer sentiment as shown by the green bar for ratings 2 to 4.
* We expect this to be dominated by negative or neutral sentiments.
* One thing we can conclude is that the customers are somewhat conservative in giving negative feedback or reviews and would still provide generally positive reviews despite giving low ratings.
  

![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/sa_acrossratings.PNG)



**Overall sentiment across Customer Fit Feedback**
We determine whether the overall sentiment of customers are aligned with fit feedback. We provide a barchart of the count of sentiments versus customer fit feedback.

* Across customer fit feedback, the customer sentiments is still majority positive.
* Even for product misfittings, the majority of sentiments is still overwhelmingly positive.


![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/sa_byfitfeedback.PNG)



**We produce a Wordcloud**
This visualization gives us an idea about the dominant words used in feedback reviews. The bigger the word, the more occurence of that word.

* The words dress is the biggest word. This is expected since this is also the most rented attire.
* There are a lot 'big sized' positive words like great, beautiful, loved, comfotable, and perfect. And this dominant occurence of positive words are in agreement with our Sentiment Analysis which has majority positive sentiments.


![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/wordcloud.PNG)


~~~ python
#Wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

text2 = " ".join(review_text for review_text in renttherunway_wcloud.review_text)
wordcloud = WordCloud(max_font_size=60, max_words=100000, background_color="white").generate(text2)

#Plotting the image with axis off as we don’t want axis ticks in our image.
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
~~~



#### Recommender System
We create a Recommender System using the attires rented by customers and the given ratings. The recommender system uses Collaborative Filtering using SVD Model(Singular Value Decomposition). The variables to be used are user_id, category and ratings.

We only consider the repeat customers with at least 10 rents so we have more relibale data to be fed into the Recommender System. We then provide a recommendation on the next most likely items or attire category the customer will avail in the future. The Top 5 attire recommended will be extracted.



![alt text](https://github.com/KarlRetumban/SampMG_SA_RS/blob/main/images/recommender_system.PNG)



~~~ python
# We only consider the repeat customer data with at least 10 rents availed.
data_10 = renttherunway_new['user_id'].value_counts()
data_10 = data_10.reset_index()
data_10 = data_10[data_10['user_id']>10]
data_10.rename(columns={"user_id":"count", "index":"user_id"}, inplace=True)

# Merge with the original data to get raw data details of the users
data_reco = pd.merge(left=data_10, right=renttherunway_new, how='left', on='user_id')

# We select the variables needed for building a recommender system
data_reco = data_reco[["user_id", "category","rating"]]

# We use the Surprise library in implementing our Recommender System
# We also import the SVD module to be used in building the model

from surprise import Reader, Dataset, SVD
from surprise.model_selection.validation import cross_validate

# Create a surprise Dataset object
reader = Reader()
data_r = Dataset.load_from_df(data_reco[["user_id", "category","rating"]], reader)
# SVD Algo
svd = SVD()

# We run a 5-fold cross-validation
cross_validate(svd, data_r, measures=['RMSE', 'MAE'], cv=10, verbose=True)

#Train the recommendation model
trainset = data_r.build_full_trainset()
svd.fit(trainset)

# Create a dataframe for the attire category list 
rent_category = data_reco.groupby(["category"])["user_id"].count()
rent_category_list = rent_category.reset_index()
rent_category_list = rent_category_list["category"]
rent_category_list = rent_category_list.reset_index()
rent_category_list

# Recommend the most likely attire the customer will rent

# Copy the attire list
attires = rent_category_list.copy()

# Get the User ID of Customer for recommendation. Sample User ID: 691468
userID = input('Select User ID from the data. \n \tInput the User ID: ')

# Recommend the attires the customer will most likely avail
attires['Estimate_Score'] = attires['category'].apply(lambda x: svd.predict(userID, x).est)
attires_reco = attires.sort_values(by=['Estimate_Score'], ascending=False)

# Rename the 'category' (recommended) to 'Recommended Attire'
attires_reco.rename(columns={"category":"Recommended Attire"}, inplace=True)
attires_reco.head(5)
~~~
