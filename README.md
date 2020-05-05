# ANLY 502 - Final Project 

## Modeling and Sentiment Analysis of Amazon Review and Twitch Stream Data

### Team: 5 AM
### Teammates: Hao Liu, Jianhao Ji, Qian Yi, Jingyu Zhang


## Executive Summary

(1) Download and unzip Amazon Review data by using hadoop. Download the Twitch game data on the local machine, then unzip the .rar format data. Save the datasets in s3 bucket. 
(2) Implement SparkSQL techniques to extract critical data, then perform data cleaning and data preprocessing. 
(3) Build NLP models with Logistic Regression, TFIDF, Naive Bayes, Random Forest and Vader methods. Then compare model accuracies. 
(4) Create Visualizations to better understand game trends and overall ratings. Generate insights based on the results. 

## Introduction

Over the past 20 years, video games have begun to shift in purpose and use. Up until 1994, games were not much more than time killers and competitive electronic games. The most vocal of this camp is the famous film critic Roger Ebert. He publicly battled against the perception of video games becoming art from 2006. But nowadays, with the flourish of gaming companies and development of gaming peripheral industries(figure 1), more and more people consider video games as a new form of art as well as one essential in daily lives.

# Figure 1


Just like other forms of arts, there are a thousand Hamlets in a thousand people's eyes, one's attitude towards games could be hugely affected by their preference of game types, their culture, their background etc. Since the centers of games are always the players, their reviews and tendencies matter. Here comes one crucial problem: how to rate a game by its review and how to judge whether a game is trending.

In our project, we are going to focus on games’ reviews on Amazon and streaming tendencies on Twitch. We all know that the Amazon review section contains ‘Overall Rate’ and ‘Review’ where customers can both express their opinions on games they purchased and give them a score. We are going to implement sentiment analysis on current Amazon review datasets then conclude a reliable model to convert players’ review to numeric rates. Besides, we will analyze Twitch Streaming datasets and figure out the trending video games.

## Dataset
#### Dataset one: Amazon Product Review Dataset
This dataset includes review ID, product ID, reviewer name, overall rating, summary, helpful vote, and review time. It includes all product categories from Amazon Fashion to Video Games. 
Unzip size: 54GB 

#### Dataset two: Twitch Streaming Dataset
This dataset includes Stream ID, current views, stream created time, game name, broadcaster ID, delay setting, follower number, partner status, language etc. 
Unzip size: 6.9GB


## Code files
(1) Data_cleaning.ipynb 
(2) EDA twitch and Amazon_game.ipynb 
(3) Logistic Regression.ipynb
(4) TFIDF_cross_validation.ipynb 
(5) NaiveBayes RandomForest.ipynb
(6) Vader.ipynb



## Methods section
### Data Preprocessing and Analysis

#### Data Preprocessing:
The raw data from Amazon review dataset contains several unrelated features including(figure 2): 

# Figure 2

And different columns have various data type(figure 3):

# Figure 3

Besides the all-product dataset, we have another dataset named ‘Video Game’, containing all the game products on Amazon with relatively smaller size, the only difference between those two datasets is the Amazon datasets have full reviews while smaller one only contains part of reviews.

After investigating the raw data, we noticed that the asin is the unique ID for each game, To access all game products in the Amazon datasets, we firstly retrieved the distinct asin code from the Video Game dataset then accessed the full review from the Amazon datasets. 

At this point, there were many unrelated features such as column ‘ helpful’, column ‘reviewerID’ and ‘reviewerName’. Before implementing models, we dropped all the unrelated columns(figure 4):


# Figure 4



Simultaneously, we also cleaned the Twitch Streaming datasets. The raw datasets were in .txt format and separated in many small files so the first thing in the data cleaning step is merging the small datasets, then we raw Twitch Streaming dataset looks like(figure 5):


# Figure 5
























