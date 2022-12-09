# kaggle datasets list elon masks tweets
# Documentation: https://www.kaggle.com/datasets/marta99/elon-musks-tweets-dataset-2022?select=cleandata.csv
# Open in DataFrame
import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv('rawdata.csv')

# clean rawdata
# delete all retweets of Elon in dataset using regex with @ pattern
idx = df.Tweets.str.contains(r'@', flags=re.IGNORECASE, regex=True, na=False)
df = df[~idx]

# delete all the tweets with URLs
idx = df.Tweets.str.contains(r'https://', flags=re.IGNORECASE, regex=True, na=False)
df = df[~idx]

# download vader_lexicon to assess the tweets sentiments (it is a lexicon which assigns a positive, neutral
# and negative sentiment values to specific words
nltk.download('vader_lexicon')

# Use tqdm to show computation progress
# tqdm.pandas()

# assign a SentimentIntensityAnalyzer function
sentiment = SentimentIntensityAnalyzer()

# here we compute .polarity_scores for each tweet in the dataframe and separate each polarity value in a list
Tweets = df.Tweets
neg, pos, neu, compound = [], [], [], []
for i in Tweets:
    neg.append(sentiment.polarity_scores(i)['neg'])
    neu.append(sentiment.polarity_scores(i)['neu'])
    pos.append(sentiment.polarity_scores(i)['pos'])
    compound.append(sentiment.polarity_scores(i)['compound'])


# here we add previously calculated sentiment scores to the initial DataFrame
df['neg'] = neg
df['neu'] = neu
df['pos'] = pos
df['compound'] = compound

# Print 10 most negative Reviews
# here we sort the dataframe by the negative sentiment score in descending order and choose the first 10
# the first line enables to see complete tweets
pd.set_option('display.max_colwidth', None)
df_neg = df.sort_values(by=['neg'], ascending=False)
df_neg = df_neg.head(10)
print(f"The most negative tweets are: {df_neg.Tweets}")

# Print 10 most positive Reviews
# here we sort the dataframe by the positive sentiment score in descending order and choose the first 10
df_pos = df.sort_values(by=['pos'], ascending=False)
df_pos = df_pos.head(10)
print(f"The most positive tweets are: {df_pos.Tweets}")

# here we add a new column - binary sentiment score : 1 if compound value bigger than 0 (positive), else 0 (negative).
df['Sentiment'] = df['compound'].apply(lambda x: 1 if x >= 0 else 0)

# here we measure correlation between Sentiment score  and quantity of likes on the tweet
corr_sentVSlikes = df['Sentiment'].corr(df['Likes'])
print(f"Correlation between Sentiment score  and quantity of likes on the tweet is: {corr_sentVSlikes} ")

# Measure correlation between Sentiment and Likes
corr_sentVSRetweets = df['Sentiment'].corr(df['Retweets'])
print(f"Correlation between Sentiment score of the tweet and quantity of retweets is: {corr_sentVSRetweets} ")


# create the correlation matrix between calculated sentiment scores and likes and retweets
columns = ["Likes", "Retweets", "pos", "neg"]
corr_matrix = df[columns].corr()
plt.title("Correlation matrix")
sn.heatmap(corr_matrix, annot=True)

# let's figure something out
plt.figure()
plt.title("Average sentiment scale per month in 2022")
df['Date'] = pd.to_datetime(df['Date'])
df.groupby(df.Date.dt.month)['Sentiment'].mean().plot.bar(figsize=(30, 5), color='blue')
ticklabels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October']
ticks = [i for i in range(0, 10)]
plt.xticks(ticks, ticklabels, fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0.7, 1)

plt.figure()
plt.title("Likes evolution in 2022")
df.groupby(df.Date.dt.month)['Likes'].mean().plot.bar(figsize=(30,5), color='red')
plt.xticks(ticks, ticklabels, fontsize=14)
plt.yticks(fontsize=14)

plt.figure()
plt.title("Evolution of negative and positive scale in 2022")
df.groupby(df.Date.dt.month)['neg'].mean().plot(figsize=(30, 5), color='red', label='Negativity')
df.groupby(df.Date.dt.month)['pos'].mean().plot(figsize=(30, 5), color='blue', label='Positivity')
ticks = [i for i in range(1, 11)]
plt.xticks(ticks, ticklabels, fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.legend(fontsize=14)

plt.show()
