# ============================================
#======SPOTIFY MUSIC ANALYSIS PROJECT ========
#======== NAME:KAVYA SINGHANIA ===============
# DATASET:SPOTIFY TRACKS DATSET(20000 songs)
#=============================================

# ------ IMPORTS ------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#------ READ DATASET ------
df=pd.read_csv("spotify-tracks-dataset.csv")
df=df.drop(columns=['key','time_signature','speechiness','liveness','Unnamed: 0','Unnamed: 0.1','instrumentalness','mode'])
newdf=df.drop_duplicates(subset=['track_id'])#no repeated songs
print(newdf.shape)

finaldf = newdf.sample(n=20000, random_state=0).reset_index(drop=True)## randomly samples 20,000 rows from the dataset, random_state=42/0 ensures same rows are picked every run, reset_index resets row numbers to 0,1,2...
print(finaldf.head())
print(finaldf.shape)
print(finaldf.info())
print(finaldf['track_genre'].mode())

dance=finaldf.groupby('track_genre')['danceability'].mean().sort_values(ascending=False)

# ---- ANALYSIS 1: Most Danceable Genres ----
print(dance.head(10))
# get top 10
top10 = dance.head(10)
plt.figure(figsize=(12, 5))
plt.bar(top10.index, top10.values, color='pink')
plt.title('Top 10 Most Danceable Genres on Spotify')
plt.xlabel('Genre')
plt.ylabel('Average Danceability Score')
plt.ylim(0.68,0.80)
plt.tight_layout()
plt.show()


# ---- ANALYSIS 2: Most Energetic vs Calm Genres ----
genre=finaldf.groupby('track_genre')['energy'].mean().sort_values(ascending=False)
print(genre.head(10))
print(genre.tail(10))
top10_genre=genre.head(10)
plt.figure(figsize=(12, 5))
plt.bar(top10_genre.index, top10_genre.values, color='lightblue')
plt.title('Top 10 Most Energetic Genres on Spotify')
plt.xlabel('Genre')
plt.ylabel('Average Energy Score')
plt.ylim(0.80,1.0)
plt.tight_layout()
plt.show()
most_energetic_genre=genre.head(5)
most_calm=genre.tail(5)
combined=pd.concat([most_energetic_genre, most_calm])
print(combined)
plt.figure(figsize=(12, 5))
colors=['red']*5 + ['green']*5
plt.bar(combined.index , combined.values , color=colors)
plt.title('Most Energetic and Calm Genres on Spotify')
plt.xlabel('Genre') 
plt.ylabel('Average Energy Score')
plt.ylim(0.10,1.0)
plt.tight_layout()
plt.show()



## ---- ANALYSIS 3: Feature Correlation Heatmap ----
corr=finaldf.select_dtypes(include='number').corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True , fmt='.2f',cmap='coolwarm')
plt.title
('Correlation Heatmap of Spotify Track Features')
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()



## ---- ANALYSIS 4: Valence vs Popularity ----
plt.figure(figsize=(10,6))
plt.scatter(finaldf['valence'],finaldf['popularity'],alpha=0.5 , color='pink', s=10)
plt.title('Valence vs Popularity of Spotify Tracks')
plt.xlabel('Valence (Musical Positivity)')
plt.ylabel('Popularity')
plt.tight_layout()
plt.show()

           






