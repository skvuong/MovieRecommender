#--------------------------------------------------------------------------------------
############### Collaborative Filtering Recommendation system ##########################
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
############### Install and load the required packages
#--------------------------------------------------------------------------------------
if(!require(arules))
  install.packages("arules")
if(!require(caret)) 
  install.packages("caret")
if(!require(dplyr))
  install.packages("dplyr")
if(!require(data.table)) 
  install.packages("data.table")
if(!require(ggplot2))
  install.packages("ggplot2")
if(!require(ggthemes))
  install.packages("ggthemes")
if(!require(lubridate)) 
  install.packages("lubridate")
if(!require(reshape2))
  install.packages("reshape2")
if(!require(recommenderlab))
  install.packages("recommenderlab")
if(!require(scales)) 
  install.packages("scales")
if(!require(stringr)) 
  install.packages("stringr")
if(!require(stringdist)) 
  install.packages("stringdist")
if(!require(tidyverse)) 
  install.packages("tidyverse")
if(!require(wordcloud))
  install.packages("wordcloud")

library(arules)
library(caret)
library(dplyr)
library(data.table)
library(ggplot2)
library(ggthemes)
library(lubridate)
library(Matrix)
library(reshape2)
library(recommenderlab)
library(scales)
library(stringr)
library(stringdist)
library(tidyverse)
library(wordcloud)

#--------------------------------------------------------------------------------------
#Setting Working Directory
#--------------------------------------------------------------------------------------
#setwd("C:/Users/Sam/Documents/Project1")

#--------------------------------------------------------------------------------------
############### Reading input CSV data files
#--------------------------------------------------------------------------------------

# Read movies file and look at first few lines
movies<-read.csv("movies.csv", header=TRUE)
head(movies)

# Read ratings file and look at first few lines
ratings<-read.csv("ratings.csv", header=TRUE)
head(ratings)

# Read tags file and look at first few lines
tags<-read.csv("tags.csv", header=TRUE)
head(tags)

# Read links file and look at first few lines
links<-read.csv("links.csv", header=TRUE)
head(links)

#--------------------------------------------------------------------------------------
############### Data Exploration Analysis
#--------------------------------------------------------------------------------------

# Checking data types of data
str(movies)
str(ratings)
str(tags)
str(links)

# Checking number of movies
length(unique(movies$movieId))
length(unique(ratings$movieId))

# Checking number of users
length(unique(ratings$userId))

# Checking Summary of data
summary(movies)
summary(ratings)

# Convert timestamp to date
ratings <- mutate(ratings, year = year(as_datetime(timestamp, origin="1970-01-01")))
head(ratings)

# Joining Ratings and Movies dataframes
moviesratings <- left_join(ratings, movies, by="movieId")
head(moviesratings)
str(moviesratings)

# Double checking number of movies
length(unique(movies$movieId))
length(unique(ratings$movieId))
length(unique(moviesratings$movieId))

# Checking Summary of data
summary(moviesratings)

#--------------------------------------------------------------------------------------
############### Data Visualization
#--------------------------------------------------------------------------------------

#1. Users vs. Ratings Distribution
moviesratings %>% group_by(userId) %>% summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "white") +
  scale_x_log10() + 
  ggtitle("Distribution of Users") +
  xlab("Number of Ratings") +
  ylab("Number of Users") + 
  scale_y_continuous(labels = comma) + 
  theme_economist()

#2. Rating Distribution Histogram
moviercount<-moviesratings %>% group_by(rating) %>% summarise(count=n())
view(moviercount)
class(moviercount)
moviercount<-as.data.frame(moviercount)

ggplot(moviercount,aes(x=rating,y=count/1000)) +geom_bar(stat="identity") + 
  ggtitle("Rating Distribution") + 
  xlab("Rating") +
  ylab("# Ratings 1*10^3") +
  theme_economist()

#3. Rating vs. Year Distribution
moviesratings %>%
  ggplot(aes(x=year)) +
  geom_bar() +
  xlab("Year") +
  ylab("Number of Ratings") +
  scale_y_continuous(labels = comma) + 
  ggtitle("Rating Distribution Per Year") +
  theme_economist()

#4. Movie Genres Tag Cloud
tag_data <- moviesratings %>% select(genres) %>% 
  group_by(genres) %>% summarise(count = n()) %>% 
  arrange(desc(count)) %>% as.data.frame()

set.seed(1234)
wordcloud(words = moviesratings$genres,
          freq = tag_data$count, min.freq = 1,
          max.words=30, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

#5. vertical bar chart for tags
top_tags <- tags %>%
  group_by(tag) %>%
  summarise(tag_count = n()) %>%
  arrange(desc(tag_count)) %>%
  slice(1:10)


ggplot(top_tags,aes(x=tag,y=tag_count,fill=tag)) + 
  geom_bar(stat = "identity") + 
  coord_flip() + 
  ggtitle("Genre Distribution") + 
  theme(legend.position = "none")

#--------------------------------------------------------------------------------------
############### Data Preparation for Modeling
#--------------------------------------------------------------------------------------

# Create a dataframe with only 3 columns required for the Recommender model
rating_df <- select(moviesratings, userId, movieId, rating)
str(rating_df)
head(rating_df)

#--------------------------------------------------------------------------------------
# Convert it as a matrix, then to realRatingMatrix
#--------------------------------------------------------------------------------------
# Using acast to convert above data as follows:
#       m1  m2   m3   m4
# u1    3   4    2    5
# u2    1   6    5
# u3    4   4    2    5
rating_df_matrix<-acast(rating_df, userId ~ movieId, value.var = "rating")
class(rating_df_matrix)

# Convert it into matrix
rating_matrix_raw<-as.matrix(rating_df_matrix)
class(rating_matrix_raw)

# Convert it into realRatingMatrix data structure
# realRatingMatrix is a recommenderlab sparse-matrix like data-structure
rating_matrix <- as(rating_matrix_raw, "realRatingMatrix")
rating_matrix

# view rating_matrix in other possible ways
#as(rating_matrix, "list")     # A list
#as(rating_matrix, "matrix")   # A sparse matrix
head(as(rating_matrix, "data.frame")) # View as a data-frame

# normalize the rating matrix
rating_matrix_norm <- normalize(rating_matrix)
rating_matrix_norm
head(as(rating_matrix_norm, "data.frame")) # View as a data-frame

# Can also turn the matrix into a 0-1 binary matrix
rating_matrix_bin <- binarize(rating_matrix, minRating=2)
rating_matrix_bin
head(as(rating_matrix_bin, "data.frame")) # View as a data-frame

#--------------------------------------------------------------------------------------
# Draw an image plot of raw-ratings & normalized ratings
#--------------------------------------------------------------------------------------
#  A column represents one specific movie and ratings by users
#   are shaded.
#   Note that some items are always rated 'black' by most users
#    while some items are not rated by many users
#     On the other hand a few users always give high ratings
#      as in some cases a series of black dots cut across items
#image(rating_matrix,      main = "Raw Ratings")       
#image(rating_matrix_norm, main = "Normalized Ratings")

#--------------------------------------------------------------------------------------
# Create a Recommender model
#--------------------------------------------------------------------------------------
#   Run anyone of the following four code lines.
#     Do not run all four
#       They pertain to four different algorithms.
#        UBCF: User-based collaborative filtering
#        IBCF: Item-based collaborative filtering
#      Parameter 'method' decides similarity measure
#        Cosine or Jaccard
#rec=Recommender(rating_matrix,method="POPULAR")
#rec=Recommender(rating_matrix,method="IBCF",param=list(normalize="Z-score",method="Jaccard"))
#rec=Recommender(rating_matrix,method="UBCF",param=list(normalize="Z-score",method="Jaccard",nn=5))
#--------------------------------------------------------------------------------------
# Create a Recommender model using UBCF (user-based collaborative filtering)
#--------------------------------------------------------------------------------------
rec=Recommender(rating_matrix,method="UBCF",param=list(normalize="Z-score",method="Cosine", nn=5))

# Examine the model we got
print(rec)
getModel(rec)$data
getModel(rec)$nn
names(getModel(rec))

#--------------------------------------------------------------------------------------
# Create predictions
#--------------------------------------------------------------------------------------
# This prediction does not predict movie ratings for test.
#   But it fills up the user 'X' item matrix so that
#    for any userid and movieid, I can find predicted rating
#     dim(r) shows there are 610 users (rows)
#      'type' parameter decides whether you want ratings or top-n items
#         get top-10 recommendations for a user, as:
#             predict(rec, rating_matrix, type="topNList", n=10)
recom <- predict(rec, rating_matrix, type="ratings")
recom

# Convert all recommendations to list structure. 
rec_list<-as(recom,"list")
head(summary(rec_list))

#--------------------------------------------------------------------------------------
# Check movie recommendations with n users in test data
#--------------------------------------------------------------------------------------
n_users  <- 5
n_movies <- 10
# For the users in test file, one by one
for ( u in 1:n_users)
{
  # Get userid
  uid  <- u

  # Obtain top n recommendations for the user
  recom <- predict(rec, rating_matrix[uid], n=n_movies)
  # Convert it to readable list
  recom_list <- as(recom, "list")  

  # Obtain movie titles of recommendations from movies dataset
  rec_movies <- matrix(0,n_movies)
  for (i in 1:n_movies){
    rec_movies[i] <- as.character(subset(movies,
                                  movies$movieId == as.integer(recom_list[[1]][i]))$title)
  }
  print(paste("==== Movie Recommendations For User: ", uid, " ===="))
  print(rec_movies)
  print("")
}

#--------------------------------------------------------------------------------------
############### Evaluation
#--------------------------------------------------------------------------------------
# Take subset of data users with more than 50 ratings
rating_matrix <- rating_matrix[rowCounts(rating_matrix) >50,]
rating_matrix

#--------------------------------------------------------------------------------------
# Create an Evaluation Scheme
#   using train/test 80/20 split validation
#   with given=5 means with 5 items given scheme
#   with goodRating=3 means items with user rating >= 3 are considered positives
#--------------------------------------------------------------------------------------
eval_scheme <- evaluationScheme(rating_matrix,
                                method="split",train=0.80,k=1,given=5,goodRating=3) 
eval_scheme

#--------------------------------------------------------------------------------------
# Compare UBCF algorithm other basic algorithms
#   Here we compare the UBCF model (user-based collaborative filtering)
#   to the POPULAR model (based on item popularity)
#   and the RANDOM model (random recommendations)
#--------------------------------------------------------------------------------------
algorithms <- list(
  UBCF = list(name = "UBCF", param = NULL),
  POPULAR = list(name = "POPULAR", param = NULL),
  RANDOM  = list(name = "RANDOM",  param = NULL)
)

# Evaluate using top-N recommendation lists
eval_topNList <- evaluate(eval_scheme,
                         algorithms,type="topNList",n=c(1,3,5,10,15,20))
# Print results
avg(eval_topNList)

# Plot ROC Curse
plot(eval_topNList, annotate=TRUE, main="ROC Curve")

plot(eval_topNList, "prec/rec", annotate=TRUE, main="Precision-Recall")

# Evaluate prediction of missing ratings
eval_ratings <- evaluate(eval_scheme, algorithms, type="ratings")

# Print results
avg(eval_ratings)

# Print results
plot(eval_ratings)

#######################################################################################