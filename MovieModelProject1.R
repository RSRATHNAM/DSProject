# Create edx set, validation set
################################
# @author Sujatha Rajarathnam

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Checking the number of rows and columns in the edx dataset
dim(edx) # 9000055       6

# checking the number of rows and columns in the validation dataset
dim(validation) # 999999      6

# Creating a function for calculating Root Mean Square Error (RMSE)
# This function will receive 2 inputs. Input 1 is the predicted rating
# Input 2 will be the actual rating from the edx dataset.
# The output will be RMSE.
RMSE <- function(predicted_rating, actual_rating)
  {
    sqrt(mean((predicted_rating - actual_rating)^2))
  }


# Building the model
# The simplest model is to find the average of all movie ratings. Which becomes the predicted
# rating that we can find the RMSE with the actual rating.

# For brevity, I am assigning edx to another variable. Because edx is a test data set
training_data <- edx
average_rating_pred <- mean(training_data$rating) # 3.512465

# Finding the RMSE for the simplest model
simplest_model_rmse <- RMSE(average_rating_pred,validation$rating)

# Value of RMSE from the simplest model
simplest_model_rmse # 1.061202
# The above tells us there is a difference of about 1 between actual rating and the
# predicted rating. I.e if a user rated a movie in the validation dataset as 2.5,
# our model predicts the rating around 2.5 + 1.061202, which is 3.561202. Or
# 2.5 - 1.061202 which is 1.438798. Which is not a good prediction.

# So the simplest model is not very accurate. Lets make the model accurate by adding 
# movie effect. Because users do not rate all movies, some movies are rated higher
# than the others due to popularity, blockbucter hits and etc.

# We will improve the prediction model by adding movie effect
# the equation is the following
# y_hat_m <- mu + movie_effect

# Also there are movies which are not rated that many times. These movies will
# have negative effect on prediction. Using regularization we can begate this effect
lambda = 3.0 # I am starting with lambda equals 3.

#movie_effect <- training_data %>% group_by(movieId) %>% summarize(bi=(mean(rating)-average_rating_pred))

movie_effect <- training_data %>% group_by(movieId) %>% summarize(bi=sum(rating-average_rating_pred)/(n()+lambda))

y_hat_m <- average_rating_pred + validation %>% left_join(movie_effect,by="movieId") %>% .$bi

# Lets calculate the RMSE for predicted rating with movie effect
me_model_RMSE <- RMSE(y_hat_m,validation$rating) # 0.9439087
# The value 0.9439087 is better than 1.061202, but still not great for prediction.

# As a next step lets find out how users are rating the movies. We can calculate user effect
# on top of the movie effect to improve our prediction.
# The equation looks like the following
# y_hat_mu = mu + movie_effect + user_effect
# Rearranging this equation 
# user_effect = y_hat_mu - mu - movie_effect
#user_effect <- training_data %>% left_join(movie_effect, by="movieId") %>% group_by(userId) %>% summarize(bu_hat_u = mean(rating-average_rating_pred-bi))

user_effect <- training_data %>% left_join(movie_effect, by="movieId") %>% group_by(userId) %>% summarize(bu_hat_u = sum(rating-average_rating_pred-bi)/(n()+lambda))

y_hat_mu <- y_hat_m + validation %>% left_join(user_effect,by="userId")%>%.$bu_hat_u

# After the user effect, lets now calculate the RMSE
user_movie_model_rmse <- RMSE(y_hat_mu,validation$rating)
user_movie_model_rmse # 0.86489