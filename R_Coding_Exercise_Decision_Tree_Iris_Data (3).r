
# Load the library for this exercise
# NOTE: Please ignore the warning messages
library(rpart)

# Load the iris dataset into memory     
data(iris) 

# View the dimension of the dataset   
dim(iris)

# View the first few rows of the dataset   
head(iris)

# View the struture of the dataset   
str(iris)

# View the summary statistics of the dataset
summary(iris)

# Randomly sample 70% of the rows in the dataset to use as our training set rows
train.index <- sample(1:nrow(iris), 0.7*nrow(iris)) 

# Assign these randomly selected 70% of rows to our training subset of data
iris.train <- iris[train.index,  ] 
dim(iris.train)

# Use the remaining 30% as the testing set
iris.test <- iris[-train.index,  ]
dim(iris.test)

# Default decision tree model
# Builds a decision tree from the iris dataset to predict species given all other columns as predictors
iris.tree <- rpart(Species ~., data = iris.train)  

# Visualize the model
# Plot the tree structure
plot(iris.tree, margin = c(0.25))
title(main = "Decision Tree Model of Iris Data")
text(iris.tree, use.n = TRUE)

# Let's see how the Petal.Length looks across different species    
boxplot(Petal.Length ~ Species, data = iris.train, main="Analyzing petal length across different flower species",
        xlab="Species",
        ylab="Petal Length (cm)")

# Print the iris.tree model
iris.tree

# Predict iris species given test data using the decision model
iris.predictions <- predict(iris.tree, iris.test, type = "class")
head(iris.predictions)

# Predicted values can also be probabilities,instead of class labels
iris.predictions.prob <- predict(iris.tree, iris.test, type = "prob")
head(iris.predictions.prob)

# Comparison table of actual values and predicted values
iris.comparison <- iris.test
iris.comparison$Predictions <- iris.predictions
iris.comparison[ , c("Species", "Predictions")]

# View misclassified rows
disagreement.index <- iris.comparison$Species != iris.comparison$Predictions 
iris.comparison[disagreement.index, ]

# Use method "anova" as a parameter to set a regression decision tree model
iris.tree <- rpart(Petal.Length ~ ., data = iris.train, method="anova")
