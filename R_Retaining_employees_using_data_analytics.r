
# install and load libraries
install.packages("gmodels")

# for data import
library(readr)
library(readr)

# for data wrangling
library(tidyverse)
library(tidyr)

# for visualization 
library(gmodels)
library(reshape2)
library(scales)
library(ggthemes)
library(ggthemes)
library(dplyr)
library(ggplot2)

# for building classification and regression trees
library(rpart)
library(rpart.plot)

# load data into dataframe
HR_comma_sep <- read_csv("HR_comma_sep.csv")

# first rows of dataset
head(HR_comma_sep, 10)

# visualizing continous variables

theme_set(theme_gray(base_size = 10))

d <- melt(HR_comma_sep)
ggplot(d,aes(x = value)) +  
    facet_wrap(~variable,scales = "free_x") + 
    geom_histogram(bins = 10, fill = "lightblue", colour = "darkblue", size = .1) +
  scale_y_continuous( name = "Number of employees" ) +
  scale_x_continuous( name = "Frequency" ) 

# second glance at the binary continous variable: Work_accident 

wa <- ggplot(data=HR_comma_sep, aes(x=Work_accident, fill=as.factor(Work_accident))) + 
                                scale_x_continuous(breaks=0:1,
                                labels = c("No accident" , "Yes accident")) +
      geom_bar(width=0.5,
               (aes(y = (..count..)/sum(..count..))))+
      scale_y_continuous(labels = scales::percent) + 
      geom_text(aes(label = scales::percent((..count..)/sum(..count..)),
                   y= (..count..)/sum(..count..)), stat= "count", vjust =.5, hjust= 1.2, size=3, color='white') +
      labs(title = "Frequency of accidents at work",
               y = "Percentage of employees") +
      theme(aspect.ratio = .4) +
      coord_flip()  

wa

# second glance at the binary continous variable: resigned

re <- ggplot(data=HR_comma_sep, aes(x = resigned , fill=as.factor(resigned))) + 
                                scale_x_continuous(breaks=0:1,
                                labels = c("Stayed" , "Resigned")) +
      geom_bar(width=0.5,
               (aes(y = (..count..)/sum(..count..))))+
      scale_y_continuous(labels = scales::percent) + 
      geom_text(aes(label = scales::percent((..count..)/sum(..count..)),
                   y= (..count..)/sum(..count..)), stat= "count", vjust =.5, hjust= 1.2, size=3, color='white') + 
      labs(title = "Frequency of resignations",
               y = "Percentage of employees") +
      theme(aspect.ratio = .3) +
      coord_flip()  

re

# second glance at the binary continous variable: promotion in last 5 years

pro <- ggplot(data=HR_comma_sep, aes(x = promotion_last_5years , fill=as.factor(promotion_last_5years))) + 
                                scale_x_continuous(breaks=0:1,
                                labels = c("No promotion" , "Yes promotion")) +
      geom_bar(width=0.5,
               (aes(y = (..count..)/sum(..count..))))+
      scale_y_continuous(labels = scales::percent) + 
      geom_text(aes(label = scales::percent((..count..)/sum(..count..)),
                   y= (..count..)/sum(..count..)), stat= "count", vjust =.5, hjust= .7, size=2, color='black') +
      labs(title = "Frequency of promotions in last 5 years",
               y = "Percentage of employees") +
      theme(aspect.ratio = .3) +
      coord_flip()  

pro

# visualizing categorical variables

# salary_grade
sg <- ggplot(data=HR_comma_sep, aes(x = salary_grade , fill=as.factor(salary_grade))) + 
      geom_bar(width=0.9,
               (aes(y = (..count..)/sum(..count..))))+
      scale_y_continuous(labels = scales::percent) + 
      geom_text(aes(label = scales::percent((..count..)/sum(..count..)),
                   y= (..count..)/sum(..count..)), stat= "count", vjust =.5, hjust= 1.2, size=3, color='white') +
      labs(title = "Salary grade of employees",
               y = "Percentage of employees") +
      theme(aspect.ratio = .4) +
      coord_flip()

sg

# department
dp <- ggplot(data=HR_comma_sep, aes(x = department , fill=as.factor(department))) + 
      geom_bar(width=0.5,
               (aes(y = (..count..)/sum(..count..))))+
      scale_y_continuous(labels = scales::percent) + 
      geom_text(aes(label = scales::percent((..count..)/sum(..count..)),
                   y= (..count..)/sum(..count..)), stat= "count", vjust = 0.5, hjust= 1.5, size=3, color='white') +
      labs(title = "Departments of the organization",
               y = "Percentage of employees") +
      theme(aspect.ratio = 1.2) +
      coord_flip()

dp

# Creating factors of categorical variables and binary, continous variable resigned

hr <- HR_comma_sep %>% 
  mutate(salary_grade = as.factor(salary_grade) ,
         department = as.factor(department),
         resigned = as.factor(resigned),
         random = runif(14999))

# rename factor levels of variable 'resigned'
levels(hr$resigned) <- c("stayed", "resigned") 

print((summary(hr$resigned))) 

print((summary(hr$salary_grade)))

print((summary(hr$department)))

# Splitting the data into training and validation sets
set.seed(123)
train <- hr %>% 
  filter(random < .7) %>% 
  select(-random)

val <- hr %>% 
  filter(random >= .7) %>% 
  select(-random)

# creating the initial model
ct1 <- rpart(resigned ~ . , data = train, method = 'class')

# plotting the model
rpart.plot(ct1)

# complexity parameter table
ct1$cptable

print(var_importance <- data.frame(ct1$variable.importance)) 

# how good is our initial model(ct1)?

val$resign_predicted <- predict(ct1, val, type = 'class')
print(summary(val))

# rename factor levels of variable 'resigned_predicted'
levels(val$resign_predicted) <- c("predicted_stay", "predicted_resign") 
print(summary(val$resign_predicted))

# Crosstable of actual resignations versus predicted resignation values by initial model

CrossTable(val$resigned , val$resign_predicted)
