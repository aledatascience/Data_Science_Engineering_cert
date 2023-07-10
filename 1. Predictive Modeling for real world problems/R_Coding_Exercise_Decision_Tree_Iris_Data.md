
Data Science Dojo <br/>
Copyright(c) 2016-2020<br/>

---

**Objective:** Building a decision tree classification model to predict the species of Iris flowers<br/>

<b>Note</b>: The below libraries are already installed on the learning portal to run this exercise.<br/>
If you would like to run this code on your local machine using R studio or Jupyter Notebook, please run the below install commands in your command line or R studio. <br/>
If you would like to run this code on your local Jupyter Notebook, you can also run the below install commands in a new cell of jupyter notebook instead of command line.<br/>

Please install "rpart" package: `install.packages("rpart")`<br/>


```R
# Load the library for this exercise
# NOTE: Please ignore the warning messages
library(rpart)
```

### The Iris dataset
Before building a model for predicting the species of Iris flowers, we first need to read and explore the Iris dataset.

#### Dataset description
**Iris:** This iris data set gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica. __[Click here to read more](https://www.rdocumentation.org/packages/datasets/versions/3.6.2/topics/iris)__.


```R
# Load the iris dataset into memory     
data(iris) 
```


```R
# View the dimension of the dataset   
dim(iris)
```


<ol class=list-inline>
	<li>150</li>
	<li>5</li>
</ol>




```R
# View the first few rows of the dataset   
head(iris)
```


<table>
<thead><tr><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th><th scope=col>Species</th></tr></thead>
<tbody>
	<tr><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.9   </td><td>3.0   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.7   </td><td>3.2   </td><td>1.3   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.6   </td><td>3.1   </td><td>1.5   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.0   </td><td>3.6   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td></tr>
</tbody>
</table>




```R
# View the struture of the dataset   
str(iris)
```

    'data.frame':	150 obs. of  5 variables:
     $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
     $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
     $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
     $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
     $ Species     : Factor w/ 3 levels "setosa","versicolor",..: 1 1 1 1 1 1 1 1 1 1 ...



```R
# View the summary statistics of the dataset
summary(iris)
```


      Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
     Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
     1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
     Median :5.800   Median :3.000   Median :4.350   Median :1.300  
     Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
     3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
     Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  
           Species  
     setosa    :50  
     versicolor:50  
     virginica :50  
                    
                    
                    


### Building the model

Build and train a model to classify the different Iris species.


```R
# Randomly sample 70% of the rows in the dataset to use as our training set rows
train.index <- sample(1:nrow(iris), 0.7*nrow(iris)) 
```


```R
# Assign these randomly selected 70% of rows to our training subset of data
iris.train <- iris[train.index,  ] 
dim(iris.train)
```


<ol class=list-inline>
	<li>105</li>
	<li>5</li>
</ol>




```R
# Use the remaining 30% as the testing set
iris.test <- iris[-train.index,  ]
dim(iris.test)
```


<ol class=list-inline>
	<li>45</li>
	<li>5</li>
</ol>




```R
# Default decision tree model
# Builds a decision tree from the iris dataset to predict species given all other columns as predictors
iris.tree <- rpart(Species ~., data = iris.train)  
```

### Visualizing the model

Let's look how the decision tree model looks visually for the Iris dataset.


```R
# Visualize the model
# Plot the tree structure
plot(iris.tree, margin = c(0.25))
title(main = "Decision Tree Model of Iris Data")
text(iris.tree, use.n = TRUE)
```


![png](output_14_0.png)



```R
# Let's see how the Petal.Length looks across different species    
boxplot(Petal.Length ~ Species, data = iris.train, main="Analyzing petal length across different flower species",
        xlab="Species",
        ylab="Petal Length (cm)")
```


![png](output_15_0.png)



```R
# Print the iris.tree model
iris.tree
```


    n= 105 
    
    node), split, n, loss, yval, (yprob)
          * denotes terminal node
    
    1) root 105 66 setosa (0.37142857 0.33333333 0.29523810)  
      2) Petal.Length< 2.45 39  0 setosa (1.00000000 0.00000000 0.00000000) *
      3) Petal.Length>=2.45 66 31 versicolor (0.00000000 0.53030303 0.46969697)  
        6) Petal.Width< 1.65 36  2 versicolor (0.00000000 0.94444444 0.05555556) *
        7) Petal.Width>=1.65 30  1 virginica (0.00000000 0.03333333 0.96666667) *


### Model predictions

We will use the test data to make predict the iris species using the model we built.


```R
# Predict iris species given test data using the decision model
iris.predictions <- predict(iris.tree, iris.test, type = "class")
head(iris.predictions)
```


<dl class=dl-horizontal>
	<dt>4</dt>
		<dd>setosa</dd>
	<dt>6</dt>
		<dd>setosa</dd>
	<dt>11</dt>
		<dd>setosa</dd>
	<dt>16</dt>
		<dd>setosa</dd>
	<dt>24</dt>
		<dd>setosa</dd>
	<dt>28</dt>
		<dd>setosa</dd>
</dl>

<details>
	<summary style=display:list-item;cursor:pointer>
		<strong>Levels</strong>:
	</summary>
	<ol class=list-inline>
		<li>'setosa'</li>
		<li>'versicolor'</li>
		<li>'virginica'</li>
	</ol>
</details>



```R
# Predicted values can also be probabilities,instead of class labels
iris.predictions.prob <- predict(iris.tree, iris.test, type = "prob")
head(iris.predictions.prob)
```


<table>
<thead><tr><th></th><th scope=col>setosa</th><th scope=col>versicolor</th><th scope=col>virginica</th></tr></thead>
<tbody>
	<tr><th scope=row>4</th><td>1</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>6</th><td>1</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>11</th><td>1</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>16</th><td>1</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>24</th><td>1</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>28</th><td>1</td><td>0</td><td>0</td></tr>
</tbody>
</table>




```R
# Comparison table of actual values and predicted values
iris.comparison <- iris.test
iris.comparison$Predictions <- iris.predictions
iris.comparison[ , c("Species", "Predictions")]
```


<table>
<thead><tr><th></th><th scope=col>Species</th><th scope=col>Predictions</th></tr></thead>
<tbody>
	<tr><th scope=row>4</th><td>setosa    </td><td>setosa    </td></tr>
	<tr><th scope=row>6</th><td>setosa    </td><td>setosa    </td></tr>
	<tr><th scope=row>11</th><td>setosa    </td><td>setosa    </td></tr>
	<tr><th scope=row>16</th><td>setosa    </td><td>setosa    </td></tr>
	<tr><th scope=row>24</th><td>setosa    </td><td>setosa    </td></tr>
	<tr><th scope=row>28</th><td>setosa    </td><td>setosa    </td></tr>
	<tr><th scope=row>36</th><td>setosa    </td><td>setosa    </td></tr>
	<tr><th scope=row>37</th><td>setosa    </td><td>setosa    </td></tr>
	<tr><th scope=row>39</th><td>setosa    </td><td>setosa    </td></tr>
	<tr><th scope=row>45</th><td>setosa    </td><td>setosa    </td></tr>
	<tr><th scope=row>50</th><td>setosa    </td><td>setosa    </td></tr>
	<tr><th scope=row>58</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>60</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>62</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>66</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>68</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>72</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>77</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>78</th><td>versicolor</td><td>virginica </td></tr>
	<tr><th scope=row>87</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>88</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>92</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>95</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>97</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>98</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>100</th><td>versicolor</td><td>versicolor</td></tr>
	<tr><th scope=row>102</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>103</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>108</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>110</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>117</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>123</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>125</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>126</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>128</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>131</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>133</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>134</th><td>virginica </td><td>versicolor</td></tr>
	<tr><th scope=row>135</th><td>virginica </td><td>versicolor</td></tr>
	<tr><th scope=row>136</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>141</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>142</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>144</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>146</th><td>virginica </td><td>virginica </td></tr>
	<tr><th scope=row>147</th><td>virginica </td><td>virginica </td></tr>
</tbody>
</table>




```R
# View misclassified rows
disagreement.index <- iris.comparison$Species != iris.comparison$Predictions 
iris.comparison[disagreement.index, ]
```


<table>
<thead><tr><th></th><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th><th scope=col>Species</th><th scope=col>Predictions</th></tr></thead>
<tbody>
	<tr><th scope=row>78</th><td>6.7       </td><td>3.0       </td><td>5.0       </td><td>1.7       </td><td>versicolor</td><td>virginica </td></tr>
	<tr><th scope=row>134</th><td>6.3       </td><td>2.8       </td><td>5.1       </td><td>1.5       </td><td>virginica </td><td>versicolor</td></tr>
	<tr><th scope=row>135</th><td>6.1       </td><td>2.6       </td><td>5.6       </td><td>1.4       </td><td>virginica </td><td>versicolor</td></tr>
</tbody>
</table>



### Regression decision tree

Using decision tree, we can also build a regression model where the outcome is a predicted number instead of a predicted class.


```R
# Use method "anova" as a parameter to set a regression decision tree model
iris.tree <- rpart(Petal.Length ~ ., data = iris.train, method="anova")
```

### Further practice

Another library called "party" can be also used to build decision trees. It provides nonparametric regression trees for nominal, ordinal, numeric, censored, and multivariate responses. Tree growth is based on statistical stopping rules, so pruning should not be required. 

See `party` manual: http://cran.r-project.org/web/packages/party/party.pdf 

Instead of `rpart()`, try to use `ctree()` in `party` for the same data. They implement a different algorithm for building the tree. But for this small amount of data, do these different functions (with different algorithms) actually give us different trees?

### Reproducible results with set.seed() 

The sample() function used above randomly selects 105 rows from the 150 in the dataset. So, the train.index stores 105 numbers corresponding to the 105 rows. However, the row numbers stored in train.index is not always fixed. If we run `train.index <- sample(1:nrow(iris), 0.7*nrow(iris))` again, the numbers in train.index will be changed. As a result, the training data will be changed.  

+ Try running this code repeatedly and see how the output changes everytme.

  + `sample(1:10, 5)`  

Often, we want our code to reproduce the exact same set of random numbers. We can use the `set.seed()` function to do this.  The `set.seed()` function takes an integer argument. Once the `set.seed()` has been used, the output of all the subsequent random operations, such as `sample()` get fixed, which ensures reproducability of the results.   

+ Try running sample(1:10, 5) again but after setting the seed as 100  
    + `set.seed(100)`
    + `sample(1:10, 5)` 
    
    
+ Now try running sample(1:10, 5) again after changing the seed to 999   
    + `set.seed(999)`
    + `sample(1:10, 5)`  
    
+ Change the seed back to 100 to see how you get the same results   
    + `set.seed(100)`
    + `sample(1:10, 5)`  
