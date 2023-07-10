
# coding: utf-8

# Data Science Dojo<br/>
# Copyright(c) 2016-2020<br/>
# 
# ---

# **Objective:** Building a decision tree classification model to predict the species of Iris flowers<br/>
# 
# 
# <b>Note</b>: The below packages are already installed on the learning portal to run this exercise.<br/>
# If you would like to run this code on your local machine, please run the below install commands in your command line tool. <br/>
# If you would like to run this code on your local Jupyter Notebook, you can also run the below install commands in a new cell of jupyter notebook instead of command line using this convention `!pip install package name`.<br/>
# 
# Please install pandas package: `pip install pandas`<br/>
# Please install scikit-learn package: `pip install scikit-learn`<br/>
# Please install numpy package: `pip install numpy`<br/>
# Please install seaborn package: `pip install sns`<br/>
# Please install matplotlib package: `pip install matplotlib`<br/>
# Please install pydotplus package: `pip install pydotplus` *(Note: You might need to also install graphviz)*<br/>

# ### The Iris dataset
# Before building a model for predicting the species of Iris flowers, we first need to read and explore the Iris dataset.
# 
# #### Dataset description
# **Iris:** This iris data set gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are Iris setosa, versicolor, and virginica. __[Click here to read more](https://www.rdocumentation.org/packages/datasets/versions/3.6.2/topics/iris)__.

# In[2]:


# Import the required package
from sklearn.datasets import load_iris
# Load the iris dataset into memory
iris = load_iris()


# In[3]:


# Import the required package
import pandas as pd
# Create a dataframe
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


# In[4]:


# View the dimension of the dataset   
iris_df.shape


# In[5]:


# View the first few rows of the dataset
iris_df.head()


# In[6]:


# View the structure/data types of the dataset   
iris_df.dtypes


# In[7]:


# View the summary statistics of the dataset
iris_df.describe()


# ### Building the model
# 
# Build and train a model to classify the different Iris species.

# In[8]:


# Randomly select 70% of the data as training set
iris_df_train = iris_df.sample(frac=0.7, random_state=1)


# In[9]:


# Use the remaining 30% as the testing set
iris_df_test = iris_df.loc[~iris_df.set_index(list(iris_df.columns)).index.isin(iris_df_train.set_index(list(iris_df_train.columns)).index)]


# In[10]:


print('Training set',iris_df_train.shape)
print('Testing set',iris_df_test.shape)


# In[13]:


# Import the required packages
from sklearn.tree import DecisionTreeClassifier
import numpy as np
# Default decision tree model
# Builds a decision tree from the iris dataset to predict species given all other columns as predictors
np.random.seed(27)
iris_features = iris_df.columns[:4]
iris_dt_clf = DecisionTreeClassifier()
iris_dt_clf = iris_dt_clf.fit(iris_df_train[iris_features], iris_df_train['Species'])


# ### Visualizing the model
# 
# Let's look how the decision tree model looks visually for the Iris dataset.

# In[15]:


# Import the required package
from sklearn import tree
# Visualize the model
# Plot the tree structure
dot_data = tree.export_graphviz(iris_dt_clf,
                                out_file=None,
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names,
                                rounded=True, 
                                filled=True)


# In[16]:


# Import the required packages
from IPython.display import Image
import pydotplus
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  
# Show graph
Image(graph.create_png())


# In[17]:


# Import the required packages
import seaborn as sns
import matplotlib.pyplot as plt
# Let's see how the Petal.Length looks across Species
plt.figure(figsize=(12, 8))
sns.set(font_scale=1.5)
ax = sns.boxplot(data=iris_df_train, x="Species", y="petal length (cm)")
# Setting the category labels for x axis
ax.set_xticklabels(['setosa', 'versicolor', 'virginica'])
# Setting the title, x-label, y-label
plt.title("Analyzing petal length across different flower species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")


# In[18]:


# Import the required package
from sklearn.externals.six import StringIO
# Print a string representation of the iris tree model
# If you have graphviz (www.graphviz.org) installed, you can write a pdf
# Visualization using graph.write_pdf(filename)
iris_dt_data = StringIO()
tree.export_graphviz(iris_dt_clf, out_file=iris_dt_data)
iris_dt_graph = pydotplus.parser.parse_dot_data(iris_dt_data.getvalue())
print(iris_dt_graph.to_string())


# ### Model predictions
# 
# We will use the test data to make predict the iris species using the model we built.

# In[19]:


# Predict iris species given test data using the decision model
iris_dt_pred = iris_dt_clf.predict(iris_df_test[iris_features])
iris_dt_pred


# In[20]:


# Predicted values can also be probabilities, instead of class labels
iris_dt_pred_prob = iris_dt_clf.predict_proba(iris_df_test[iris_features])
iris_dt_pred_prob


# In[21]:


# Comparison table of actual values and predicted values
iris_comparison = iris_df_test.copy()
iris_comparison['Predictions'] = iris_dt_pred
iris_comparison[['Species','Predictions']]


# In[22]:


# View misclassified rows
iris_comparison.loc[iris_comparison['Species'] != iris_comparison['Predictions']]


# ### Regression decision tree
# 
# Using decision tree, we can also build a regression model where the outcome is a predicted number instead of a predicted class.

# In[24]:


# Import the required package
from sklearn import preprocessing
# Label encoding the categorial features
le = preprocessing.LabelEncoder()
le.fit(iris_df_train['Species'])
iris_df_train['Species'] = le.transform(iris_df_train['Species'])


# In[25]:


# Import the required package
from sklearn.tree import DecisionTreeRegressor
# Use DecisionTreeRegressor() to build a regression decision tree model
features = ['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)', 'Species']
iris_dt_clf = DecisionTreeRegressor()  
iris_dt_clf = iris_dt_clf.fit(iris_df_train[features], iris_df_train['petal length (cm)'])
iris_dt_clf


# ### Reproducible results with random_state

# The sample() function used above, randomly selects rows from the dataset. So, the iris_df_train stores these rows. However, the rows stored in iris_df_train is not always fixed. If we run `iris_df_train = iris_df.sample(frac=0.7)` again, the rows in iris_df_train will be changed. As a result, the training data will be changed.  
# 
# + Try running this code repeatedly and see how the output changes everytme.
# 
#   + `iris_df.sample(frac=0.7)`  

# Often, we want our code to reproduce the exact same set of random numbers. We can use the `random_state` option to do this.  The `random_state` option takes an integer argument. Once the `random_state` has been used, the output of all the subsequent random operations, such as sample() get fixed, which ensures reproducability of the results.   
# 
# + Try running iris_df.sample(frac=0.7) again but after setting the seed as 100  
#     + `iris_df.sample(frac=0.7, random_state=100)` 
#     
#     
# + Now try running iris_df.sample(frac=0.7) again after changing the seed to 999 
#     + `iris_df.sample(frac=0.7, random_state=999)`  
#     
# + Change the seed back to 100 to see how you get the same results 
#     + `iris_df.sample(frac=0.7, random_state=100)`  
