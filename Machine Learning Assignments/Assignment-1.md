# Assignment-1 Questions

#####

1. What does one mean by the term "machine learning"?

   *Ans: Machine Learning is a subset of Artificial Intelligence which allows usage of mathematical tools and algorithms for doing Exploratory Data Analysis, making              predictive models and much more.*

2.Can you think of 4 distinct types of issues where it shines?

   *Ans: a. Classification of data points into class labels of the target variable of the training data based on values of independent variables using 
    labeled data.*
  
   *b. Prediction of value of a continuous target variable value based on values of independent variables*
  
   *c. Classification of data points into class labels based on non-labeled data (having no information regarding the class labels of training data.)*
  
   *d. Both classification and regression by an agent that acts in an environment where the actions of the agent are governed by penalties using data as and when it is generated to learn on the fly.*

3.What is a labeled training set, and how does it work?

   *Ans: A labeled training set is one that has the actual, correct values of the target variable for all data points. 
    Using the labeled training set, the machine learning models form a complex algorithmic model by training on the data that stands true and learns to make        classifications, predictions or clusters within the confines of the training set’s actual target variable values.*


4.What are the two most important tasks that are supervised?

   *Ans: Classification and Regression are two most important tasks that are supervised. In most cases, a labeled data is required for correct predictions.*


5.Can you think of four examples of unsupervised tasks?

   *Ans: a. Detection of objects during a proctored exam/online exam*
  
   *b. Automated labeling of documents or corpus for Sentiment Analysis.*
  
   *c. Recommendation of movies belonging to the clusters from which you have watched most movies.*
  
   *d. Clustering similar genetic material for identification and/or classification of creatures into species.*


6.State the machine learning model that would be best to make a robot walk through various unfamiliar terrains?

   *Ans: Reinforcement or deep neural network machine learning models would be the best bet for either recursive learning over a period of time or for a single time learning from large dump of data.*


7.Which algorithm will you use to divide your customers into different groups?

   *Ans: The K-means algorithm should work well if the clusters that need to be formed are not of non-globular shapes and of different densities and sizes.
Covering above stated limitations would require the DBSCAN algorithm.*


8.Will you consider the problem of spam detection to be a supervised or unsupervised learning problem?

   *Ans: Spam detection has been done using labeled data, but unsupervised learning can also be used for the same problem with similar performance and efficacy.*


9.What is the concept of an online learning system?

   *Ans: A system in which a machine continuously learns using data being continuously fed or streamed is an online learning system.*


10.What is out-of-core learning, and how does it differ from core learning?

   *Ans: System in which a machine learns using data that cannot fit inside the machine’s on board memory is out of core learning. Core learning relies on data stored temporarily or permanently on the machine memory.*


11.What kind of learning algorithm makes predictions using a similarity measure?

   *Ans: Instance based algorithm*


12.What's the difference between a model parameter and a hyperparameter in a learning algorithm?

   *Ans: Model parameter affects the predictions made by the model while hyperparameters affect the learning curve, which needs to be tuned or optimised using optimisation techniques.*


13.What are the criteria that model-based learning algorithms look for? What is the most popular method they use to achieve success? What method do they use to make predictions?

   *Ans: Best value of hyper parameters and parameters are looked out for, to get best performance. A loss function is incorporated to get success. Previously unseen data points and parameter values are used for making predictions.*


14.Can you name four of the most important Machine Learning challenges?

   *Ans: a.Under representation of one class label due to it being a minority class,*
  
   *b. Small dataset to train model on,*
  
   *c. High variance model,*
  
   *d. And High bias model training.*


15.What happens if the model performs well on the training data but fails to generalize the results to new situations? Can you think of three different options?

   *Ans: If the model fails to perform well on new situations, we may have trained a high variance(overfit) model. To remedy this,*
  
   *a. Upsampling can be done to bring the number of minority class label data points up to the majority class.*
  
   *b. Downsampling to reduce the majority class to the number of minority class label data points.*
  
   *c. Removing outliers present in the data*
  
   *d. Adding class weights if the model parameters allow it.*



16.What exactly is a test set, and why would you need one?

   *Ans: Before deploying or using a machine learning model, it needs to be evaluated for its performance and efficacy against data points that were not a part of the training data. This unseen data is called test data.*


17.What is a validation set's purpose?

   *Ans: Comparing of different models requires a validation set so that the best model can be chosen out of ranked base learners, that will be optimised later.*


18.What precisely is the train-dev kit, when will you need it, how do you put it to use?

   *Ans: Train-Dev kit is used to find the split of data for training data and dev data. Dev set is used to find the best performing model and for choosing it out of many base learners and ultimately optimise the chosen model.*


19.What could go wrong if you use the test set to tune hyperparameters?

   *Ans: Test sets are generally much smaller than the training data. If test data is used to tune hyperparameters then the values gained after optimisation for hyperparameters would be for that small dataset. Using these values during actual usage will lead to wrong results *
