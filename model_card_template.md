# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Build the model by Aiden Nguyen (Udacity's student)
Model type: RandomForestClassifier.
Description: A supervised machine learning algorithm used for classification prediction. It generates the result based on the conclusion of many decision trees behind the algorithm. Each tree is the collection of random subset data features. When drawing the result, each tree will vote, and the majority of the vote will become a predicted result. 

## Intended Use
Primary intended use: The model is trained to classify whether a person's salary is more or less than $50K a year based on the census data from 1994.

Primary intended users: business consultants, researchers, or economist students who want to understand the trail of income related to individuals.

## Training Data
The dataset used for training the model: The Census Income.
Link: https://archive.ics.uci.edu/dataset/20/census+income
Features: 14 columns.
Feature types: Categorical and Integer.
Label: 1 (Salary) column.
Data used for training 80%: 26,048 rows

## Evaluation Data
Data used for testing 20%: 6,513 rows
Metrics for evaluation: Precision, Recall, and F1

The evaluation is conducted by comparing the model results predicting the 6,516 objects. Then, the model 

The procedure for conducting the evaluation:
    1. After the model is trained based on the 80% of the original data. 
     2. The model predicts the result for the 20% of the remaining (testing) data. 
    3. The model's prediction results will compare with the actual 20% of the (testing) label data. 
    4. The comparison (evaluation) will generate Precision, Recall, and F1 metrics. 

## Metrics
Precision metric: This metric will indicate the precision of the prediction results by the model. 
 _ In this study: out of all predicted people earning more than fifty thousand a year, how many people actually do? 
 _ Precision score (0.73) means When a model runs a prediction on a person based on their variables, the result of the prediction could be correct 73% of the time 
    
Recall metric: Out of all the people who earn more than fifty thousand a year, how many people did the model find?
 _ Recall score (0.64) means the model caught 64% of the high earners. 

F1 metric: this is the harmonic mean of the precision and recall score. It shows how well the model balances the two metrics above (0.69) as 69%.


## Ethical Considerations
This model is trained from historical data, which might not be compatible with the prediction of the current economy. It could also create unfair predictions or violate individual rights. 


## Caveats and Recommendations
The model should be used for study purposes to help build and evaluate the machine learning model. 
