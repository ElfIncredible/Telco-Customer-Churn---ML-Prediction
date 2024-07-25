# Telco Customer Churn - ML Prediction
This initiative aims to identify customers who are likely to leave our services, allowing us to take proactive measures to retain them. Predicting churn will help us improve customer satisfaction and reduce revenue losses.
## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Machine Learning Prediction](#machine-learning-prediction)
  - [Install dependencies](#install-dependencies)
  - [Data collection and processing](#data-collection-and-processing)
  - [Data visualization](#data-visualization)
  - [Encoding](#encoding)
  - [Separating Features and Targets](#separating-features-and-targets)
  - [Scaling the data](#scaling-the-data)
  - [Split data into training and testing data](#split-data-into-training-and-testing-data)
  - [Model training - Random Forest Classifier](#model-training---random-forest-classifier)
  - [Model Prediction](#model-prediction)
  - [Model Evaluation](#model-evaluation)
  - [Feature Importance](#feature-importance)
  - [Building a predictive system](#building-a-predictive-system)
## Project Overview
The primary objectives of this project are:
- Develop a machine learning model to predict customer churn with high accuracy.
- Analyze customer data to identify key factors contributing to churn.
- Implement actionable insights to reduce churn based on model predictions.
- Evaluate model performance using metrics such as precision, recall, F1-score, and ROC-AUC.
## Dataset
7043 observations with 33 variables. 
find the dataset [here](https://drive.google.com/file/d/190TNAVQX7b8mXUfQmrzqrziiNdRS46ll/view)

- *CustomerID:* A unique ID that identifies each customer.
- *Count:* A value used in reporting/dashboarding to sum up the number of customers in a filtered set.
- *Country:* The country of the customer’s primary residence.
- *State:* The state of the customer’s primary residence.
- *City:* The city of the customer’s primary residence.
- *Zip Code:* The zip code of the customer’s primary residence.
- *Lat Long:* The combined latitude and longitude of the customer’s primary residence.
- *Latitude:* The latitude of the customer’s primary residence.
- *Longitude:* The longitude of the customer’s primary residence.
- *Gender:* The customer’s gender: Male, Female
- *Senior Citizen:* Indicates if the customer is 65 or older: Yes, No
- *Partner:* Indicate if the customer has a partner: Yes, No
- *Dependents:* Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.
- *Tenure Months:* Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
- *Phone Service:* Indicates if the customer subscribes to home phone service with the company: Yes, No
- *Multiple Lines:* Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No
- *Internet Service:* Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.
- *Online Security:* Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
- *Online Backup:* Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No
- *Device Protection:* Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No
- *Tech Support:* Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No
- *Streaming TV:* Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.
- *Streaming Movies:* Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.
- *Contract:* Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
- *Paperless Billing:* Indicates if the customer has chosen paperless billing: Yes, No
- *Payment Method:* Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check
- *Monthly Charge:* Indicates the customer’s current total monthly charge for all their services from the company.
- *Total Charges:* Indicates the customer’s total charges, calculated to the end of the quarter specified above.
- *Churn Label:* Yes = the customer left the company this quarter. No = the customer remained with the company. Directly related to Churn Value.
- *Churn Value:* 1 = the customer left the company this quarter. 0 = the customer remained with the company. Directly related to Churn Label.
- *Churn Score:* A value from 0-100 that is calculated using the predictive tool IBM SPSS Modeler. The model incorporates multiple factors known to cause churn. The higher the score, the more likely the customer will churn.
- *CLTV:* Customer Lifetime Value. A predicted CLTV is calculated using corporate formulas and existing data. The higher the value, the more valuable the customer. High value customers should be monitored for churn.
- *Churn Reason:* A customer’s specific reason for leaving the company. Directly related to Churn Category.
## Machine Learning Prediction
### Install dependencies
### Data collection and processing
### Data visualization
### Encoding
### Separating Features and Targets
### Scaling the data
### Split data into training and testing data
### Model training - Random Forest Classifier
### Model Prediction
### Model Evaluation
### Feature Importance
### Building a predictive system
