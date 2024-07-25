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
- [Summary of Findings and Mitigation Strategies](#summary-of-findings-and-mitigation-strategies)
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
Imports various libraries and modules necessary for data manipulation, visualization, geographic data handling, machine learning, and model evaluation.
  - *pandas (pd):* Used for data manipulation and analysis, particularly with dataframes.
  - *numpy (np):* Provides support for numerical operations, including array handling.
  - *seaborn (sns):* A statistical data visualization library based on matplotlib.
  - *matplotlib.pyplot (plt):* A plotting library used for creating static, interactive, and animated visualizations.
  - *shapely.geometry.Point:* Used for creating point geometries.
  - *geopandas (gpd):* Extends pandas to allow spatial operations on geometric types.
  - *contextily (ctx):* Adds basemaps to geospatial visualizations.
  - *sklearn.preprocessing.LabelEncoder:* Encodes target labels with values between 0 and n_classes-1.
  - *sklearn.model_selection.train_test_split:* Splits arrays or matrices into random train and test subsets.
  - *sklearn.preprocessing.StandardScaler:* Standardizes features by removing the mean and scaling to unit variance.
  - *sklearn.ensemble.RandomForestClassifier:* A machine learning algorithm for classification that uses an ensemble of decision trees.
  - *sklearn.metrics.classification_report:* Builds a text report showing the main classification metrics.
  - *sklearn.metrics.roc_auc_score:* Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
  - *sklearn.metrics.roc_curve:* Computes Receiver Operating Characteristic (ROC) curve.
### Data collection and processing
 Perform initial data exploration, including:
 - Loading the data
 - Displaying a sample
 - Summarizing statistics
 - Checking for missing values
 - Analyzing the distribution of certain categorical columns.
### Data visualization
Perform a comprehensive exploratory data analysis (EDA) using a combination of histograms, KDE plots, bar charts, box plots, scatter plots, and geospatial visualizations. This analysis provides deep insights into customer demographics, behavior, and churn patterns, which can be useful for developing targeted strategies to improve customer retention and satisfaction.
### Encoding
Encode the specified categorical columns in the DataFrame into numerical values, which are often required for machine learning models that cannot handle categorical data directly. The mappings dictionary retains the mapping information, which can be useful for interpreting the encoded values or for inverse transformations if needed.
### Separating Features and Targets
Prepare the data for machine learning by separating the features (input variables) and the target variable (output variable). The features X contain all the relevant information needed to make predictions, while y contains the values that the model aims to predict. This separation is a common practice in machine learning to facilitate model training and evaluation.
### Scaling the data
Standardize the numerical features in the DataFrame X, ensuring that each feature has a mean of 0 and a standard deviation of 1. This standardization is important for many machine learning algorithms, which perform better when the input features are on a similar scale.
### Split data into training and testing data
Split the standardized feature set and the target variable into training and testing subsets, with 80% of the data used for training the machine learning model and 20% used for testing its performance. The random_state parameter ensures that the split is reproducible.
### Model training - Random Forest Classifier
Initialize a Random Forest Classifier with a fixed random seed for reproducibility and then train this model using the training data (X_train and y_train). The trained model can then be used to make predictions on new data.
### Model Prediction
Prepare the results of the model's predictions for further evaluation and analysis, such as calculating performance metrics or generating ROC curves.
### Model Evaluation
Evaluate the performance of a classification model using several metrics and visualizations.
- *classification_report* provides a comprehensive evaluation of the model’s performance.
- *roc_auc_score* quantifies the model's ability to discriminate between positive and negative classes.
- *ROC curve plot* visualizes the trade-off between true and false positive rates across different thresholds and provides a graphical representation of the model’s performance.
### Feature Importance
This process helps in understanding which features contribute the most to the predictions made by the Random Forest model, guiding feature selection and model interpretation.
- *Calculates Feature Importance:* Uses the feature_importances_ attribute from the trained Random Forest model to get the importance scores of each feature.
- *Creates a DataFrame:* Organizes the feature names and their corresponding importance scores into a DataFrame.
- *Sorts and Displays:* Sorts the DataFrame by feature importance in descending order and prints it to highlight which features are most influential in the model.
### Building a predictive system
Handle the complete process of preprocessing a new input for prediction and then use the trained Random Forest model to determine if a customer is likely to churn. It ensures that the input data is formatted and scaled in a way that matches the model's training data, providing a reliable prediction based on the new data provided.
## Summary of Findings and Mitigation Strategies
Several important insights regarding customer churn have been uncovered as a result of the analysis. The following provides a thorough overview of the results and suggested approaches to deal with them:

**1. High Churn Rate in the First Month**

*Finding:* 

A significant portion of customers churn within the first month, but those who stay beyond this period are likely to remain for a much longer duration.

*Mitigation Strategies:*
- *Seamless Onboarding:* Simplify the onboarding process with clear guidance and a welcome package.
- *Initial Engagement:* Offer incentives for the first month and conduct personalized check-ins.
- *Proactive Support:* Provide responsive and dedicated support to resolve issues promptly.
- *Value Demonstration:* Clearly communicate the key benefits and set usage milestones with rewards.
- *Feedback and Improvement:* Continuously enhance the onboarding experience based on customer feedback.

**2. Monthly Charges and Dependent Status**

*Finding:* 

Customers with dependents generally pay less monthly, and those without dependents tend to have higher monthly charges. Customers with higher charges generally stay longer.

*Mitigation Strategies:*

- *Family Plans:* Introduce discounted family plans or bundled services.
- *Customizable Packages:* Allow customers to customize packages based on their needs.
- *Incentives for High Usage:* Offer incentives for high-usage customers without dependents.
- *Usage-Based Pricing:* Implement pricing based on actual usage to ensure fairness.
- *Education:* Educate customers about optimizing their plans to get the best value.

**3. Contract Types and Churn**

*Finding:* 

The majority of customers are on Month-to-Month contracts, and these customers churn at higher rates.

*Mitigation Strategies:*

- Incentives for Long-Term Contracts: Offer discounts or bonuses for longer-term commitments.
- Enhance Customer Experience: Improve support and personalize services.
- Loyalty Programs: Reward long-term customers with points, discounts, or exclusive perks.
- Flexible Options: Allow easy plan transitions and upgrades.
- Regular Communication: Keep customers informed and address their concerns regularly.

**4. Impact of Senior Citizenship and Payment Methods**

*Finding:* Senior citizens generally pay more monthly and might be targeted differently. Electronic Check users have a higher churn rate.

*Mitigation Strategies:*

- *Targeted Marketing:* Highlight features appealing to younger demographics and offer exclusive content.
- *Flexible Plans:* Provide customizable plans and incentives for upgrades.
- *Payment Options:* Improve the Electronic Check process and offer a variety of payment options.
- *Educate and Incentivize:* Educate customers about payment methods and offer incentives to switch to more reliable options.

**5. Internet Service Types and Churn**

*Finding:* Fiber optic internet users have higher churn rates compared to DSL and no internet service users.

*Mitigation Strategies:*

- *Address Concerns:* Gather feedback and address issues affecting Fiber optic customers.
- *Incentives:* Offer discounts or rewards to retain Fiber optic customers.

**6. General Observations**

*Finding:* Customers with no dependents have the highest churn rates, and there are high concentrations of customers in major cities.

*Mitigation Strategies:*

- *Targeted Strategies:* Develop city-specific strategies and understand the unique needs of customers with no dependents.
- *Community Engagement:* Build community ties in major cities and enhance local customer experiences.

**Conclusion**

To reduce churn and improve retention, it is crucial to implement targeted strategies addressing the specific needs and behaviors of different customer segments. By focusing on improving onboarding, offering tailored incentives, and enhancing customer support, the company can mitigate churn rates and enhance long-term customer satisfaction. Regular feedback and adaptive strategies will help in continuously refining these approaches for optimal results.







