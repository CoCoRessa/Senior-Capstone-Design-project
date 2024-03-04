![image](https://github.com/CoCoRessa/Senior-Capstone-Design-project/assets/154608668/16f8be5c-1ab5-4edb-8e65-5fac17ee8381)Title: An Analysis of Influencing Factors on Firm’s Labor Productivity in Bangladesh using Explainable Artificial Intelligence(XAI)
======================
Senior Capstone Design Project

# Abstract
Bangladesh was once one of the Asian Least Developed Countries, but since the 2000s, it has been rapidly growing with an annual average economic growth rate of around 6-7%, based on its abundant and affordable labor force. However, despite this rapid economic growth, the standard of living in Bangladesh remains poor. To expect an overall improvement in the quality of life in Bangladesh, it's crucial that the country moves beyond mere rapid growth and strives for sustained economic development. Sustainable economic growth involves various economic entities, and among them, the growth of enterprises generally has a significant impact. Despite the importance of the role of enterprises, there is a lack of research and up-to-date insights into the current trends of Bangladesh enterprises. Therefore, this study aims to identify the current situation of Bangladesh companies and draw policy implications for corporate growth by analyzing the influencing factors of corporate labor productivity, an indicator that can measure corporate growth using Explainable Artificial Intelligence(XAI)

# 1. Utilized Data
## To analyze the internal environment of an enterprise, Enterprise Survey are utilized
### - The reasons for using the Bangladesh Enterprise Survey
- Despite the lack of recent data in Bangladesh, it provides the most current data (2022) on the business environment
- The Enterprise Survey consists of over 300 specific survey items, allowing for a realistic assessment of the situation in Bangladesh
### - Bangladesh Enterprise Survey Overview
- The survey conducted by WBES (World Bank Enterprise Surveys) targeted companies in the manufacturing and service sectors
- The purpose is to assess general company characteristics, infrastructure, sales and supply, competition, innovation, land and permits, finance, firm-government relations, exposure to bribery, labor, and performance
- Exists as sampling data not a complete review on every

## To analyze the external environment, such as the characteristics of the region where a enterprise is located, Boost Data and Economic Census are utilized
### - Boost Data Overview
- Data depicting the status of budget and financial transactions in Bangladesh, categorized by executing departments, administrative regions, economic impacts, and thematic sectors
### - Economic Census Overview
- A survey conducted nationwide targeting economic entities (establishments and households) to understand the industrial structure and distribution in Bangladesh

# 2. Sampling Methodology
## The total sample size was determined based on Bangladesh's Gross National Income (GNI)
### - Determine the total sample size based on the 2016 Bangladesh GNI
- In 2016, the GNI was $435 billion, resulting in a sample size of 1,000
- Using the Business Directory 2019 as a sampling frame, 1,000 samples were drawn from a list of 18,102 entries
- The Business Directory 2019 serves as the sampling frame produced by the Bangladesh Bureau of Statistics
### - Based on stratified random sampling, it was divided into three strata
1. Industry: 7 categories: Garments, Textiles, Food, Other manufacturing, Retail, Hotels, Other Services
2. Region: 7 categories: Dhaka MA, Greater Dhaka, Chattogram, Cox’s Bazar, Rajshahi, Khulna, Sylhet, Barisal
3. Size: 3 categories: Small, Medium, Large
### - The sample sizes for each stratum were determined using the size of the sample frame, aiming for a confidence interval of 90% and precision between 5% to 7.5%
▶ It can be said to have representativeness for each stratum

# 3-1. The analyzable scope of the data 1) Industry
## Five industries(Garments, Textiles, Food, Hotel and Restaurants, Retail) surveyed in all two years with sufficient data counts were chosen as the subjects for analysis
- When there is a small number of data for a specific industry, there may be representativeness, but there is a possibility of a large margin of error in estimating the population
- From 2013 to 2022, the proportion of the Services industry increased, with higher proportions observed in the Hotel and Restaurant industry as well as the Retail industry
- In the manufacturing sector, the survey proportions for Garments, Textiles, and Food industries were high in 2013, and 2022
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/eb280168-c95c-4f38-ac25-08938e51958f" width="40%" height="40%" />
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/1425a017-f201-41b4-9ee5-e4fbb4c16518" width="40%" height="40%" />

# 3-2. The analyzable scope of the data 2) Spatial
## The survey area expanded to reflect urbanization, setting the urban areas as the spatial scope of the study, enabling analysis and conclusions regarding enterprises
- As the surveyed enterprises are not distributed evenly across all regions, it's essential to examine the characteristics of the areas where these enterprises are located
- The comparison of surveyed locations between 2013 and 2022 shows a gradual expansion of the surveyed areas
- Comparing the distribution of Night Time Light (NTL), a proxy for urbanization, to the distribution of surveyed areas, we found a similar distribution
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/83b2d5c3-8aae-4ef6-b296-b010252a011c" width="40%" height="40%" />
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/88ddbf80-3460-43a4-84ea-c0743eaeab1e" width="40%" height="40%" />

# 4. Labor Productivity
## To measure enterprise performance, labor productivity is used
- Labor productivity is an important metric that measures efficiency and performance in a company's operational activities by comparing output to input
- Labor productivity in this study = (Real total annual sales) /(Total employees)
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/67e90b50-a715-4a99-9fcd-cfd1ff3511da" width="70%" height="70%" />

# 5. Analysis Methodology - Machine Learning
## Machine learning is a field of AI that applies mathematical techniques to analyze patterns in data, minimizing prediction errors of algorithms, and providing reliable predictions
- Machine learning is divided into supervised, unsupervised, and reinforcement learning. In this study, we employ supervised learning, which predicts Y (dependent feature) from X (independent feature)
- Machine learning methodologies offer higher explanatory power compared to traditional regression models. They have the advantage of analyzing complex interactions between features exhibiting non-linearity
- In this study, the dependent feature is the labor productivity of companies, and the independent features are the survey items from the Enterprise Survey
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/d159614c-0231-4b3f-8bc4-2d761c857c11" width="70%" height="70%" />

## Based on Gradient Boosting and automatically handling missing values, the study employed XGBoost, LightGBM, and CatBoost, known for their excellent explanatory power
### - Algorithm used in step 1: XGBoost & LightGBM & CatBoost
- Gradient Boosting is a technique that minimizes residuals by iteratively creating new Decision Trees based on the residuals left after training the data with Decision Trees
- The optimal algorithm can vary depending on the data and circumstances
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/e3439c72-606a-4e1f-a2eb-dbe44e8048b9" width="70%" height="70%" />

## Utilizing feature selection techniques to identify the optimal feature combinations
### - The techniques used in step 2: Feature Selection 
- Feature Selection is a method of identifying subsets of data from the original dataset that exhibit the best performance to enhance the accuracy of the model
- Each year, the survey questions exceed 300, making it difficult for the user to consider all combinations of features
- ▶ Use feature selection techniques to find the optimal combination of features
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/e2732dc0-d60d-4b04-956b-b54e75100b82" width="50%" height="50%" />

## Utilizing Explainable Artificial Intelligence(XAI) to determine the individual contribution of each explanatory feature to the model's predictions
### - Algorithm used in step 3: Explainable Artificial Intelligence(XAI)
- Machine learning methodologies offer higher interpretability compared to traditional regression models, yet models trained with algorithms are often regarded as black boxes, making it challenging to discern how much each explanatory features contributes to the model's predictions
- ▶ With the advancement of XAI, the interpretation of black-box models has become possible. Consequently, it has begun to be utilized in urban and transportation planning domains
### - SHAP Value
- One of the XAI methodologies, based on Shapley Values, calculates the importance of each feature in a model's prediction by determining how much each feature contributes to that prediction (contribution calculation)
- When a specific feature is included or excluded in the model, the difference in predicted values is used to calculate the contribution of the feature
- SHAP Values should not be interpreted as causal relationships
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/36637d79-a640-40d7-adb0-b9d62235812d" width="90%" height="90%" />

## Through the SHAP value plot, the influence of each explanatory features on dependent features can be understood
### - SHAP Value plot example
- The SHAP Value Bar Plot is a bar graph that represents the absolute SHAP values for each feature
- The SHAP Value Summary Plot represents the impact of features by displaying all data points as dots, representing the direction and magnitude of the features' influence
- The sign of SHAP Value indicates how a particular feature influences the predicted value
- ▶ When it's positive, it means the feature tends to increase the model's predicted value, and when it's negative, the feature tends to decrease the model's predicted value
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/c5a74eeb-7d08-49f6-8332-092c903b4af3" width="40%" height="40%" />
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/840a62a8-f031-4cce-8e67-8391c3d42147" width="40%" height="40%" />

# 6. Research Flow Chart
## Step 1: Determining the most suitable model for the dataset
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/aa7183e2-8a6d-4ca8-a917-d99a79dcef0a" width="70%" height="70%" />

## Step 2: Finding the best combination of explanatory that best describe labor productivity (dependent features) - Feature selection
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/3185d0fd-a264-4a23-9050-2db2a25ba970" width="90%" height="90%" />

## Step 3: Analyzing the factors influencing labor productivity based on the final model and the final combination of explanatory features
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/7105866d-44a6-47ec-81f8-172e0f40e42e" width="70%" height="70%" /> <br>
- Using an Alluvial plot, the comparison and interpretation of the factors influencing corporate labor productivity between 2013 and 2022 can be facilitated by categorizing entities and visualizing their temporal trends and compositional ratios

# 7. Research Result
## Step 1 result: XGBoost, which exhibited the most outstanding performance across all years, has been chosen as the final analytical model
### - Training the machine learning model
- Each year, survey questions with more than 70% null values were removed, and the dataset was split into training data, which constitutes 80% of the full dataset, and test data, which represents 20%
- The evaluation results of the XGBoost, LightGBM, and CatBoost models are presented in Table 1, showing that XGBoost appears to be the most suitable for predicting annual labor productivity
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/3368742b-2c22-46cb-8778-683ab1907d72" width="90%" height="90%" /> <br>

### - XGBoost Hyper Parameter
1. learning_rate: Tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function
2. max_depth: The maximum depth of a decision tree
3. min_child_weight: The total sum of weights needed to decide whether to add branches in a decision tree
4. colsample_bytree & subsample: A parameter to prevent overfitting by controlling the excessive complexity of the trees being generated
5. reg_alpha & reg_lambda: Regularization parameter to prevent overfitting
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/8199bd12-574a-4df8-958d-4f7cdbca9370" width="90%" height="90%" />

## Step 2 result: Using a total of 12 different feature selection approaches, the final set of explanatory features was determined to be 28 for 2013 and 30 for 2022
### - Feature Selection result
- Feature selection was performed using the SHAP Value method, which determines the contribution of each feature
- The Backward Elimination technique was employed, where at each step, the number of features to remove was set to 1, 3, 5, 10, 1%, 3%, 5%, and 10% of the total number of features, resulting in a total of 8 cases
- Using the Forward Selection technique, the number of features to add at each step was set to 1, 3, 5, and 10, resulting in a total of 4 cases
- After obtaining the optimal feature combinations at the moments with the lowest RMSE in each of the total 12 cases, select the features that commonly exist in the majority of cases as the final choice
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/2720794a-cdcd-45e9-8e93-93529dbe0182" width="90%" height="90%" />

## Step 3 result: Influencing factors on firm’s Labor productivity 
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/e228f628-d99d-44c4-9c19-a7b856340afc" width="90%" height="90%" /> <br>
- The left side of the figure shows the key variables influencing on labor productivity in 2013, and the right side shows the key variables influencing on labor productivity in 2022
- The line thickness is proportional to the impact of the variable on labor productivity
- 2013, 2022 influencing factors are sorted in order of importance score

### - Impact of infrastructure, regional investments on labor productivity declines in 2022
- In 2022, due to well-developed infrastructure and sufficient regional investments compared to 2013, the influence on labor productivity has decreased
- 【Infrastructure】: In both 2013 and 2022, labor productivity increases with better electricity infrastructure in the region, and in 2022, labor productivity increases with better access to port facilities
- 【Regional investment】: In both 2013 and 2022, we find that higher investment in education and skills improves the overall quality of labor in a region, making it easier for firms to find skilled workers and increasing labor productivity
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/049160ed-57e4-45cb-8b26-b6d7a4fd7957" width="90%" height="90%" />

### - Operating costs, labor, human capital, and firm information remain crucial in both 2013 and 2022
- Capital and labor are traditionally the key factors in determining labor productivity, and entrepreneurship and the history of a firm are key factors in the operation of a traditional business
- 【Operating cost】: In both 2013 and 2022, labor productivity tends to increase with greater firm operating costs
- 【Labor & Human capital】: Quality of labor determines labor productivity in both 2013 and 2022
- 【Firm Information】: In both 2013 and 2022, older firms are estimated to have increased labor productivity due to accumulated know-how
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/c36ca2bf-f324-47f9-bf53-77b1eca890e2" width="90%" height="90%" />

### - Increasing importance of finance and urbanization to labor productivity in 2022
- Estimated to have increased financial importance as economic growth boosts market activity and investment. In addition, urbanization has made cities better places to do business due to expanding infrastructure and labor markets
- 【Finance】: By 2022, labor productivity is estimated to be higher when firms have diverse or flexible sources of financing
- 【Urbanization】: In 2022, the relationship between Night Time Light (NTL), which is a proxy for urbanization, and labor productivity was checked, and it was estimated that the more urbanized the country, the higher the labor productivity due to a better labor supply and a better business environment with good infrastructure and services
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/6d11ab13-e4e1-4da3-97e7-45b2e235e2bc" width="90%" height="90%" />

# 7. Conclusion & Limitations
## Conclusion
- Using XAI, 28 key features were reliably selected in 2013, and 30 in 2022, out of a pool of 300 explanatory variables, to identify the core factors influencing labor productivity in enterprises
- Incorporating the changes from the past, it is confirmed that in 2022, finance and urbanization have newly influenced labor productivity
## Main Takeaway
- From the Enterprises's perspective, Due to variations in labor productivity influencing factors across industries and over time from the perspective of businesses, it is necessary to develop differentiated strategies through regular research and analysis. Additionally, funding sources should be diversified and managed flexibly, while operational costs of companies need to be rigorously controlled
- From the government's perspective, policies aimed at enhancing labor productivity should focus on improving the qualitative level of workers, along with continuous investments in electrical and port infrastructure, as well as urban development
## Limitations
- While annual enterprise data is representative, it is not panel data, thus posing limitations on precise comparisons
- In the 2022 model, due to the absence of Economic Census data, external environmental variables for companies could not be included
- In Step 2, the Feature Selection stage, we were unable to proceed with considering more cases than the 12 scenarios initially planned
- Analysis considering the varying characteristics of industries and the ripple effects of associated industries based on spatial factors may be insufficiently addressed





















