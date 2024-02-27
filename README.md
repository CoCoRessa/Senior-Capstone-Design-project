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
### Bangladesh Enterprise Survey Overview
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

# Research Flow Chart
## Step 1: Determining the most suitable model for the dataset
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/aa7183e2-8a6d-4ca8-a917-d99a79dcef0a" width="70%" height="70%" />

## Step 2: Finding the best combination of explanatory that best describe labor productivity (dependent features) - Feature selection
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/3185d0fd-a264-4a23-9050-2db2a25ba970" width="90%" height="90%" />

## Step 3: Analyzing the factors influencing labor productivity based on the final model and the final combination of explanatory features
<img src="https://github.com/CoCoRessa/CoCoRessa/assets/154608668/7105866d-44a6-47ec-81f8-172e0f40e42e" width="70%" height="70%" /> <br>
- Using an Alluvial plot, the comparison and interpretation of the factors influencing corporate labor productivity between 2013 and 2022 can be facilitated by categorizing entities and visualizing their temporal trends and compositional ratios

# Research Result
## Step 1 result: XGBoost, which exhibited the most outstanding performance across all years, has been chosen as the final analytical model




