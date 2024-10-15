# %% [markdown]
# # Data Cleaning and Analysis of Various Strains of Coffee using Python
# 
# A project for course "Data Exploration using Python" (STA 4243) at the University of Texas at San Antonio (UTSA)
# 
# Participants: Robert Hall, Max Moran, Ryan Berberek, Dulce Ximena Cid Sanabria

# %% [markdown]
# ## Table of Contents
# 
# 1. Data and Library Importation
# 2. Exploratory Data Analysis & Cleaning
# 3. What palate-related variable has the highest correlation with score?
# 4. Are there statistically significant diferences in palate-related quantifications with respect to diferent countries of origin?
# 5. Do there exist strong correlations between altitude and certain taste quantifications?

# %% [markdown]
# ## Data and Library Importation

# %%
import pandas as pd
coffee = pd.read_csv('coffee_ratings.csv')

# %%
coffee.head()

# %% [markdown]
# ## Exploratory Data Analysis

# %%
coffee.dtypes

# %%
coffee.columns

# %%
for i in coffee.columns:
    if pd.api.types.is_numeric_dtype(coffee[str(i)]):
        print(f"Column {str(i)} Maximum: {coffee[str(i)].max()}")
        print(f"Column {str(i)} Minimum: {coffee[str(i)].min()}")
        print('\n')
    else:
        continue


# %%
print(coffee['species'].value_counts())

# %% [markdown]
# ## What palate-related variable has the highest correlation with score?

# %% [markdown]
# ### Features:
# 
# - Aftertaste 
# - Aroma 
# - Acidity 
# - Body 
# - Balance 
# - Clean Cup 
# - Uniformity 
# - Sweetness
# - Moisture

# %%
coffee.columns

# %%
coffee_salient = coffee[['total_cup_points', 'aftertaste', 'aroma', 'acidity', 'body', 'balance', 'clean_cup', 'uniformity', 'sweetness', 'moisture']]

# %%
coffee_salient.head()

# %%
coffee_salient.isnull().sum()

# %%
features = coffee_salient[['aftertaste', 'aroma', 'acidity', 'body', 'balance', 'clean_cup', 'uniformity', 'sweetness', 'moisture']]
labels = coffee_salient[['total_cup_points']]

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, test_size=0.2)

# %%
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
model = mlr.fit(x_train, y_train)
points_pred = model.predict(x_test)

# %%
print(features.columns)


# %%
print(model.coef_)

# %%
coefs = []
for subset in model.coef_:
    for coef in subset:
        coefs.append(round(coef, 4))

#print(coefs)

cols = [col for col in features.columns]
#print(cols)

feature_coefs = pd.DataFrame({'features': cols,
                              'coefficients': coefs})

feature_coefs.sort_values('coefficients', ascending=False)
feature_coefs

# %% [markdown]
# ## Are there statistically significant diferences in palate-related quantifications with respect to diferent countries of origin?

# %% [markdown]
# In order to reduce overcomplication, only six nations of the many surveyed were chosen for this study. Countries were not randomly selected, and chosen based on geographical representation. 

# %%
coffee['country_of_origin'].value_counts()

# %%
top8 = coffee[coffee['country_of_origin'].isin(['Mexico',  
                                         'Brazil', 
                                         'Taiwan',
                                         'Ethiopia', 
                                         'Tanzania, United Republic Of',
                                         'Indonesia'])]

top8['country_of_origin'].replace('Tanzania, United Republic Of', 'Tanzania')

top8['country_of_origin'].value_counts()

# %%
from statsmodels.stats.multicomp import pairwise_tukeyhsd
sig = 0.05 # significance threshold
tukey_results = pairwise_tukeyhsd(top8['aftertaste'], top8['country_of_origin'], sig)
print(tukey_results)
# under the “reject” column, if True, then there is a significant difference; if False, no significant difference.

# %%
from statsmodels.stats.multicomp import pairwise_tukeyhsd
sig = 0.05 # significance threshold
tukey_results = pairwise_tukeyhsd(top8['acidity'], top8['country_of_origin'], sig)
print(tukey_results)
# under the “reject” column, if True, then there is a significant difference; if False, no significant difference.

# %%
from statsmodels.stats.multicomp import pairwise_tukeyhsd
sig = 0.05 # significance threshold
tukey_results = pairwise_tukeyhsd(top8['aftertaste'], top8['country_of_origin'], sig)
print(tukey_results)
# under the “reject” column, if True, then there is a significant difference; if False, no significant difference.

# %% [markdown]
# ## Do there exist correlations between altitude and certain taste quantifications?

# %% [markdown]
# Data on altitude was extremely messy -- lots of variation in hyphens, where units of measurement were placed, etc. All measurements seem to be consistently in meters. Each instance of the new altitude column is the minimum altitude, both for ease in data cleaning, and to keep the estimates conservative.

# %%
coffee.dropna(subset=['altitude'])
coffee['altitude'] = coffee['altitude'].str[:4]
coffee['altitude'] = coffee['altitude'].apply(lambda x: str(x)[:4] if str(x)[:4].isdigit() else None)
coffee['altitude'] = pd.to_numeric(coffee['altitude'], errors='coerce').astype('Int64')
coffee['altitude'].head()

# %%
coffee_salient = coffee[['altitude', 'aftertaste', 'aroma', 'acidity', 'body', 'balance', 'clean_cup', 'uniformity', 'sweetness', 'moisture']]
feature = coffee['altitude']
labels = coffee[['aftertaste', 'aroma', 'acidity', 'body', 'balance', 'clean_cup', 'uniformity', 'sweetness', 'moisture']]

# %% [markdown]
# Since data has missing values, we will fill the 'altitude' column using the mean of the altitude.

# %%
from sklearn.impute import SimpleImputer
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.2, train_size=0.8, random_state=27)

imputer = SimpleImputer(strategy='mean')

x_train = np.array(x_train).reshape(-1,1)
x_train = imputer.fit_transform(x_train)

x_test = np.array(x_test).reshape(-1,1)
x_test = imputer.fit_transform(x_test)

# %%
mlr = LinearRegression()
model = mlr.fit(x_train, y_train)
points_pred = model.predict(x_test)

# %%
labels.columns

# %%
model.coef_

# %%
coefs = []
for subset in model.coef_:
    for coef in subset:
        coefs.append(coef)

#print(coefs)

cols = [col for col in features.columns]
#print(cols)

feature_coefs = pd.DataFrame({'features': cols,
                              'coefficients': coefs})

feature_coefs.sort_values('coefficients', ascending=False)
feature_coefs

# %% [markdown]
# ## Overall Takeaways and Answers

# %% [markdown]
# #### **What palate-related variable has the highest correlation with score?**
# 
# * 'aftertaste' has the highest correlation with score. The scikit-learn linear regression coefficient score for aftertaste is approximately 1.967

# %% [markdown]
# #### **Are there statistically significant diferences in taste quantifications with respect to diferent countries of origin?**
# 
# Statistically significant differences were found between the scores of the following nations under the forementioned categories:
# 
# (Bold text indicates the nation(s) with significantly higher scores than their counterpart nations in their respective categories)
# 
# aftertaste:
# 
# - **Brazil** and Ethiopia
# - Brazil and **Mexico**
# - Ethiopia and **Indonesia**, **Mexico**, **Taiwan** and **Tanzania**
# - **Mexico** and Taiwan, Tanzania
# 
# acidity:
# 
# - **Brazil** and Ethiopia
# - Ethiopia and **Indonesia**, **Mexico**, **Taiwan** and **Tanzania**
# 
# aftertaste:
# 
# - **Brazil** and Ethiopia
# - Brazil and **Mexico**
# - Ethiopia and **Indonesia**, **Mexico**, **Taiwan** and **Tanzania**
# - **Mexico** and Taiwan, Tanzania
# 
# A significance threshold of a = 0.05 was used. ANOVA using a Tukey range test was applied to achieve the p-scores. 

# %% [markdown]
# #### **Do there exist strong correlations between altitude and certain taste quantifications?**
# 
# * The coefficients outputted by the model are quite small, likely due to a lack of feature scaling or incidence of multicollinearity, not within the scope of the course for which this project is created.
# 
# * In relative terms, altitude puts downward pressure on the scores of 'aftertaste' and 'body', and places upward pressure on balance, uniformity, acidity, and the propensity for the coffee cup to be clean after drinking to completion.


