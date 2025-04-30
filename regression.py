import seaborn as seabornInstance  # Standard alias for seaborn
import pandas as pd
from matplotlib.lines import lineStyles
from matplotlib.pyplot import scatter
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt  # Correct import for matplotlib
import numpy
# Load dataset
dataset = pd.read_csv("weather.csv")

# Display dataset shape, a sample, data types, and summary statistics
print(dataset.shape)
print(dataset.sample(5))
print(dataset.dtypes)
print(dataset.describe())

#A scatter plot to observe the correlation between min and max temperatures.
# Plot MinTemp and MaxTemp
dataset.plot(x='MinTemp',y= 'MaxTemp',style='o')

# Add a separate title
plt.title('MinTemp vs MaxTemp')

# Adding labels
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')

# Show plot
plt.show()

#A distribution plot to check the spread of max temperatures.
""" Purpose of a Distribution Plot:
A distribution plot (like seaborn.distplot() or histplot()) is used to:

Understand how values in a column are distributed (e.g., normal, skewed, multimodal).

Identify outliers, skewness, and data spread.

Check for normality—which is often a prerequisite for some statistical models.

🔹 When to Use It:
You can apply a distribution plot to any numerical feature, not just the target. For example:

Use it on input features to check assumptions before applying models.

Use it on the target variable to understand the nature of what you're predicting.

🔸 Example Use Cases:

Variable Type	Why Plot Distribution?
Target	To check if it's normally distributed for regression.
Numeric Feature	To identify skewed features, outliers, or transformation needs.
After Prediction	To compare predicted vs. actual distributions."""

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.histplot(dataset['MaxTemp'])

plt.show()

#data splicing

x=dataset['MinTemp'].values.reshape(-1,1)
y=dataset['MaxTemp'].values.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train) # train the algorithm

print("Intercept ",regressor.intercept_)
#show how significance your input is
print("Coefficient ",regressor.coef_) # the change in the maximum value when the unit of minimum value changes

y_pred=regressor.predict(x_test)

df=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
print(df)

df1=df.head(20)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5' ,color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5' ,color='black')
plt.show()

plt.scatter(x_test,y_test, color='gray')

plt.plot(x_test,y_pred,color='red',linewidth=0.2)
plt.show()

print('Mean absolute error :', metrics.mean_absolute_error(y_test,y_pred))
print('Mean squared Error ',metrics.mean_squared_error(y_test,y_pred))
print('Root mean squared error ',metrics.root_mean_squared_error(y_test,y_pred))