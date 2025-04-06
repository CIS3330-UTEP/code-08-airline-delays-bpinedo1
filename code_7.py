import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
#If any of this libraries is missing from your computer. Please install them using pip.

filename = 'Flight_Delays_2018.csv'
df = pd.read_csv(filename)
#Descriptive Analytics
#df= df.dropna(subset=['DEP_DELAY', 'ARR_DELAY']) #AI generated
#sns.scatterplot(x='DEP_DELAY', y='ARR_DELAY', data=df)
#plt.xlabel('Departure delay')
#plt.ylabel('Arrival delay')
#plt.show()

#Prescriptive Analytics
x= df['DEP_DELAY']
y= df['ARR_DELAY']
x= sm.add_constant(x)
model= sm.OLS(y, x).fit()
print(model.summary())

#ARR_DELAY is the column name that should be used as dependent variable (Y).
#Chat-GPT(4o).Date of query(4/6/25)."How do i remove null values from a data set in python." Generated using OpenAI Chat-GPT. https://chatgpt.com/