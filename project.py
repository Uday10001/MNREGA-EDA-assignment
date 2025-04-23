import pandas as pd
df = pd.read_csv('Base Dataset1.csv')


# =========================PreProcessing==============================#
yr = df.columns.get_loc('fin_year')
df_split = df['fin_year'].str.split(pat='-', expand=True, n=1).add_prefix('fin_year_')
df = pd.concat([df.iloc[:, :yr], df_split, df.iloc[:, yr:]], axis=1)
df = df.drop(columns=['fin_year'])
df = df.rename(columns={'fin_year_0': 'Fiscal Year'})
df = df.drop(columns=['fin_year_1'])
df = df.astype({'Fiscal Year': 'int32'})

# Fill zeroes in wage rate with district average
df['Average_Wage_rate_per_day_per_person'] = df.groupby('district_name')['Average_Wage_rate_per_day_per_person'].transform(lambda x: x.replace(0, x[x > 0].mean()))
# Fill Average_Wage_rate_per_day_per_person with zero where Average_days_of_employment_provided_per_Household is zero
df.loc[df['Average_days_of_employment_provided_per_Household'] == 0, 'Average_Wage_rate_per_day_per_person'] = 0
# Drop column: 'Remarks'
df = df.drop(columns=['Remarks'])

#Drop Duplicate columns
df = df.drop_duplicates()



print(df.info())
print(df.describe())
print(df.head())  


import matplotlib.pyplot as plt
import seaborn as sb

# =========================Graph Plots=================================#

# 1. States with highest average Approved Labor Budget 
# === Visualizations ===

# 1. Bar Plot: Average Labour Budget by State
plt.figure(figsize=(12, 6))
avg_labour = df.groupby("state_name")["Approved_Labour_Budget"].mean().sort_values(ascending=False)
sb.barplot(x=avg_labour.index, y=avg_labour.values, palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Average Labour Budget by State")
plt.tight_layout()
plt.show()

# 2. Box Plot: Total Expenditure Bistribution by state  
plt.figure(figsize=(16, 8))
sb.boxplot(
    data=df, x='state_name', y='Total_Exp',palette='Spectral'
)
plt.xticks(rotation=90)
plt.title("Total Expenditure Distribution by State and Fiscal Year")
plt.xlabel("State Name")
plt.ylabel("Total Expenditure")
plt.tight_layout()
plt.show()


# 3. Line Graph: Yearly trends
df.groupby("Fiscal Year")["Total_No_of_Works_Takenup"].mean().plot(label='Works Taken Up')
df.groupby("Fiscal Year")["Number_of_Completed_Works"].mean().plot(label='Works Completed')
plt.title("Yearly Trend of Works Taken Up and Completed")
plt.xlabel("Fiscal Year")
plt.ylabel("Average Works")
plt.legend()
plt.show()

#4. Violin Plot: Wage Rate Distribution
sb.violinplot(x='state_name', y='Average_Wage_rate_per_day_per_person', data=df)
plt.xticks(rotation=90)
plt.title("Wage Rate Distribution by State")
plt.tight_layout()
plt.show()


# 5. Stacked Bar Plot: NRM & Agriculture Expenditure
state_group = df.groupby('state_name')[['percent_of_NRM_Expenditure', 'percent_of_Expenditure_on_Agriculture_Allied_Works']].mean()
state_group.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
plt.title("Expenditure on NRM and Agriculture per State")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 6. Histogram: Employment Provided
plt.figure(figsize=(10, 6))
sb.histplot(df['Average_days_of_employment_provided_per_Household'], bins=30, kde=True, color='coral')
plt.title("Histogram of Employment Days per Household")
plt.xlabel("Average Days")
plt.ylabel("Frequency")
plt.show()

# 7. Pie Chart: Completed Works by a Few Top States
top_states = df['state_name'].value_counts().nlargest(5).index
pie_data = df[df['state_name'].isin(top_states)].groupby('state_name')['Number_of_Completed_Works'].sum()
plt.figure(figsize=(8, 8))
pie_data.plot.pie(autopct='%1.1f%%', startangle=140, colors=sb.color_palette("pastel"))
plt.title("Completed Works Distribution Among Top 5 States")
plt.ylabel("")
plt.tight_layout()
plt.show()

# 8. Correlation Matrix
correlation_cloumns = ["Wages", "Approved_Labour_Budget", "Average_Wage_rate_per_day_per_person", "Total_Exp", "Total_No_of_Active_Workers", "Total_No_of_Works_Takenup"]
plt.figure(figsize=(14, 10))
corr_matrix = df[correlation_cloumns].corr()
sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#9. Scatter Plot: Total no of works taken up vs to number of workers

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X = df[['Total_No_of_Works_Takenup']]
y = df['Total_No_of_Workers']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("\n--- Linear Regression to Predict Total Number of Workers ---")
print("MAE:", mean_squared_error(y_test, y_pred))

sb.scatterplot(df, x = "Total_No_of_Works_Takenup", y="Total_No_of_Workers", hue = "Fiscal Year", palette = "cool")
plt.xlabel("Number of Works Taken up")
plt.ylabel("Total number of workers")
plt.plot(X_test, y_pred, color = 'r')
plt.title("Works Taken Up vs Total Workers with Regression Line")
plt.show()



# =======Regression model to predict the average wage rate======
features = [
    "Wages",
    'Average_days_of_employment_provided_per_Household',
    'Average_Wage_rate_per_day_per_person',
    'Approved_Labour_Budget',
]
target = 'Total_Exp'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse =  mean_squared_error(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {rmse:.2f}")
