import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


def plot_actual_vs_predicted(x_values, actual_values, predicted_values, column_name, title=''):
    # Plot actual values
    plt.figure(figsize=(13, 10))
    plt.scatter(x_values, actual_values, color='b', label='Actual')

    # Plot predicted values
    plt.plot(x_values, predicted_values, color='r', label='Predicted')

    plt.title(f'{column_name}: Actual vs Predicted Values {title}')
    plt.xlabel('X Values')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()


# Load the data
file_path = 'Life_Expectancy_Data.csv'
data = pd.read_csv(file_path)

# Filter data for Israel only
israel_data = data[data['Country'] == 'Israel']

# Plot the Life.expectancy values over Year for Israel

israel_data_filtered = israel_data[
    (israel_data['Life.expectancy'] < 84) & (israel_data['Year'] != 2004)]  # Remove outliers
plt.figure(figsize=(10, 6))
plt.plot(israel_data_filtered['Year'], israel_data_filtered['Life.expectancy'], marker='o', linestyle='-', color='b')
plt.title('Life Expectancy in Israel (2000-2016)')
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.grid(True)
plt.show()

# Linear model to predict Life.expectancy for the next two years
X = israel_data_filtered['Year']
y = israel_data_filtered['Life.expectancy']
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print the model summary
model_summary = model.summary()
print(model_summary)

# Predict life expectancy for 2000 and 2016
future_years = pd.DataFrame({'Year': [2000, 2016]})
future_years = sm.add_constant(future_years)
future_predictions = model.predict(future_years)

# Add predictions to the DataFrame
israel_data_with_predictions = israel_data_filtered.copy()
israel_data_with_predictions = pd.concat(
    [israel_data_with_predictions, pd.DataFrame({'Year': [2000, 2016], 'Life.expectancy': future_predictions})])

# Plot actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(israel_data_filtered['Year'], israel_data_filtered['Life.expectancy'], marker='o', linestyle='-', color='b',
         label='Actual')
plt.plot([2000, 2016], future_predictions, marker='o', linestyle='--', color='r', label='Predicted')
plt.title('Life Expectancy in Israel (Actual vs Predicted)')
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.legend()
plt.grid(True)
plt.show()

print("Yes, the model is linear. The life expectancy is increasing linearly over the years.")
p_value_first_model = model.pvalues[1]  # P-value
print(f"P-value of the first model: {p_value_first_model}")
r_squared_first_model = model.rsquared
print(f"R-squared of the first model: {r_squared_first_model}")  # R of the model

# Calculate residuals for each year
# print the predicted values of 2050
predicted_values = model.predict(sm.add_constant(np.array([[1, 2050]])))
print(f"Predicted life expectancy for 2050: {predicted_values[0]}")

# print the number of months that added evrey year
# Set the year
year = 2020
# Calculate the additional months for the specified year
life_expectancy_start = model.predict(sm.add_constant(np.array([[1, year]])))[0]
life_expectancy_end = model.predict(sm.add_constant(np.array([[1, year + 1]])))[0]
additional_months = (life_expectancy_end - life_expectancy_start) * 12
# It is enough to check on one year because it is linear
# Print the result
print(f"Additional months added to the average life expectancy : {additional_months:.2f} months")

# Plot actual data points
plt.figure(figsize=(10, 6))
plt.scatter(israel_data_filtered['Year'], israel_data_filtered['Life.expectancy'], color='b', label='Actual')

# Plot regression line
plt.plot(israel_data_filtered['Year'], predictions, color='r', label='Regression Line')

# Calculate and plot residuals
residuals = y - predictions
plt.vlines(israel_data_filtered['Year'], predictions, predictions + residuals, colors='g', linestyles='dotted',
           label='Residuals')

plt.title('Life Expectancy in Israel (Actual vs Predicted with Residuals)')
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.legend()
plt.grid(True)
plt.show()

# Filter data for Afghanistan only
afghanistan_data = data[data['Country'] == 'Afghanistan']

# Remove 'Country' and 'Status' columns
afghanistan_data.drop(['Country', 'Status'], axis=1, inplace=True)

# Here is the code that I used to remove the outliers
"""   
for column in columns:
    print(f"Column: {column}")
    # Add a constant to the independent values
    x = sm.add_constant(afghanistan_data['Year'])

    # Create the model
    model = sm.OLS(afghanistan_data[column], x)

    # Fit the model
    results = model.fit()

    # Make predictions
    predictions = results.predict(x)
    plot_actual_vs_predicted(afghanistan_data['Year'], afghanistan_data[column],predictions,column,'before filtering')

    # Calculate the residuals
    residuals_abs =np.abs( afghanistan_data[column] - predictions)
    print(f"Residuals for {column}: {residuals_abs}")
    # Find the index of the point with the largest residual
    max_residual_index = residuals_abs.idxmax()
    threshold = (residuals_abs.mean()) * 2
    # If the largest residual is greater than the threshold, remove the point
    print(f"{residuals_abs[max_residual_index]} > {threshold}: {residuals_abs[max_residual_index] > threshold}")
    if residuals_abs[max_residual_index] > threshold:
        afghanistan_data = afghanistan_data.drop(max_residual_index)

        # Fit the model again without the removed point
        x = sm.add_constant(afghanistan_data['Year'])
        model = sm.OLS(afghanistan_data[column], x)
        results = model.fit()
        predictions = results.predict(x)
        plot_actual_vs_predicted(afghanistan_data['Year'], afghanistan_data[column], predictions ,column ," - after filtering")

        # Calculate the residuals
        residuals_abs =np.abs( afghanistan_data[column] - predictions)
        max_residual_index = residuals_abs.idxmax()
        threshold = (residuals_abs.mean()) * 2
        if residuals_abs[max_residual_index] > threshold:
            print(f"Could not remove the outlier for {column}")
            coloms_to_remove.append(column)
            continue
        else:
            plot_actual_vs_predicted(afghanistan_data['Year'], afghanistan_data[column], predictions,"- pass, The column is linear after filtering")

    else:
        plot_actual_vs_predicted(afghanistan_data['Year'], afghanistan_data[column], predictions,column ,"- pass, The column is linear")

"""
coloms_to_remove = ["percentage.expenditure", "Hepatitis.B", "Measles", "Polio", "Total.expenditure", "Diphtheria",
                    "GDP", "Population", "Alcohol", "thinness.10-19.years", "thinness.5-9.years"]
coloms_to_fix = ["Life.expectancy", "Adult.Mortality"]

afghanistan_data.drop(coloms_to_remove, axis=1, inplace=True)
for column in coloms_to_fix:
    x = sm.add_constant(afghanistan_data['Year'])
    model = sm.OLS(afghanistan_data[column], x)
    results = model.fit()
    predictions = results.predict(x)

    residuals_abs = np.abs(afghanistan_data[column] - predictions)
    max_residual_index = residuals_abs.idxmax()
    afghanistan_data.drop(max_residual_index, inplace=True)

for column in afghanistan_data.columns:
    print(f"Column: {column}")
    # Add a constant to the independent values
    x = sm.add_constant(afghanistan_data['Year'])

    # Create the model
    model = sm.OLS(afghanistan_data[column], x)

    # Fit the model
    results = model.fit()

    # Make predictions
    predictions = results.predict(x)
    plot_actual_vs_predicted(afghanistan_data['Year'], afghanistan_data[column], predictions, column, 'After filtering')

X = afghanistan_data.drop(['Life.expectancy'], axis=1)
y = afghanistan_data['Life.expectancy']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

coefficients = model.params
p_values = model.pvalues
print("\nCoefficients:\n", coefficients, sep='')
print("\nP-values:\n", p_values, sep='')

most_influential_param = coefficients.idxmax()
print(f"\nMost influential parameter: {most_influential_param}")

model_summary = model.summary()
print(model_summary)
