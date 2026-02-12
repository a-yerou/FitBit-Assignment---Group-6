import scripts.Part1 as p1
import pandas as pd

data = pd.read_csv("data/daily_acivity.csv")

print("Amount of users:", data["Id"].nunique())

p1.plot_total_distances(data)

p1.plot_calories_user(data, 1503960366)

p1.plot_has_worked_out(data)

p1.fit_pooled_model(data)

example_user = data["Id"].iloc[0]
p1.plot_user_regression(data, example_user)