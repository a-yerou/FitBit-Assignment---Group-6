import numpy as np

import scripts.Part1 as p1
import scripts.Part3 as p3
import pandas as pd

#Part 1
data = pd.read_csv("data/daily_acivity.csv")

print("Amount of users:", data["Id"].nunique())

p1.plot_total_distances(data)

p1.plot_calories_user(data, 1503960366)

p1.plot_has_worked_out(data)

p1.fit_pooled_model(data)

example_user = data["Id"].iloc[0]
p1.plot_user_regression(data, example_user)



#Part 3
DB_PATH = "data/fitbit_database.db"
counts = data["Id"].value_counts()
classified_data = counts.rename_axis("Id").reset_index(name = "n")
classified_data["Class"] = np.select([classified_data["n"] <= 10, classified_data["n"] >= 16], ["Light", "Heavy"], default = "Moderate")
classified_data = classified_data[["Id", "Class"]]



p3.regress_activity_on_sleep(1503960366, DB_PATH)

p3.analyse_sed_sleep(DB_PATH)
p3.barplot_steps(DB_PATH)
p3.barplot_calories(DB_PATH)
p3.barplot_minutes_sleep(DB_PATH)

p3.plot_hr_and_intensity(2022484408, DB_PATH)

API_KEY = "JZ6LCG93XVTAQMHX4Y9DLA46B" #we don't mind having the API key like this in the open, it's a free one anyway

p3.weather_vs_activity(API_KEY)




