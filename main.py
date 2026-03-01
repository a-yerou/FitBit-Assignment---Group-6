import numpy as np

import scripts.Part1 as p1
import scripts.Part3 as p3
import scripts.Part4 as p4
import scripts.data_loaders as loaders
import pandas as pd

DB_PATH = "data/fitbit_database.db"
API_KEY = "JZ6LCG93XVTAQMHX4Y9DLA46B" #we don't mind having the API key like this in the open, it's a free one anyway

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


p3.weather_vs_activity(API_KEY, DB_PATH)


p4.load_weight_log(DB_PATH)


daily_activity = loaders.load_daily_activity(DB_PATH)
minute_sleep = loaders.load_minute_sleep(DB_PATH)
hourly_steps = loaders.load_hourly_steps(DB_PATH)
hourly_calories = loaders.load_hourly_calories(DB_PATH)
hourly_intensity = loaders.load_hourly_intensity(DB_PATH)
heart_rate = loaders.load_heart_rate(DB_PATH)
weight_log = loaders.load_weight_log(DB_PATH)

person_day = loaders.build_person_day(daily_activity, minute_sleep, weight_log)
loaders.coverage_report(person_day)

sleep_eligible = loaders.sleep_eligible(person_day)

person_hour = loaders.build_person_hour(hourly_steps, hourly_calories, hourly_intensity, heart_rate)

p4.weekend_effect(person_day)
p4.relation_activity_sleep(person_day)
p4.weather_relation(daily_activity, API_KEY)


