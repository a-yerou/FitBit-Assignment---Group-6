import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.formula.api as smf
import datetime as dt



def plot_total_distances(data):
    total_distances = data.groupby("Id")["TotalDistance"].sum()
    print(total_distances)
    short_labels = [s[:5] for s in total_distances.index.astype(str)]
    plt.figure(figsize=(8, 4))
    plt.bar(short_labels, total_distances.values)
    plt.xlabel("Users")
    plt.ylabel("Total Distance")
    plt.title("Total Distance Per User")
    plt.tight_layout()
    plt.xticks(rotation=45, ha="right")
    plt.show()


def plot_calories_user(data, id, start=None, end=None):
    data["ActivityDate"] = pd.to_datetime(data["ActivityDate"])

    d = data[data["Id"] == id]

    if start:
        d = d[d["ActivityDate"] >= pd.to_datetime(start)]
    if end:
        d = d[d["ActivityDate"] <= pd.to_datetime(end)]

    d = d.sort_values("ActivityDate")

    plt.plot(d["ActivityDate"], d["Calories"])
    plt.xlabel("Date")
    plt.ylabel("Calories")
    plt.title(f"Calories per day (User {id})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_has_worked_out(data):
    data["ActivityDate"] = pd.to_datetime(data["ActivityDate"])
    data["Weekday"] = data["ActivityDate"].dt.day_name()
    worked_out = data[data["TotalSteps"] > 0]
    counts = worked_out.groupby("Weekday").size()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts = counts.reindex(order)
    plt.figure(figsize=(8, 4))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Day of Week")
    plt.ylabel("Workout Frequency")
    plt.title("Workout frequency per weekday (TotalSteps > 0)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def fit_pooled_model(data):
    model = smf.ols("Calories ~ TotalSteps + C(Id)", data=data).fit()
    print(model.summary())
    return model

def plot_user_regression(data, id):
    d = data[data["Id"] == id]

    model = smf.ols("Calories ~ TotalSteps", data=d).fit()

    plt.figure(figsize=(6,4))
    plt.scatter(d["TotalSteps"], d["Calories"])

    x = sorted(d["TotalSteps"])
    y = model.params["Intercept"] + model.params["TotalSteps"] * pd.Series(x)

    plt.plot(x, y)
    plt.xlabel("TotalSteps")
    plt.ylabel("Calories")
    plt.title(f"User {id}")
    plt.tight_layout()
    plt.show()