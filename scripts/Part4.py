import sqlite3
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

DB_PATH = "../data/fitbit_database.db"



def load_weight_log(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT Id, Date, WeightKg, WeightPounds, Fat, BMI, IsManualReport, LogId FROM weight_log")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)

    df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("int64")
    df["LogId"] = pd.to_numeric(df["LogId"], errors="coerce").astype("int64")

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df["day"] = df["Date"].dt.date

    mask = df["WeightKg"].isna() & df["WeightPounds"].notna()
    repaired = int(mask.sum())
    df.loc[mask, "WeightKg"] = df.loc[mask, "WeightPounds"] / 2.2046226218

    df["flag_bad_weight"] = False
    df.loc[(df["WeightKg"].notna()) & ((df["WeightKg"] < 30) | (df["WeightKg"] > 250)), "flag_bad_weight"] = True

    df["flag_bad_bmi"] = False
    df.loc[(df["BMI"].notna()) & ((df["BMI"] < 10) | (df["BMI"] > 60)), "flag_bad_bmi"] = True

    fat_missing = float(df["Fat"].isna().mean())
    print("\n weight_log summary:")
    print("weight_log rows:", len(df))
    print("unique users:", df["Id"].nunique())
    print("date range:", df["Date"].min(), "to", df["Date"].max())
    print("repaired WeightKg:", repaired)
    print("fat_missing_pct:", fat_missing)
    print("Factor of missing values per column:")
    print((df.isna().sum() / len(df)).sort_values(ascending=False))
    print("The column 'Fat' will be left out in further cases due to it's high NA content.")

    return df


def weekend_effect(person_day):
    df = person_day.dropna(subset=["TotalSteps", "is_weekend"]).copy()

    out = df.groupby("is_weekend", as_index=False).agg(
        mean_steps=("TotalSteps","mean"),
        sd_steps=("TotalSteps", lambda x: x.std(ddof=1)),
        n_days=("TotalSteps","size"),
        n_users=("Id","nunique")
    )
    out["se"] = out["sd_steps"] / (out["n_days"] ** 0.5)

    print("\n weekend effect (overall)")
    print(out)

    plt.figure(figsize=(6,4))
    plt.bar(out["is_weekend"].astype(str), out["mean_steps"], yerr=out["se"])
    plt.title("Mean steps: weekday vs weekend (± SE)")
    plt.xlabel("is_weekend")
    plt.ylabel("Mean steps")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,4))
    df2 = df.copy()
    df2["label"] = df2["is_weekend"].map({False:"Weekday", True:"Weekend"})
    plt.boxplot([df2[df2["label"]=="Weekday"]["TotalSteps"], df2[df2["label"]=="Weekend"]["TotalSteps"]], labels=["Weekday","Weekend"])
    plt.title("Steps distribution: weekday vs weekend")
    plt.ylabel("Steps")
    plt.tight_layout()
    plt.show()

    return out

def relation_activity_sleep(person_day, min_days=10):
    df = person_day.dropna(subset=["sleep_minutes"]).copy()
    counts = df.groupby("Id").size()
    keep = counts[counts >= min_days].index
    df = df[df["Id"].isin(keep)]

    print("sleep relation coverage")
    print("eligible users:", int(len(keep)))
    print("eligible rows:", int(len(df)))

    model = smf.ols("sleep_minutes ~ active_minutes + C(Id)", data=df).fit()
    print(model.summary())

    plt.figure(figsize=(6,4))
    plt.scatter(df["active_minutes"], df["sleep_minutes"], alpha=0.6)
    plt.title("Sleep vs activity (eligible users only)")
    plt.xlabel("Active minutes")
    plt.ylabel("Sleep minutes")
    plt.tight_layout()
    plt.show()

    return model

def weather_relation(df_daily_activity, API_KEY, city="Chicago"):
    df_a = df_daily_activity.copy()
    df_a = df_a.dropna(subset=["Id", "day", "TotalSteps"])

    df_a["day"] = pd.to_datetime(df_a["day"]).dt.date
    dmin = df_a["day"].min()
    dmax = df_a["day"].max()

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{dmin}/{dmax}"
    params = {"unitGroup": "us", "include": "days", "key": API_KEY, "contentType": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df_w = pd.DataFrame(data.get("days", []))
    df_w["day"] = pd.to_datetime(df_w["datetime"], errors="coerce").dt.date
    df_w = df_w[["day", "temp", "precip"]].dropna(subset=["day"])

    df = df_a.merge(df_w, on="day", how="inner").dropna(subset=["TotalSteps", "temp", "precip"])

    df_day = df.groupby("day", as_index=False).agg(
        mean_steps=("TotalSteps", "mean"),
        mean_active=("active_minutes", "mean") if "active_minutes" in df.columns else ("TotalSteps", "mean"),
        temp=("temp", "mean"),
        precip=("precip", "mean"),
        n_users=("Id", "nunique"),
        n_rows=("Id", "size")
    ).sort_values("day")

    m_temp = smf.ols("mean_steps ~ temp", data=df_day).fit()
    m_precip = smf.ols("mean_steps ~ precip", data=df_day).fit()
    m_both = smf.ols("mean_steps ~ temp + precip", data=df_day).fit()

    print("\n weather_relation summary")
    print("city:", city)
    print("days:", int(len(df_day)))
    print("date range:", df_day["day"].min(), "to", df_day["day"].max())
    print("avg users per day:", float(df_day["n_users"].mean()))
    print("\n mean_steps ~ temp")
    print(m_temp.summary())
    print("\n mean_steps ~ precip")
    print(m_precip.summary())
    print("\n mean_steps ~ temp + precip")
    print(m_both.summary())

    plt.figure(figsize=(7,4))
    plt.scatter(df_day["temp"], df_day["mean_steps"], alpha=0.8)
    x = pd.Series(sorted(df_day["temp"]))
    plt.plot(x, m_temp.predict(pd.DataFrame({"temp": x})))
    plt.xlabel("Temperature (F)")
    plt.ylabel("Mean steps (across users)")
    plt.title("Daily mean steps vs temperature")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,4))
    plt.scatter(df_day["precip"], df_day["mean_steps"], alpha=0.8)
    x = pd.Series(sorted(df_day["precip"]))
    plt.plot(x, m_precip.predict(pd.DataFrame({"precip": x})))
    plt.xlabel("Precipitation")
    plt.ylabel("Mean steps (across users)")
    plt.title("Daily mean steps vs precipitation")
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize=(9,4))
    ax1.plot(df_day["day"], df_day["mean_steps"], label="Mean steps")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Mean steps (across users)")
    ax2 = ax1.twinx()
    ax2.plot(df_day["day"], df_day["temp"], label="Temperature (F)")
    ax2.set_ylabel("Temperature (F)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    plt.title("Daily mean steps and temperature over time")
    fig.tight_layout()
    plt.show()

    df_day2 = df_day.copy()
    df_day2["temp_bin"] = pd.qcut(df_day2["temp"], q=8, duplicates="drop")
    df_bin = df_day2.groupby("temp_bin", as_index=False).agg(
        mean_steps=("mean_steps", "mean"),
        mean_temp=("temp", "mean"),
        n_days=("day", "size")
    ).sort_values("mean_temp")

    plt.figure(figsize=(7,4))
    plt.plot(df_bin["mean_temp"], df_bin["mean_steps"], marker="o")
    plt.xlabel("Temperature (F) (bin mean)")
    plt.ylabel("Mean steps (bin mean)")
    plt.title("Binned view: steps vs temperature")
    plt.tight_layout()
    plt.show()

    return df_day