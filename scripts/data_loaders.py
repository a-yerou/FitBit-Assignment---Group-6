import sqlite3
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import scripts.Part3 as p3

DB_PATH = "../data/fitbit_database.db"

def load_daily_activity(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT Id, ActivityDate, TotalSteps, Calories, TotalDistance,
                          VeryActiveMinutes, FairlyActiveMinutes, LightlyActiveMinutes, SedentaryMinutes
                          FROM daily_activity""")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)

    df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("int64")
    df["dt"] = pd.to_datetime(df["ActivityDate"], errors="coerce")
    df["day"] = df["dt"].dt.date
    df["active_minutes"] = df["VeryActiveMinutes"] + df["FairlyActiveMinutes"] + df["LightlyActiveMinutes"]
    df["is_weekend"] = df["dt"].dt.weekday >= 5
    return df

def load_minute_sleep(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT Id, date, value, logId FROM minute_sleep")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)

    df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("int64")
    df["logId"] = pd.to_numeric(df["logId"], errors="coerce")
    df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["dt"])
    df["day"] = df["dt"].dt.date
    df["sleep_day"] = (df["dt"] - pd.Timedelta(hours=12)).dt.date
    return df

def load_hourly_steps(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT Id, ActivityHour, StepTotal FROM hourly_steps")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)

    df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("int64")
    df["dt"] = pd.to_datetime(df["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["dt"])
    df["hour_dt"] = df["dt"].dt.floor("h")
    df["day"] = df["hour_dt"].dt.date
    df["hour"] = df["hour_dt"].dt.hour
    return df

def load_hourly_calories(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT Id, ActivityHour, Calories FROM hourly_calories")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)

    df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("int64")
    df["dt"] = pd.to_datetime(df["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["dt"])
    df["hour_dt"] = df["dt"].dt.floor("h")
    df["day"] = df["hour_dt"].dt.date
    df["hour"] = df["hour_dt"].dt.hour
    return df

def load_hourly_intensity(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT Id, ActivityHour, TotalIntensity, AverageIntensity FROM hourly_intensity")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)

    df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("int64")
    df["dt"] = pd.to_datetime(df["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["dt"])
    df["hour_dt"] = df["dt"].dt.floor("h")
    df["day"] = df["hour_dt"].dt.date
    df["hour"] = df["hour_dt"].dt.hour
    return df

def load_heart_rate(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT Id, Time, Value FROM heart_rate")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)

    df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("int64")
    df["dt"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["dt"])
    df["hour_dt"] = df["dt"].dt.floor("h")
    df["day"] = df["hour_dt"].dt.date
    df["hour"] = df["hour_dt"].dt.hour
    df = df.rename(columns={"Value": "heart_rate"})
    return df

def load_weight_log(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT Id, Date, WeightKg, WeightPounds, Fat, BMI, IsManualReport, LogId FROM weight_log")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(rows, columns=cols)

    df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("int64")
    df["LogId"] = pd.to_numeric(df["LogId"], errors="coerce").astype("int64")
    df["dt"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["dt"])
    df["day"] = df["dt"].dt.date

    mask = df["WeightKg"].isna() & df["WeightPounds"].notna()
    df.loc[mask, "WeightKg"] = df.loc[mask, "WeightPounds"] / 2.2046226218

    df["flag_bad_weight"] = False
    df.loc[(df["WeightKg"].notna()) & ((df["WeightKg"] < 30) | (df["WeightKg"] > 250)), "flag_bad_weight"] = True

    df["flag_bad_bmi"] = False
    df.loc[(df["BMI"].notna()) & ((df["BMI"] < 10) | (df["BMI"] > 60)), "flag_bad_bmi"] = True

    return df


def make_sleep_daily(df_minute_sleep):
    df = df_minute_sleep.copy()
    df_sleep = df.groupby(["Id", "sleep_day"], as_index=False)["value"].size().rename(
        columns={"sleep_day": "day", "size": "sleep_minutes"}
    )
    return df_sleep

def build_person_day(df_daily_activity, df_minute_sleep, df_weight_log=None):
    df_sleep = make_sleep_daily(df_minute_sleep)
    df = df_daily_activity.merge(df_sleep, on=["Id", "day"], how="left")
    if df_weight_log is not None:
        df = df.merge(df_weight_log[["Id", "day", "WeightKg", "BMI", "flag_bad_weight", "flag_bad_bmi"]], on=["Id", "day"], how="left")
    return df

def coverage_report(person_day):
    df = person_day.copy()
    rep = df.groupby("Id", as_index=False).agg(
        days_activity=("day", "count"),
        days_sleep=("sleep_minutes", lambda x: x.notna().sum()),
        days_weight=("WeightKg", lambda x: x.notna().sum())
    )
    rep["sleep_coverage"] = rep["days_sleep"] / rep["days_activity"]
    rep["weight_coverage"] = rep["days_weight"] / rep["days_activity"]
    rep = rep.sort_values("sleep_coverage")
    print(rep)
    return rep

def sleep_eligible(person_day, min_days=10):
    df = person_day.dropna(subset=["sleep_minutes"]).copy()
    counts = df.groupby("Id").size()
    keep = counts[counts >= min_days].index
    df = df[df["Id"].isin(keep)]
    print("\n Coverage info for sleep analysis based on enough data being present:")
    print("eligible users:", len(keep))
    print("eligible rows:", len(df))
    return df

def filter_person_day(person_day, id=None, start=None, end=None):
    df = person_day.copy()
    if id is not None:
        df = df[df["Id"] == int(id)]
    d = pd.to_datetime(df["day"])
    if start:
        df = df[d >= pd.to_datetime(start)]
        d = pd.to_datetime(df["day"])
    if end:
        df = df[d <= pd.to_datetime(end)]
    return df

def summarize_person_day(person_day, id, start=None, end=None):
    df = filter_person_day(person_day, id=id, start=start, end=end)

    out = {}
    out["Id"] = int(id)
    out["days_activity"] = int(len(df))
    out["days_sleep"] = int(df["sleep_minutes"].notna().sum())
    out["days_weight"] = int(df["WeightKg"].notna().sum()) if "WeightKg" in df.columns else 0

    out["steps_mean"] = float(df["TotalSteps"].mean()) if len(df) else float("nan")
    out["active_mean"] = float(df["active_minutes"].mean()) if len(df) else float("nan")
    out["calories_mean"] = float(df["Calories"].mean()) if len(df) else float("nan")
    out["sedentary_mean"] = float(df["SedentaryMinutes"].mean()) if len(df) else float("nan")
    out["sleep_mean"] = float(df["sleep_minutes"].mean()) if out["days_sleep"] else float("nan")

    return out

def build_person_hour(df_steps, df_calories, df_intensity, df_hr=None):
    df = df_steps.merge(df_calories[["Id", "hour_dt", "Calories"]], on=["Id", "hour_dt"], how="left")
    df = df.merge(df_intensity[["Id", "hour_dt", "TotalIntensity", "AverageIntensity"]], on=["Id", "hour_dt"], how="left")
    if df_hr is not None:
        hr = df_hr.groupby(["Id", "hour_dt"], as_index=False)["heart_rate"].mean().rename(columns={"heart_rate": "mean_heart_rate"})
        df = df.merge(hr, on=["Id", "hour_dt"], how="left")
    df["day"] = df["hour_dt"].dt.date
    df["hour"] = df["hour_dt"].dt.hour
    return df

def filter_person_hour(person_hour, id=None, start=None, end=None, hour_start=None, hour_end=None):
    df = person_hour.copy()

    if id is not None:
        df = df[df["Id"] == int(id)]

    d = pd.to_datetime(df["day"])
    if start:
        df = df[d >= pd.to_datetime(start)]
        d = pd.to_datetime(df["day"])
    if end:
        df = df[d <= pd.to_datetime(end)]

    if hour_start is not None or hour_end is not None:
        hs = 0 if hour_start is None else int(hour_start)
        he = 24 if hour_end is None else int(hour_end)
        if hs < he:
            df = df[(df["hour"] >= hs) & (df["hour"] < he)]
        else:
            df = df[(df["hour"] >= hs) | (df["hour"] < he)]

    return df

def summarize_person_hour(person_hour, id=None, start=None, end=None, hour_start=None, hour_end=None):
    df = filter_person_hour(person_hour, id=id, start=start, end=end, hour_start=hour_start, hour_end=hour_end)

    out = {}
    out["Id"] = int(id) if id is not None else None
    out["rows"] = int(len(df))
    out["unique_days"] = int(pd.Series(df["day"]).nunique()) if len(df) else 0

    out["steps_mean_hour"] = float(df["StepTotal"].mean()) if len(df) else float("nan")
    out["steps_sum"] = float(df["StepTotal"].sum()) if len(df) else float("nan")
    out["calories_mean_hour"] = float(df["Calories"].mean()) if len(df) else float("nan")
    out["intensity_mean_hour"] = float(df["TotalIntensity"].mean()) if len(df) else float("nan")

    if "mean_heart_rate" in df.columns:
        out["hr_mean"] = float(df["mean_heart_rate"].mean()) if df["mean_heart_rate"].notna().any() else float("nan")

    return out

def plot_person_day(person_day, id, start=None, end=None):
    df = filter_person_day(person_day, id=id, start=start, end=end).sort_values("day")

    plt.figure(figsize=(9,4))
    plt.plot(pd.to_datetime(df["day"]), df["TotalSteps"])
    plt.title(f"Daily steps (Id={id})")
    plt.xlabel("Day")
    plt.ylabel("Steps")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    if df["sleep_minutes"].notna().any():
        plt.figure(figsize=(9,4))
        plt.plot(pd.to_datetime(df["day"]), df["sleep_minutes"])
        plt.title(f"Sleep minutes (Id={id})")
        plt.xlabel("Day")
        plt.ylabel("Sleep minutes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        d2 = df.dropna(subset=["sleep_minutes"])
        plt.figure(figsize=(6,4))
        plt.scatter(d2["active_minutes"], d2["sleep_minutes"])
        plt.title(f"Sleep vs activity (Id={id})")
        plt.xlabel("Active minutes")
        plt.ylabel("Sleep minutes")
        plt.tight_layout()
        plt.show()

def plot_person_hour(person_hour, id=None, start=None, end=None, hour_start=None, hour_end=None):
    df = filter_person_hour(person_hour, id=id, start=start, end=end, hour_start=hour_start, hour_end=hour_end).sort_values("hour_dt")

    plt.figure(figsize=(10,4))
    plt.plot(df["hour_dt"], df["StepTotal"])
    plt.title(f"Hourly steps (Id={id})")
    plt.xlabel("Hour")
    plt.ylabel("Steps")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    if df["TotalIntensity"].notna().any():
        plt.figure(figsize=(10,4))
        plt.plot(df["hour_dt"], df["TotalIntensity"])
        plt.title(f"Hourly intensity (Id={id})")
        plt.xlabel("Hour")
        plt.ylabel("TotalIntensity")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    if "mean_heart_rate" in df.columns and df["mean_heart_rate"].notna().any():
        plt.figure(figsize=(10,4))
        plt.plot(df["hour_dt"], df["mean_heart_rate"])
        plt.title(f"Hourly mean heart rate (Id={id})")
        plt.xlabel("Hour")
        plt.ylabel("Mean HR")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return df