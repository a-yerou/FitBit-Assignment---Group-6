import sqlite3
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests


DB_PATH = "../data/fitbit_database.db"
def inspect_table(table_name, DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]  # column names from the cursor
        df = pd.DataFrame(rows, columns=cols)
        print(df)


#inspect_table("daily_activity", DB_PATH)
#inspect_table("minute_sleep", DB_PATH)
#inspect_table("heart_rate", DB_PATH)


def sleep_duration(id, DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT logId, COUNT(*) AS sleep_minutes  from minute_sleep WHERE Id = ? GROUP BY logId", (id,))
        result = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(result, columns = cols)
        df["logId"] = df["logId"].astype("int64")
        print(df[["sleep_minutes", "logId"]])



def get_sleep_minutes(id, DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if id == None:
            cursor.execute("SELECT date, value, Id  from minute_sleep")
        else:
            cursor.execute("SELECT date, value  from minute_sleep WHERE Id = ?", (id,))
        result = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(result, columns=cols)
        return df


#print(get_sleep_minutes(8378563200, DB_PATH))



def regress_activity_on_sleep(id, DB_PATH):
    minute_sleep = get_sleep_minutes(id, DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT (VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes) AS active_minutes, ActivityDate from daily_activity WHERE id = ?", (id,))
        result = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(result, columns=cols)
        df["ActivityDate"] = pd.to_datetime(df["ActivityDate"]).dt.date
        df = df.rename(columns={"ActivityDate" : "day"})
        minute_sleep["day"] = pd.to_datetime(minute_sleep["date"]).dt.date
        ts = pd.to_datetime(minute_sleep["date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        minute_sleep["sleep_day"] = (ts - pd.Timedelta(hours=12)).dt.date
        df_sleep = minute_sleep.groupby("sleep_day", as_index=False)["value"].size().rename(columns={"sleep_day": "day", "size" : "sleep_minutes"})
        df_reg = df.merge(df_sleep, on="day", how="left")
        df_reg = df_reg.dropna(subset=["sleep_minutes"])
        print(df_reg)
        model = smf.ols("sleep_minutes ~ active_minutes", data=df_reg).fit()
        print(model.summary())
        plt.scatter(df_reg["active_minutes"], df_reg["sleep_minutes"])
        x_line = pd.Series(sorted(df_reg["active_minutes"]))
        y_line = model.predict(pd.DataFrame({"active_minutes": x_line}))
        plt.plot(x_line, y_line)
        plt.xlabel("Activity in Minutes")
        plt.ylabel("Sleep in Minutes")
        plt.title("Regression of Sleep on Activity")
        plt.tight_layout()
        plt.show()
        return model



def analyse_sed_sleep(DB_PATH):
    minute_sleep = get_sleep_minutes(None, DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT SedentaryMinutes AS sedentary_minutes, ActivityDate, Id from daily_activity")
        result = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df = pd.DataFrame(result, columns=cols)
        df["ActivityDate"] = pd.to_datetime(df["ActivityDate"]).dt.date
        df = df.rename(columns={"ActivityDate": "day"})
        ts = pd.to_datetime(minute_sleep["date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        minute_sleep["sleep_day"] = (ts - pd.Timedelta(hours=12)).dt.date
        df_sleep = minute_sleep.groupby(["Id", "sleep_day"], as_index=False)["value"].size().rename(
            columns={"sleep_day": "day", "size": "sleep_minutes"})
        df_reg = df.merge(df_sleep, on=["Id", "day"], how="left")
        df_reg = df_reg.dropna(subset=["sleep_minutes"])
        print(df_reg)
        model = smf.ols("sleep_minutes ~ sedentary_minutes", data=df_reg).fit()
        print(model.summary())
        plt.scatter(df_reg["sedentary_minutes"], df_reg["sleep_minutes"])
        x_line = pd.Series(sorted(df_reg["sedentary_minutes"]))
        y_line = model.predict(pd.DataFrame({"sedentary_minutes": x_line}))
        plt.plot(x_line, y_line)
        plt.xlabel("Sedentary Minutes")
        plt.ylabel("Sleep in Minutes")
        plt.title("Regression of Sleep on Sedentary Minutes")
        plt.tight_layout()
        plt.show()
        sm.qqplot(model.resid, line="45")
        plt.title("Q-Q plot of residuals")
        plt.tight_layout()
        plt.show()
        return model



def barplot_steps(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM hourly_steps")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df_steps = pd.DataFrame(rows, columns=cols)
        df_steps["dt"] = pd.to_datetime(df_steps["ActivityHour"], errors="coerce")
        df_steps["block"] = (df_steps["dt"].dt.hour // 4).astype("int")
        df_steps["day"] = df_steps["dt"].dt.date
        total_steps_per_participant_per_block = df_steps.dropna(subset=["dt"]).groupby(["Id", "day", "block"], as_index=False)["StepTotal"].sum().rename(columns = {"StepTotal" : "steps_4h"})
        avg_block = total_steps_per_participant_per_block.groupby(["block"], as_index=False)["steps_4h"].mean().rename(columns = {"steps_4h" : "avg_steps"})
        avg_block["block_label"] = avg_block["block"].map({0: "0–4", 1: "4–8", 2: "8–12", 3: "12–16", 4: "16–20", 5: "20–24"})
        plt.figure(figsize=(8, 4))
        plt.bar(avg_block["block_label"], avg_block["avg_steps"])
        plt.xlabel("Hour Blocks")
        plt.ylabel("Average Steps per Block")
        plt.title("Average steps per 4-hours")
        plt.show()



def barplot_calories(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM hourly_calories")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df_cal = pd.DataFrame(rows, columns=cols)
        df_cal["dt"] = pd.to_datetime(df_cal["ActivityHour"], errors="coerce")
        df_cal["block"] = (df_cal["dt"].dt.hour // 4).astype("int")
        df_cal["day"] = df_cal["dt"].dt.date
        total_steps_per_participant_per_block = df_cal.dropna(subset=["dt"]).groupby(["Id", "day", "block"], as_index=False)["Calories"].sum().rename(columns = {"Calories" : "calorie_4h"})
        avg_block = total_steps_per_participant_per_block.groupby(["block"], as_index=False)["calorie_4h"].mean().rename(columns = {"calorie_4h" : "avg_calories"})
        avg_block["block_label"] = avg_block["block"].map({0: "0–4", 1: "4–8", 2: "8–12", 3: "12–16", 4: "16–20", 5: "20–24"})
        plt.figure(figsize=(8, 4))
        plt.bar(avg_block["block_label"], avg_block["avg_calories"])
        plt.xlabel("Hour Blocks")
        plt.ylabel("Average Calories per Block")
        plt.title("Average Calories burnt per 4-hours")
        plt.show()



def barplot_minutes_sleep(DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM minute_sleep")
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        df_sleep = pd.DataFrame(rows, columns=cols)
        df_sleep["dt"] = pd.to_datetime(df_sleep["date"], errors="coerce")
        df_sleep["block"] = (df_sleep["dt"].dt.hour // 4).astype("int")
        df_sleep["day"] = df_sleep["dt"].dt.date
        total_steps_per_participant_per_block = df_sleep.dropna(subset=["dt"]).groupby(["Id", "day", "block"], as_index=False).size().rename(columns = {"size" : "sleep_4h"})
        avg_block = total_steps_per_participant_per_block.groupby(["block"], as_index=False)["sleep_4h"].mean().rename(columns = {"sleep_4h" : "avg_sleep"})
        avg_block["block_label"] = avg_block["block"].map({0: "0–4", 1: "4–8", 2: "8–12", 3: "12–16", 4: "16–20", 5: "20–24"})
        plt.figure(figsize=(8, 4))
        plt.bar(avg_block["block_label"], avg_block["avg_sleep"])
        plt.xlabel("Hour Blocks")
        plt.ylabel("Average Minutes of Sleep per Block")
        plt.title("Average Minutes of Sleep per 4-hours")
        plt.show()



def plot_hr_and_intensity(id, DB_PATH):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT Time, Value AS heart_rate FROM heart_rate WHERE Id = ?", (id,))
        hr_rows = cursor.fetchall()
        hr_cols = [d[0] for d in cursor.description]
        df_hr = pd.DataFrame(hr_rows, columns=hr_cols)
        print(df_hr["Time"].head(10))
        print(type(df_hr["Time"].iloc[0]) if len(df_hr) else None)
        cursor.execute("SELECT ActivityHour, TotalIntensity FROM hourly_intensity WHERE Id = ?", (id,))
        int_rows = cursor.fetchall()
        int_cols = [d[0] for d in cursor.description]
        df_int = pd.DataFrame(int_rows, columns=int_cols)
    df_hr["Time"] = pd.to_datetime(df_hr["Time"], errors="coerce")
    df_int["ActivityHour"] = pd.to_datetime(df_int["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df_hr = df_hr.dropna(subset=["Time"])
    df_int = df_int.dropna(subset=["ActivityHour"])
    df_hr["ActivityHour"] = df_hr["Time"].dt.floor("h")
    df_int["ActivityHour"] = df_int["ActivityHour"].dt.floor("h")
    df_hr_hourly = df_hr.groupby("ActivityHour", as_index=False)["heart_rate"].mean().rename(columns={"heart_rate": "mean_heart_rate"})
    df_plot = df_hr_hourly.merge(df_int, on="ActivityHour", how="left")
    print(df_hr.shape, df_int.shape, df_plot.shape)
    print(df_plot.head())
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df_plot["ActivityHour"], df_plot["mean_heart_rate"])
    ax1.set_xlabel("Time (hour)")
    ax1.set_ylabel("Heart rate (mean bpm)")
    ax2 = ax1.twinx()
    ax2.plot(df_plot["ActivityHour"], df_plot["TotalIntensity"])
    ax2.set_ylabel("Total intensity (per hour)")
    plt.title(f"Heart rate and total intensity for Id={id}")
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.show()
    return df_plot



def weather_vs_activity(API_KEY):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(ActivityDate) AS dmin, MAX(ActivityDate) AS dmax FROM daily_activity")
        dr_rows = cursor.fetchall()
        dr_cols = [d[0] for d in cursor.description]
        df_range = pd.DataFrame(dr_rows, columns=dr_cols)
        cursor.execute("SELECT Id, ActivityDate, TotalSteps, (VeryActiveMinutes+FairlyActiveMinutes+LightlyActiveMinutes) AS active_minutes FROM daily_activity")
        a_rows = cursor.fetchall()
        a_cols = [d[0] for d in cursor.description]
        df_a = pd.DataFrame(a_rows, columns=a_cols)
    dmin = pd.to_datetime(df_range.loc[0, "dmin"], errors="coerce").date()
    dmax = pd.to_datetime(df_range.loc[0, "dmax"], errors="coerce").date()
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Chicago/{dmin}/{dmax}"
    params = {"unitGroup":"us","include":"days","key":API_KEY,"contentType":"json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df_w = pd.DataFrame(data.get("days", []))
    df_w["day"] = pd.to_datetime(df_w["datetime"], errors="coerce").dt.date
    df_w = df_w[["day", "temp", "precip"]].dropna(subset=["day"])
    df_a["day"] = pd.to_datetime(df_a["ActivityDate"], errors="coerce").dt.date
    df = df_a.merge(df_w, on="day", how="inner").dropna(subset=["TotalSteps", "temp", "precip"])
    df_day = df.groupby("day", as_index=False).agg(
        TotalSteps=("TotalSteps","mean"),
        active_minutes=("active_minutes","mean"),
        temp=("temp","mean"),
        precip=("precip","mean"),
        n_participants=("Id","nunique"),
        n_rows=("Id","size")
    ).sort_values("day")
    m_temp = smf.ols("TotalSteps ~ temp", data=df_day).fit()
    m_precip = smf.ols("TotalSteps ~ precip", data=df_day).fit()
    plt.figure(figsize=(7,4))
    plt.scatter(df_day["temp"], df_day["TotalSteps"])
    x = pd.Series(sorted(df_day["temp"]))
    plt.plot(x, m_temp.predict(pd.DataFrame({"temp": x})))
    plt.xlabel("Temperature (F)")
    plt.ylabel("Mean steps (across participants)")
    plt.title("Daily mean steps vs temperature")
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(7,4))
    plt.scatter(df_day["precip"], df_day["TotalSteps"])
    x = pd.Series(sorted(df_day["precip"]))
    plt.plot(x, m_precip.predict(pd.DataFrame({"precip": x})))
    plt.xlabel("Precipitation")
    plt.ylabel("Mean steps (across participants)")
    plt.title("Daily mean steps vs precipitation")
    plt.tight_layout()
    plt.show()
    fig, ax1 = plt.subplots(figsize=(9,4))
    ax1.plot(df_day["day"], df_day["TotalSteps"], color="tab:blue", label="Mean steps")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Mean steps (across participants)")
    ax2 = ax1.twinx()
    ax2.plot(df_day["day"], df_day["temp"], color="tab:orange", label="Temperature (F)")
    ax2.set_ylabel("Temperature (F)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    plt.title("Daily mean steps and temperature over time")
    fig.tight_layout()
    plt.show()
    df_day2 = df_day.copy()
    df_day2["temp_bin"] = pd.qcut(df_day2["temp"], q=10, duplicates="drop")
    df_bin = df_day2.groupby("temp_bin", as_index=False).agg(mean_steps=("TotalSteps","mean"), mean_temp=("temp","mean"), n_days=("day","size")).sort_values("mean_temp")
    plt.figure(figsize=(7,4))
    plt.plot(df_bin["mean_temp"], df_bin["mean_steps"], marker="o")
    plt.xlabel("Temperature (F) (bin mean)")
    plt.ylabel("Mean steps (bin mean)")
    plt.title("Binned view: steps vs temperature")
    plt.tight_layout()
    plt.show()
    print(m_temp.summary())
    print(m_precip.summary())
    return df_day
