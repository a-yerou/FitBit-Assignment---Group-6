import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scripts.Part4 as p4

import scripts.data_loaders as loaders


st.set_page_config(page_title="Fitbit Dashboard", layout="wide", initial_sidebar_state="expanded")

DB_PATH_DEFAULT = "data/fitbit_database.db"


def pick_db_path():
    if os.path.exists(DB_PATH_DEFAULT):
        return DB_PATH_DEFAULT

    alt = "../data/fitbit_database.db"
    if os.path.exists(alt):
        return alt

    return DB_PATH_DEFAULT


DB_PATH = pick_db_path()


@st.cache_data(ttl=3600)
def load_everything(db_path):
    df_daily = loaders.load_daily_activity(db_path)
    df_sleep_min = loaders.load_minute_sleep(db_path)

    df_steps_h = loaders.load_hourly_steps(db_path)
    df_cal_h = loaders.load_hourly_calories(db_path)
    df_int_h = loaders.load_hourly_intensity(db_path)
    df_hr = loaders.load_heart_rate(db_path)

    try:
        df_weight = loaders.load_weight_log(db_path)
    except Exception:
        df_weight = None

    df_person_day = loaders.build_person_day(df_daily, df_sleep_min, df_weight)
    df_person_hour = loaders.build_person_hour(df_steps_h, df_cal_h, df_int_h, df_hr)

    return df_daily, df_sleep_min, df_weight, df_person_day, df_person_hour


@st.cache_data(ttl=3600)
def get_ids(_df_person_day):
    return sorted(_df_person_day["Id"].dropna().astype("int64").unique().tolist())


@st.cache_data(ttl=3600)
def get_date_range(_df_person_day):
    dmin = pd.to_datetime(_df_person_day["day"]).min().date()
    dmax = pd.to_datetime(_df_person_day["day"]).max().date()
    return dmin, dmax


@st.cache_data(ttl=3600)
def get_overview_kpis(_df_person_day):
    df = _df_person_day.copy()
    return {
        "n_users": int(df["Id"].nunique()),
        "date_min": str(pd.to_datetime(df["day"]).min().date()),
        "date_max": str(pd.to_datetime(df["day"]).max().date()),
        "mean_steps": float(df["TotalSteps"].mean()),
        "mean_calories": float(df["Calories"].mean()),
        "sleep_coverage": float(df["sleep_minutes"].notna().mean()),
    }


@st.cache_data(ttl=3600)
def fit_sleep_models(_df_person_day, min_days=10):
    df = _df_person_day.dropna(subset=["sleep_minutes"]).copy()

    counts = df.groupby("Id").size()
    keep_ids = counts[counts >= min_days].index
    df = df[df["Id"].isin(keep_ids)].copy()

    m_active = smf.ols("sleep_minutes ~ active_minutes", data=df).fit()
    m_sedentary = smf.ols("sleep_minutes ~ SedentaryMinutes", data=df).fit()
    m_inter = smf.ols("sleep_minutes ~ active_minutes * is_weekend", data=df).fit()

    return df, m_active, m_sedentary, m_inter


def coef_table(model, label):
    params = model.params
    conf = model.conf_int()

    out = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "ci_low": conf[0].values,
        "ci_high": conf[1].values,
        "p_value": model.pvalues.values,
    })

    out.insert(0, "model", label)
    return out


try:
    df_daily, df_sleep_min, df_weight, df_person_day, df_person_hour = load_everything(DB_PATH)
except Exception as e:
    st.error(f"DB load failed at: {DB_PATH}\n\n{e}")
    st.stop()

ids = get_ids(df_person_day)
dmin, dmax = get_date_range(df_person_day)


st.sidebar.title("Controls")

page = st.sidebar.radio("Page", ["Overview", "Individual", "Sleep analysis", "Weather"])

if page in ["Individual", "Sleep analysis"]:
    chosen_id = st.sidebar.selectbox("Select participant Id", ids)

if page in ["Individual", "Overview", "Sleep analysis"]:
    date_range = st.sidebar.date_input(
        "Date range (daily data)",
        value=(dmin, dmax),
        min_value=dmin,
        max_value=dmax,
    )

    if not (isinstance(date_range, tuple) and len(date_range) == 2):
        st.sidebar.error("Pick a start and end date.")
        st.stop()

    start_date, end_date = date_range

if page == "Individual":
    st.sidebar.subheader("Time-of-day filter (hourly data)")
    hour_start = st.sidebar.slider("Hour start", 0, 23, 0)
    hour_end = st.sidebar.slider("Hour end", 1, 24, 24)

if page == "Weather":
    st.sidebar.subheader("Weather")
    weather_city = st.sidebar.text_input("City", value="Chicago")
    weather_api_key = st.sidebar.text_input("Visual Crossing API key", value="", type="password")


if page == "Overview":
    st.title("Fitbit study overview")

    k = get_overview_kpis(df_person_day)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participants", k["n_users"])
    c2.metric("Date range", f'{k["date_min"]} → {k["date_max"]}')
    c3.metric("Mean steps/day", f"{k['mean_steps']:.0f}")
    c4.metric("Sleep coverage", f"{100*k['sleep_coverage']:.1f}%")

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots()
        x = df_person_day["TotalSteps"].dropna()
        ax.hist(x, bins=30)
        ax.set_title("Distribution of daily steps (all users)")
        ax.set_xlabel("TotalSteps")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with right:
        df = df_person_day.copy()
        dt = pd.to_datetime(df["day"])
        df["weekday"] = dt.dt.day_name()

        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        avg = df.groupby("weekday", as_index=False)["TotalSteps"].mean()
        avg["weekday"] = pd.Categorical(avg["weekday"], categories=order, ordered=True)
        avg = avg.sort_values("weekday")

        fig, ax = plt.subplots()
        ax.bar(avg["weekday"].astype(str), avg["TotalSteps"])
        ax.set_title("Average steps by weekday")
        ax.set_xlabel("Weekday")
        ax.set_ylabel("Mean steps")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)


elif page == "Individual":
    df_day = loaders.filter_person_day(df_person_day, id=chosen_id, start=start_date, end=end_date)
    df_hour = loaders.filter_person_hour(df_person_hour, id=chosen_id, start=start_date, end=end_date, hour_start=hour_start, hour_end=hour_end)

    st.title(f"Individual view: Id {chosen_id}")

    s_day = loaders.summarize_person_day(df_person_day, chosen_id, start=start_date, end=end_date)
    s_hour = loaders.summarize_person_hour(df_person_hour, chosen_id, start=start_date, end=end_date, hour_start=hour_start, hour_end=hour_end)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days (activity)", s_day["days_activity"])
    c2.metric("Days (sleep)", s_day["days_sleep"])
    c3.metric("Avg steps/day", f"{s_day['steps_mean']:.0f}" if not np.isnan(s_day["steps_mean"]) else "NA")
    c4.metric("Avg sleep (min)", f"{s_day['sleep_mean']:.1f}" if not np.isnan(s_day["sleep_mean"]) else "NA")

    df = df_day.sort_values("day").copy()
    df["day_dt"] = pd.to_datetime(df["day"])

    fig, ax = plt.subplots()
    ax.plot(df["day_dt"], df["TotalSteps"])
    ax.set_title("Daily steps")
    ax.set_xlabel("Day")
    ax.set_ylabel("Steps")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(df["day_dt"], df["Calories"])
    ax.set_title("Daily calories")
    ax.set_xlabel("Day")
    ax.set_ylabel("Calories")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    if df["sleep_minutes"].notna().any():
        fig, ax = plt.subplots()
        ax.plot(df["day_dt"], df["sleep_minutes"])
        ax.set_title("Sleep minutes")
        ax.set_xlabel("Day")
        ax.set_ylabel("Sleep minutes")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.info("No sleep data for this participant in the selected date range.")

    st.subheader("Hourly view (date range + hour window)")
    st.write(f"Hour window: {hour_start}:00 → {hour_end}:00")

    dfh = df_hour.sort_values("hour_dt").copy()

    fig, ax = plt.subplots()
    ax.plot(dfh["hour_dt"], dfh["StepTotal"])
    ax.set_title("Hourly steps (filtered)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Steps")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    if dfh["TotalIntensity"].notna().any():
        fig, ax = plt.subplots()
        ax.plot(dfh["hour_dt"], dfh["TotalIntensity"])
        ax.set_title("Hourly total intensity (filtered)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("TotalIntensity")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    if "mean_heart_rate" in dfh.columns and dfh["mean_heart_rate"].notna().any():
        fig, ax = plt.subplots()
        ax.plot(dfh["hour_dt"], dfh["mean_heart_rate"])
        ax.set_title("Hourly mean heart rate (filtered)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Mean HR")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    with st.expander("Raw tables"):
        st.dataframe(df)
        st.dataframe(dfh)


elif page == "Sleep analysis":
    st.title("Sleep duration analysis")

    min_days = st.slider("Minimum sleep-days per user to include", 3, 20, 10)

    with st.spinner("Fitting models..."):
        df_reg, m_active, m_sedentary, m_inter = fit_sleep_models(df_person_day, min_days=min_days)

    c1, c2, c3 = st.columns(3)
    c1.metric("Eligible users", int(df_reg["Id"].nunique()))
    c2.metric("Eligible rows", int(len(df_reg)))
    c3.metric("Mean sleep (min)", f"{df_reg['sleep_minutes'].mean():.1f}")

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots()
        ax.scatter(df_reg["active_minutes"], df_reg["sleep_minutes"], alpha=0.5)
        x = pd.Series(np.linspace(df_reg["active_minutes"].min(), df_reg["active_minutes"].max(), 100))
        y = m_active.predict(pd.DataFrame({"active_minutes": x}))
        ax.plot(x, y)
        ax.set_title("sleep_minutes ~ active_minutes")
        ax.set_xlabel("Active minutes")
        ax.set_ylabel("Sleep minutes")
        st.pyplot(fig)

    with right:
        fig, ax = plt.subplots()
        ax.scatter(df_reg["SedentaryMinutes"], df_reg["sleep_minutes"], alpha=0.5)
        x = pd.Series(np.linspace(df_reg["SedentaryMinutes"].min(), df_reg["SedentaryMinutes"].max(), 100))
        y = m_sedentary.predict(pd.DataFrame({"SedentaryMinutes": x}))
        ax.plot(x, y)
        ax.set_title("sleep_minutes ~ SedentaryMinutes")
        ax.set_xlabel("Sedentary minutes")
        ax.set_ylabel("Sleep minutes")
        st.pyplot(fig)

    tbl = pd.concat([
        coef_table(m_active, "active_only"),
        coef_table(m_sedentary, "sedentary_only"),
        coef_table(m_inter, "active_x_weekend"),
    ], ignore_index=True)

    st.subheader("Key coefficients")
    st.dataframe(tbl)

    st.subheader("Residual Q-Q plot (sedentary model)")
    fig = sm.qqplot(m_sedentary.resid, line="45")
    plt.title("Q-Q plot: residuals")
    st.pyplot(fig)


elif page == "Weather":
    st.title("Weather vs activity")

    if not weather_api_key:
        st.warning("Enter your Visual Crossing API key in the sidebar.")
        st.stop()

    plt.close("all")
    df_day_weather = p4.weather_relation(df_daily, weather_api_key, city=weather_city)

    fig_nums = plt.get_fignums()
    if len(fig_nums) == 0:
        st.error("No figures were generated.")
    else:
        for n in fig_nums:
            fig = plt.figure(n)
            st.pyplot(fig)

    st.subheader("Merged daily weather table")
    st.dataframe(df_day_weather)