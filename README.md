# Fitbit Assignment — Group 6

This repository contains our full submission for the Fitbit assignment (Parts 1–5).

Parts 1–4: analysis scripts (data loading/cleaning, statistical analysis, plots)
Part 5: interactive Streamlit dashboard

------------------------------------------------------------
Repository structure (key files)
------------------------------------------------------------
- dashboard.py
  Streamlit dashboard (Part 5)

- main.py
  Script-style runner used during development/testing of Parts 1–4

- scripts/
  Implementation for Parts 1–4
  - data_loaders.py
    SQLite loading + cleaning + merged datasets (daily/hourly)
  - Part1.py, Part3.py, Part4.py
    Analysis functions and plots

- data/fitbit_database.db
  SQLite database 

------------------------------------------------------------
Run the dashboard (Part 5)
------------------------------------------------------------
streamlit run dashboard.py
or see the link in the submitted file on canvas

------------------------------------------------------------
Run scripts (Parts 1–4)
------------------------------------------------------------
python main.py

------------------------------------------------------------
Install dependencies
------------------------------------------------------------
pip install -r requirements.txt

------------------------------------------------------------
Notes
------------------------------------------------------------
- The Weather page uses Visual Crossing; enter an API key in the sidebar when using the Weather page.
- Do not run the dashboard with: python dashboard.py
  Use: streamlit run dashboard.py or visit the link in the submitted file in canvas
