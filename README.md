# FitBit-Assignment---Group-6

This repo contains the full submission for the Fitbit project. Parts $1$–$4$ are implemented as analysis scripts (data loading, cleaning, statistical analysis, and plots). Part $5$ is an interactive Streamlit dashboard.

To run the dashboard (Part $5$), execute:
`streamlit run dashboard.py`

To run the script workflow (Parts $1$ – $4$), execute:
`python main.py`

Dependencies can be installed via:
`pip install -r requirements.txt`
or, if no requirements file is present:
`pip install pandas numpy matplotlib statsmodels streamlit requests`

The SQLite database used for Parts $3$–$5$ must be located at:
`data/fitbit_database.db`

$$
\textbf{Notes:}\quad \text{The Weather page uses Visual Crossing; provide an API key in the sidebar.}\quad
\text{Do not run the dashboard with } \texttt{python dashboard.py}\text{; use } \texttt{streamlit run}\text{.}
$$
