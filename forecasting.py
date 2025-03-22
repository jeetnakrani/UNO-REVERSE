
import pandas as pd
from prophet import Prophet

def forecast_missed_tasks(df):
    df_forecast = df.copy()
    df_forecast['DATE'] = pd.to_datetime(df_forecast['DATE'])
    daily_misses = df_forecast.groupby('DATE')['was_missed'].sum().reset_index()
    daily_misses.rename(columns={'DATE':'ds', 'was_missed':'y'}, inplace=True)

    prophet_model = Prophet()
    prophet_model.fit(daily_misses)

    future_dates = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future_dates)

    return prophet_model, forecast
