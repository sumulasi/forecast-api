from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin
import json

import pandas as pd
import statsmodels.api as sm
from datetime import datetime
from dateutil.relativedelta import relativedelta

from statsmodels.tsa.stattools import adfuller
from pandas.tseries.offsets import DateOffset

app = Flask(__name__)
api = Api(app)

cors = CORS(app, resources={r"/sales": {"origins": "http://autoconcept-forecast.s3-website.eu-north-1.amazonaws.com/"},
                            r"/income": {"origins": "http://autoconcept-forecast.s3-website.eu-north-1.amazonaws.com/"}})
app.config['CORS_HEADERS'] = 'Content-Type'

def adfuller_test(income):
    result = adfuller(income)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print(
            "strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print(
            "weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


@app.route("/sales/<months>")
@cross_origin(origins='*')
def forecast_sales(months):
    shiftMonth = 47
    predict_start = 87
    dataset = pd.read_csv("./MonthlySales.csv")
    dataset['Month'] = pd.to_datetime((dataset['Month']))
    dataset.set_index('Month', inplace=True)

    dataset['future'] = dataset['Sales'] - dataset['Sales'].shift(shiftMonth)
    adfuller_test(dataset['future'].dropna())

    model = sm.tsa.statespace.SARIMAX(dataset['Sales'], order=(1, 1, 0), seasonal_order=(1, 1, 0, shiftMonth))
    results = model.fit()

    future_dates = [dataset.index[-1] + DateOffset(months=x) for x in range(0, 36)]

    future_datest_df = pd.DataFrame(index=future_dates[1:], columns=dataset.columns)

    future_df = pd.concat([dataset, future_datest_df])

    future_df['forecast'] = results.predict(start=predict_start, end=(predict_start + int(months)), dynamic=True)
    future_df[['Sales', 'forecast']].plot(figsize=(12, 8))

    date_format = '%d/%m/%Y'

    end_date = datetime.strptime("1/4/2023", date_format)
    future_date = end_date + relativedelta(months=int(months))

    return {
        "original": json.dumps(future_df['Sales'].to_list()),
        "forecast": json.dumps(future_df['forecast'].to_list()),
        "startMonth": "1/1/2016",
        "endMonth": future_date.strftime(date_format)
    }


@app.route("/income/<months>")
@cross_origin(origins='*')
def forecast_income(months):
    shift_month = 9
    predict_start = 87
    dataset = pd.read_csv("./MonthlyIncome.csv")
    dataset['Month'] = pd.to_datetime((dataset['Month']))
    dataset.set_index('Month', inplace=True)

    dataset['future'] = dataset['Income'] - dataset['Income'].shift(shift_month)
    adfuller_test(dataset['future'].dropna())

    model = sm.tsa.statespace.SARIMAX(dataset['Income'], order=(1, 1, 0), seasonal_order=(1, 1, 0, shift_month))
    results = model.fit()

    future_dates = [dataset.index[-1] + DateOffset(months=x) for x in range(0, 36)]

    future_datest_df = pd.DataFrame(index=future_dates[1:], columns=dataset.columns)

    future_df = pd.concat([dataset, future_datest_df])

    print((predict_start + int(months)))

    future_df['forecast'] = results.predict(start=predict_start, end=(predict_start + int(months)), dynamic=True)
    future_df[['Income', 'forecast']].plot(figsize=(12, 8))

    date_format = '%d/%m/%Y'

    end_date = datetime.strptime("1/4/2023", date_format)
    future_date = end_date + relativedelta(months=int(months))

    return {
        "original": json.dumps(future_df['Income'].to_list()),
        "forecast": json.dumps(future_df['forecast'].to_list()),
        "startMonth": "1/1/2016",
        "endMonth": future_date.strftime(date_format)
    }


if __name__ == "__main__":
    app.run()
