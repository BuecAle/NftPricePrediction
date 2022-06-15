import pandas as pd
import requests
import regex as re
import numpy as np
import yfinance as yf
import time
import os
import sys



# Retrieved form XXX
def atomic_data_collection(start_date, end_date):
    def api_request_atomic(limit, time_data, time_start, lines_to_save_data, data_folder):
        counter = 0
        df = pd.DataFrame({}, dtype=str)
        url = "https://wax.api.atomicassets.io/atomicmarket/v1/sales/"

        state = {'sold': '3'}
        page = 1

        time_start = int(time_start.timestamp() * 10 ** 3)
        time_data = int(time_data.timestamp() * 10 ** 3)
        while (time_data > time_start):
            # time_data = int(time_data.timestamp()*10**3)

            querystring = {"state": state["sold"],
                           "before": time_data,
                           "page": str(page),
                           "limit": str(limit)}

            response = requests.request("GET", url, params=querystring)

            # print(response.status_code)
            if response.status_code == 200:
                df_supp = pd.DataFrame(response.json()['data'], dtype=str)
                if len(df_supp) == 0:
                    break
                df = df.append(df_supp, ignore_index=True)

                time_data_supp = int(df_supp.created_at_time.min())

                if page > 1:
                    if (pd.to_datetime(time_data, unit='ms') - pd.to_datetime(time_data_supp, unit='ms')) > pd.Timedelta(
                            '3 hours'):
                        page = 1
                        time_data = time_data_supp
                    else:
                        page += 1
                else:
                    if time_data == time_data_supp:
                        page += 1
                    else:
                        time_data = time_data_supp

                counter += limit
                if counter % lines_to_save_data == 0:
                    print(len(df_supp), len(df), pd.to_datetime(time_data, unit='ms'),
                          pd.to_datetime(time_data_supp, unit='ms'), page)
                    supp = pd.to_datetime(time_start, unit='ms')
                    df.to_csv(data_folder + 'NFT_Atomic_' + str(supp.month) + '_' + str(supp.year) + '.csv', index=False)
            time.sleep(1)

        supp = pd.to_datetime(time_start, unit='ms')
        if len(df) > 0:
            df.to_csv(data_folder + 'NFT_Atomic_' + str(supp.month) + '_' + str(supp.year) + '.csv', index=False)
        else:
            print('No data in this month')




    try:
        dt_start_time = pd.to_datetime(start_date)
        dt_end_time = pd.to_datetime(end_date)
    except:
        sys.exit('Wrong time format')

    dt_time = [dt_start_time, dt_end_time]
    print('From: ', dt_time[0])
    print('To: ', dt_time[1])

    lines_to_save_data = 5000
    limit = 100
    for i in range(len(dt_time) - 1):
        data_folder = './csv/raw_data/'
        # if not os.path.exists(data_folder):
        #     os.mkdir(data_folder)
        #     print(data_folder)
        #     # os.system('mkdir '+data_folder)
        # data_folder += str(dt_time[-2 - i].month) + '_' + str(dt_time[-2 - i].year) + '/'
        # if not os.path.exists(data_folder):
        #     os.mkdir(data_folder)
        #     # os.system('mkdir '+data_folder)
        #     print(data_folder)
        api_request_atomic(limit, dt_time[-1 - i], dt_time[-2 - i], lines_to_save_data, data_folder)


# Get prices in USD
def get_price(df, regex, column_name, crypto_prices):
    attributes = []
    for row in df.iterrows():
        if re.findall("token_symbol': '(.+?)'", row[1]["price"])[0] == "WAX":
            price_found = False
            for i in range(len(crypto_prices['date'])):
                if crypto_prices['date'][i] == row[1]["date"]:
                    exchange_rate = crypto_prices['Open'][i]
                    attribute = float(re.findall(regex, row[1]["price"])[0]) * 0.00000001 * exchange_rate
                    attribute = round(attribute, 2)
                    attributes.append(attribute)
                    price_found = True
            if not price_found:
                attributes.append(np.NaN)
        else:
            attributes.append(np.NaN)
    df[column_name] = attributes
    return df


# Get attributes
def get_expression(df, regex, column_name):
    attributes = []
    for row in df.iterrows():
        if bool(re.search(regex, row[1]["assets"])):
            attribute = re.findall(regex, row[1]["assets"])[0]
            attributes.append(attribute)
        else:
            attributes.append(np.NaN)
    df[column_name] = attributes
    return df


# Get collection attributes
def get_collection_expression(df, regex, column_name):
    attributes = []
    for row in df.iterrows():
        if bool(re.search(regex, row[1]["collection"])):
            attribute = re.findall(regex, row[1]["collection"])[0]
            attributes.append(attribute)
        else:
            attributes.append(np.NaN)
    df[column_name] = attributes
    return df


# Get data attributes
def get_data_expression(df, regex, column_name):
    attributes = []
    for row in df.iterrows():
        if bool(re.search(regex, row[1]["assets"])):
            data_string = str(re.findall("(?<='data': )(.*)", row[1]["assets"]))
            if bool(re.search(regex, data_string)):
                attribute = re.findall(regex, data_string)[0]
                if column_name == 'media':
                    if not attribute.startswith("https://"):
                        attribute = 'https://ipfs.atomichub.io/ipfs/' + attribute
                attributes.append(attribute)
            else:
                attributes.append(np.NaN)
        else:
            attributes.append(np.NaN)

    df[column_name] = attributes
    return df


# Download WAX<->USD price rate from yahoo finance
def get_wax_exchangerate(start_date, end_date, currency='WAXP-USD'):
    wax = yf.download(currency, start_date, end_date)
    wax = pd.DataFrame(wax['Open'])
    wax['date'] = wax.index
    return wax


# Get selling date in correct format
def get_date(df, column_name):
    df[column_name] = pd.to_datetime(df["updated_at_time"], utc=True, unit="ms")
    df[column_name] = df[column_name].dt.date
    return df


# Get Images
def get_images(df):
    counter = 0
    for row in df.iloc.iterrows():
        print(counter)
        counter += 1
        image_url = row[1]["media"]
        asset_id = str(row[1]["asset_id"])
        price = str(row[1]["price_usd"])
        try:
            image_data = requests.get(image_url).content
            image_path = "images/_" + price + "_" + asset_id + "_.jpg"
            with open(image_path, "wb") as image:
                image.write(image_data)
        except requests.exceptions.ConnectionError as e:
            r = "No response"
