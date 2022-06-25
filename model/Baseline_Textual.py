import math
import pandas as pd
import statistics
import parameter

# Specify directory
df = pd.read_csv(parameter.Baseline_Textual.file)


# Sums up all prices in list
def price_sum(df):
    price = 0
    for row in df.iterrows():
        price += row[1]['price_usd']
    total_price = price / len(df)
    return total_price


# Calculate average price per collection
def average_price_collection(df):
    collection_name = ''
    collections = []
    prices = []
    collection_found = False
    for row in df.iterrows():
        collection_name = row[1]["coll_name"]
        for collection in collections:
            if collection == row[1]["coll_name"]:
                collection_found = True
        if not collection_found:
            collections.append(collection_name)
            df_filtered = df.loc[df['coll_name'] == collection_name]
            total_price = price_sum(df_filtered)
            prices.append(total_price)
        collection_found = False
    df_avgprice = pd.DataFrame(
        {'coll_name': collections,
         'total_price': prices
        })
    return df_avgprice


# Calculates error metrics for mean, median or mode
def error_metrics(df, method, metrics):
    differences = 0
    price_list = list(df["price_usd"])
    if method == "mean":
        input_value = statistics.mean(price_list)
    elif method == "median":
        input_value = statistics.median(price_list)
    elif method == "mode":
        input_value = statistics.mode(price_list)
    if metrics == "mae":
        for price in price_list:
            difference = abs(price - input_value)
            differences += difference
        output_value = (differences / len(price_list))
    elif metrics == "rmse":
        for price in price_list:
            difference = (price - input_value)**2
            differences += difference
        output_value = math.sqrt(differences / len(price_list))
    elif metrics == "mape":
        for price in price_list:
            difference = abs((price - input_value)/price)
            differences += difference
        output_value = 100 * (differences /  len(price_list))
    return output_value



# Calculates error metrics for average price per collection
def error_metrics_coll(df, df_avgprice):
    differences_mae = 0
    differences_rmse = 0
    differences_mape = 0
    for row in df.iterrows():
        for coll in df_avgprice.iterrows():
            coll_name = row[1]["coll_name"]
            coll_name_coll = coll[1]["coll_name"]
            paid_price = row[1]["price_usd"]
            if coll_name == coll_name_coll:
                avg_price = coll[1]["total_price"]
                difference_mae = abs(paid_price - avg_price)
                differences_mae += difference_mae
                difference_rmse = (paid_price - avg_price)**2
                differences_rmse += difference_rmse
                difference_mape = abs(((paid_price - avg_price) / paid_price))
                differences_mape += difference_mape
    mae_value = (differences_mae / len(df))
    rmse_value = math.sqrt((differences_rmse / len(df)))
    mape_value = 100 * (differences_mape / len(df))
    print(mae_value)
    print(rmse_value)
    print(mape_value)


# df_avgprice = average_price_collection(df)
# error_metrics_coll(df, df_avgprice)
print(error_metrics(df, "mode", "mae"))
print(error_metrics(df, "mode", "rmse"))
print(error_metrics(df, "mode", "mape"))



