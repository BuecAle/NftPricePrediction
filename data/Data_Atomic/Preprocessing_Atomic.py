# NFT Data collection of wax.atomichub.io
import datetime as dt
import pandas as pd
import Functions_Atomic, parameter

# Get Data from Atomic API
# API_Atomic.Atomic_Data_Collection(2022-03-01, 2022-03-30)

# dates have to be changed in parameter.py file
date = parameter.Preprocessing_Atomic.date
start_date = parameter.Preprocessing_Atomic.start_date
end_date = parameter.Preprocessing_Atomic.end_date
directory = './csv/raw_data/'
filepath = directory + '/NFT_Atomic_' + date + '.csv'

# Filter only atomicmarket nfts
df = pd.read_csv(filepath)
df = df[df["market_contract"].str.contains("atomicmarket")]
df = df[["price", "assets", "seller", "buyer", "collection", "collection_name", "updated_at_time"]]

# WAX <-> USD value form yahoo finance
wax = Functions_Atomic.get_wax_exchangerate(start_date, end_date)

# Get all necessary attributes
Functions_Atomic.get_date(df, "date")
print("Date column done")
Functions_Atomic.get_price(df, "amount': '(.+?)'", "price_usd", wax)
print("Price column done")
Functions_Atomic.get_expression(df, "asset_id': '(.+?)'", "asset_id")
print("Asset_id column done")
Functions_Atomic.get_expression(df, "'owner': '(.+?)'", "owner")
print("Owner column done")
Functions_Atomic.get_expression(df, "is_burnable': (.+?),", "burnable")
print("Burnable column done")
Functions_Atomic.get_data_expression(df, "name': '(.+?)'", "name")
print("Name column done")
Functions_Atomic.get_data_expression(df, "img': '(.+?)'", "media")
print("Media column done")
Functions_Atomic.get_collection_expression(df, "'author': '(.+?)'", "coll_author")
print("Collection author column done")
Functions_Atomic.get_collection_expression(df, "market_fee': (.+?),", "coll_market_fee")
print("Market fee column done")


df = df[["asset_id", "name", "owner", "seller", "buyer", "burnable", "date", "price_usd", "media", "collection_name",
         "coll_author", "coll_market_fee"]]
df.rename(columns={'collection_name':'coll_name'}, inplace=True)
df = df.dropna()
df = df.drop_duplicates(subset=["asset_id"])
df = df.drop_duplicates(subset=["name"])

file_to_save_data = "dataAtomic_" + date
df.to_csv("csv/prepared_data" + file_to_save_data + ".csv")

