# NFT Data collection of wax.atomichub.io
import datetime as dt
import pandas as pd
import Functions_Atomic

# Get Data from Atomic API
# API_Atomic.Atomic_Data_Collection(2022-03-01, 2022-03-30)

date = '12_2020'
start_date = dt.datetime(2020,1,1)
end_date = dt.datetime(2022,5,1)
directory = 'C:/Users/BuecAle/Desktop/Uni/Masterthesis/NFT_Database/Data_Atomic/'
filepath = directory + date + '/' + 'NFT_Atomic_' + date + '.csv'

# Filter only atomicmarket nfts
df = pd.read_csv(filepath)
df = df[df["market_contract"].str.contains("atomicmarket")]
df = df[["price", "assets", "seller", "updated_at_time"]]

# WAX <-> USD value form yahoo finance
wax = Functions_Atomic.get_wax_exchangerate(start_date, end_date)

# Get all necessary attributes
Functions_Atomic.get_date(df, "date")
Functions_Atomic.get_price(df, "amount': '(.+?)'", "price_usd", wax)
Functions_Atomic.get_expression(df, "asset_id': '(.+?)'", "asset_id")
Functions_Atomic.get_expression(df, "'author': '(.+?)'", "collection_author")
Functions_Atomic.get_expression(df, "collection_name': '(.+?)'", "collection_name")
Functions_Atomic.get_data_expression(df, "name': '(.+?)'", "name")
Functions_Atomic.get_data_expression(df, "img': '(.+?)'", "media")

df = df[["asset_id", "collection_author", "collection_name", "name", "date", "price_usd", "seller", "media"]]
df = df.dropna()
df = df.drop_duplicates(subset=["asset_id"])
df = df.drop_duplicates(subset=["name"])

file_to_save = "dataAtomic_" + date + ".csv"
df.to_csv("csv/" + file_to_save)
