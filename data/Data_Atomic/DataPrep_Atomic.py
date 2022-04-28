# NFT Data collection of wax.atomichub.io

import pandas as pd
import API_Atomic

# Get Data from Atomic API
# API_Atomic.Atomic_Data_Collection(2022-03-01, 2022-03-30)

date = '2_2021'
directory = 'C:/Users/BuecAle/Desktop/Uni/Masterthesis/NFT_Database/Data_Atomic/'
filepath = directory + date + '/' + 'NFT_Atomic_' + date + '.csv'

# Filter only atomicmarket nfts
df = pd.read_csv(filepath)
df = df[df["market_contract"].str.contains("atomicmarket")]
df = df[["price", "assets", "seller"]]

# WAX <-> USD value form "https://coinranking.com/de/coin/mWg3P2dAashSL+wax-wax"
API_Atomic.get_price(df, "amount': '(.+?)'", "price_usd")
API_Atomic.get_expression(df, "asset_id': '(.+?)'", "asset_id")
API_Atomic.get_expression(df, "collection_name': '(.+?)'", "collection_name")
API_Atomic.get_data_expression(df, "name': '(.+?)'", "name")
API_Atomic.get_data_expression(df, "img': '(.+?)'", "media")

df = df[["asset_id", "collection_name", "name", "price_usd", "seller", "media"]]
df = df.dropna()
df = df.drop_duplicates(subset=["asset_id"])
df = df.drop_duplicates(subset=["name"])

file_to_save = "dataAtomic_" + date + ".csv"
df.to_csv("csv/" + file_to_save)
