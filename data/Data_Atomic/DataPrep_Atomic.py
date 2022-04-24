# NFT Data collection of wax.atomichub.io

import pandas as pd
import API_Atomic

# Get Data from Atomic API
# API_Atomic.Atomic_Data_Collection(2022-03-01, 2022-03-30)

# Pfad auf Datum anpassen
df = pd.read_csv('./Data_Atomic/1_2022/NFT_Atomic_1_2022.csv')
df = df[df["market_contract"].str.contains("atomicmarket")]
df = df[["price","listing_price","assets"]]

# WAX <-> USD value form "https://coinranking.com/de/coin/mWg3P2dAashSL+wax-wax"
API_Atomic.get_price(df, "amount': '(.+?)'", "price_usd")
API_Atomic.get_expression(df, "asset_id': '(.+?)'", "asset_id")
API_Atomic.get_expression(df, "collection_name': '(.+?)'", "collection_name")
API_Atomic.get_data_expression(df, "name': '(.+?)'", "name")
API_Atomic.get_data_expression(df, "img': '(.+?)'", "media")

df = df[["asset_id","collection_name","name","price_usd","seller","media"]]
df = df.dropna()
df = df.drop_duplicates(subset=["asset_id"])
df = df.drop_duplicates(subset=["name"])


df.to_csv("dataAtomic_10_2021.csv")