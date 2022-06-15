import requests
import pandas as pd

df = pd.read_csv("/disk1/transfer-alex/Data_Atomic/csv/prepared_data/dataAtomic_complete.csv")


# Get attributes
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
            image_path = "/disk1/transfer-alex/Data_Atomic/images/_"+price + "_" + asset_id + "_.jpg"
            with open(image_path, "wb") as image:
                image.write(image_data)
        except requests.exceptions.ConnectionError as e:
            r = "No response"


get_images(df)
