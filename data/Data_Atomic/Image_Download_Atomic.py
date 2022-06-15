import requests
import parameter
import Functions_Atomic
import pandas as pd


df = pd.read_csv(parameter.Image_Download_Atomic.file)

Functions_Atomic.get_images(df)
