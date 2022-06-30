# NftPricePrediction
This project is used for a static price prediction of Non-fungible Tokens by the use of machine learningmodels.

## Models
The price prediction process has been separated into three different approaches:
- Price prediction with textual data
- Price prediction with image data
- Price prediction with textual and image data

For this different approaches, differnt models have been used. For the textual as well as the textual and image data price prediction a linear regression model,
a decision tree regressor and a random forest regressor are used. The used models are taken from the tensorflow.keras library. For the price prediction based on image
data alone a ResNet50 network is used. 

## Data
The collected textual data is stored in this repository, however, the image data including more than 60'000 images are stored on the servers of the University of
Liechtenstein. The code for the data collection is also provided in the repository. All the used NFT data has been taken from Atomichub.
