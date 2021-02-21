# BITweets

Predicting Bitcoin prices using a multi-dimensional LSTM and sentiment analysis.

## Inspiration
The rise (and fall) and subsequent rise of Bitcoin has captivated aspiring traders across the world. The very fact that Bitcoin has seen such widespread mainstream exposure has made cryptocurrencies a hot topic of discussion. Recently, Elon Musk acquired $1.5 bn worth of Bitcoin, and confirmed that it would be accepted as payment in the future. There is thus much value in developing a model for algorithmic trading of these volatile cryptocurrencies. 

Our team aims to develop a predictive model for Bitcoin prices, in the hopes that it will serve as a starting point for other traders who want to develop a full-fledged trading strategy.

## What it does
Given the constantly growing interest in cryptocurrencies such as Bitcoin, there is much value in developing a model for algorithmic trading of these volatile cryptocurrencies. In view of time constraints, our team developed a predictive model for close BTC-USD prices to serve as a starting point for those who want to develop a full-fledged trading strategy. Besides time-series data, sentiment scores of Tweets relating to Bitcoin and the relative popularity of Bitcoin searches on Google Trends were included independently as features in our multi-dimensional Long Short Term Memory (LSTM) model. We found that there was close agreement between our predicted prices and the actual close prices for all 3 models.

## How we built it
In addition to the spot trading dataset provided by Kaiko, we scraped hourly and daily BTC-USD price data from Yahoo! Finance using the APIs available [1]. 

We also wanted to investigate whether Bitcoin sentiment (on Twitter) can better predict the close prices. Thus, we cleaned up data obtained from a Kaggle dataset [2]. The Kaggle user scraped all tweets mentioning ‘bitcoin’ from 1st August 2017 through 21st January 2019. The tweets with negative and positive sentiment were then assigned a score from -1 to 0 and from 0 to 1 respectively. The average of the negative sentiment scores was calculated for each hour, as was the average of the positive sentiment scores. We added up these averages to obtain an ‘overall’ sentiment for ‘bitcoin’ for each hour and each day. 

To further complement our project, we looked into the relative popularity of ‘bitcoin’ searches daily on Google Trends for the year 2020 [3]. 

## Challenges we ran into
Initially, we experimented with a single dimension LSTM model which uses the current close price to predict the close price for the following day. The model had one input layer, 4 LSTM neural layers, and one output layer.

To predict the closing price for the following day using multiple inputs, we used a multi-dimensional LSTM model. We first trained the model on open prices, close prices and adjusted close prices. Subsequently, we trained the model on open prices, close prices, adjusted close prices, and Twitter sentiment scores from the Kaggle dataset. Lastly, we trained the model on open prices, close prices, adjusted close prices, and the daily relative popularity of ‘bitcoin’ searches on Google Trends.

## Accomplishments that we're proud of
Machine learning using multi-dimensional LSTM model.

## What we learned
Machine learning using multi-dimensional LSTM model.

## What's next for BITweets
In the future, we aim to explore other models to analyse the dataset and predict future prices. We hope to further fine-tune the model based on other feature inputs and parameters that are relevant to Bitcoin. Currently, our model uses one previous timestep to predict the next. In our extension, we hope to predict the next timestep using the past N steps, where N can be fine-tuned as well.

## Results
![alt text](https://github.com/cwlroda/btc-predictor/blob/main/img/ohlcv.png)
![alt text](https://github.com/cwlroda/btc-predictor/blob/main/img/yf_2y.png)
![alt text](https://github.com/cwlroda/btc-predictor/blob/main/img/sentiment.png)
![alt text](https://github.com/cwlroda/btc-predictor/blob/main/img/sentiment_prediction.png)
![alt text](https://github.com/cwlroda/btc-predictor/blob/main/img/trend.png)
![alt text](https://github.com/cwlroda/btc-predictor/blob/main/img/trend_prediction.png)
