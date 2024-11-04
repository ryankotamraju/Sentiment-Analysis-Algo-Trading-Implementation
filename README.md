This project is an implementation of a machine learning-based sentiment analysis model developed for automated stock trading. The model analyzes news headlines, assigns a sentiment score (from 1 to 5) based on the headline’s favorability, and generates a corresponding buy, sell, or hold signal. This project includes a training module that accepts labeled data to improve accuracy over time.

# Key Features
- Sentiment Scoring: Classifies news headlines into a sentiment score between 1 (very negative) and 5 (very positive).
- Trading Signal Generation: Produces a buy, sell, or hold signal based on the sentiment score.
- Training Module: Allows for the input of labeled training data to improve model performance.
- Modular Design: Easily adjustable sentiment thresholds for different market conditions.

# Methodology
- Data Collection: Requires a labeled dataset of financial news headlines and associated sentiment scores for model training.
- Preprocessing: Tokenizes, cleans, and processes the text data for feature extraction.
- Model Training: Trains on labeled data to map words and phrases to sentiment scores.
- Sentiment Scoring: Assigns a score from 1 to 5 to new headlines based on predicted sentiment.
- Signal Generation:
  - Buy: If the sentiment score is 4 or 5 (indicating favorable news).
  - Sell: If the sentiment score is 1 or 2 (indicating unfavorable news).
  - Hold: If the sentiment score is 3 (neutral news).

# Prerequisites
- Python 3.x
- Required libraries: pandas, scikit-learn, torch, and transformers.

Usage
- Training the Model: Edit the train_texts and train_labels variables to input labeled data and train the sentiment analysis model.
- Evaluating Headlines: Run main.py with new headlines to classify sentiment and generate trading signals.

License
This project is licensed under the MIT License. See LICENSE for details.

Acknowledgments
Special thanks to the *UChicago FINM Quantitative Portfolio Management and Algorithmic Trading* faculty for their support and insights, and to the developers of NLP tools and libraries that made this project possible.
