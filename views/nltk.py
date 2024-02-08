import nltk
nltk.download('vader_lexicon')
# nltk.download('punkt', quiet=True)
# nltk.download

# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
def predict_text(text):
    ss = sid.polarity_scores(text)
    for key in ss:
        ss[key] *= 100
    if ss['compound'] >= 5:
        ss['result'] = "POSITIVE"
    elif ss['compound'] <= -5:
        ss['result'] = "NEGATIVE"
    else:
        ss['result'] = "NEUTRAL"
    return ss
