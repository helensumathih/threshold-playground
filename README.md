#Spam Threshold Playground

An interactive app that demonstrates the precision/recall tradeoff in binary classification

#What it does
Trains a logistic regression spam classifier (TF-IDF + SMS Spam Collection Dataset), then lets users drag a threshold slider to see, in real time, how the decision cutoff affects the precision, recall, and the confision matrix. This also includes a live text box to test your own messages against the current threshold.

##Why
Most explanations of precision/recall are static forumulas. This turns the concept into something you can manipulate and feel the trade of, directly.

##Stack
Python, scikit-learn, (TF-IDF + Logistic Regression), Streamlit

##Run locally
\'''
pip install -r requirements.txt
streamlit run app.py
\'''

##Live demo
