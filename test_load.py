import joblib

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

test_message = ["Congratulations! You have worn a free prize, claim now!"]
vec = vectorizer.transform(test_message)
prob = model.predict_proba(vec)[:, 1]

print(f"Spam probability: {prob[0]:.3f}")