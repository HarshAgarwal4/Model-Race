import asyncio
import joblib

# Load trained model
saved_model = joblib.load("models/sentiment_model.pkl")

vectorizer = saved_model["vectorizer"]
model = saved_model["model"]
labels = saved_model["labels"]


async def model_call(review: str):
    review_tfidf = vectorizer.transform([review])

    prediction = model.predict(review_tfidf)[0]

    return prediction


async def main():
    review = input("Enter your review: ")

    prediction = await model_call(review=review)

    print("\nPredicted Sentiment:", labels[prediction])


if __name__ == "__main__":
    asyncio.run(main())