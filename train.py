import pandas as pd
import re 
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# ==== Trying again because the model wasn't accurate ===

# STEP 1 :
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>","",text)   # removes HTML Tags
    text = re.sub(r"[^a-z\s]","",text) # keeps letters & spaces
    text = re.sub(r"\s+"," ",text).strip() # removes extra spaces
    return text


# ====== Loading the Dataset ========
print("====LOADING DATASET====")
df =  pd.read_csv(r"C:\Users\talha\IMDB-Sent_Analysis\Sentimental-Analysis Project\data\IMDB Dataset.csv")
# df.rename(columns={"review":"text"},inplace=True)
print(f"DATASET LOADED : {df.shape[0]} rows")

# This is Optional : If you want to test your model on a Smaller Sample
# df = df.sample(10000,random_state=42)

# STEP 2: Clean text column
print("Cleaning Text ...")
df["review"] = df["review"].apply(clean_text)


# STEP 3 : Train-Test Split
x_train,x_test,y_train,y_test = train_test_split(df["review"],df["sentiment"],test_size=0.2,random_state=42)
print(f"Training set: {x_train.shape[0]} rows")
print(f"Test set: {x_test.shape[0]} rows")

# Pipelining ===> Chain of steps into one object.
# ====> It automatically applies the vectorizer and classifier in order.

# Step 4 : Create Pipeline 
print('Building Pipeline ...')
model = Pipeline([
    ("tfidf",TfidfVectorizer(stop_words="english",max_df=0.7,min_df=5,ngram_range=(1,2))),
    ("clf",LogisticRegression(max_iter=1000))
])

# step 5 : Train the Model
model.fit(x_train,y_train)


# Step 6: Evaluating the Model
y_pred = model.predict(x_test)
print("Confusion Matrix : ",confusion_matrix(y_test,y_pred))
print("Classification Report : ",classification_report(y_test,y_pred))

# Now, Save the Model
import os
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/sentiment_model.pkl")
print("✅ Your model is tested and saved in 'models/sentiment_model.pkl' 🙂")
