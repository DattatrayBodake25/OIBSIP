import streamlit as st
import joblib

# Load the CountVectorizer and the trained SVM model
vectorizer = joblib.load(r"C:\Users\bodak\OneDrive\Desktop\Oasis Infobyte\Task 4\Task 4 - CountVectorizer.pkl")
model = joblib.load(r"C:\Users\bodak\OneDrive\Desktop\Oasis Infobyte\Task 4\Task 4 - svm.pkl")

# Function to preprocess and predict whether an email is spam or not
def predict_spam(email_text):
    preprocessed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([preprocessed_text])

    # Convert sparse matrix to dense array
    vectorized_text = vectorized_text.toarray()

    prediction = model.predict(vectorized_text)
    return prediction[0]

# Function to preprocess the email text
def preprocess_text(text):
    # Add your preprocessing steps here (e.g., lowercase, remove punctuation, etc.)
    return text

# Streamlit web app
def main():
    st.title("Email Spam Detector")
    st.write("Enter the email text below to check if it's spam or ham (not spam).")

    # Input for user to enter email text
    email_text = st.text_area("Email Text", "")

    if st.button("Predict"):
        if email_text:
            # Call the predict_spam function to get the prediction
            prediction = predict_spam(email_text)
            if prediction == 1:
                st.error("Spam Detected!")
            else:
                st.success("Ham (Not Spam)")

if __name__ == '__main__':
    main()
