import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string


def transform_text(text):
    text = text.lower() 
    text = nltk.word_tokenize(text)  
    output = []
    
    
    for word in text:
        if word.isalnum():
            output.append(word)
    
    text = output[:]
    output.clear()

    
    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            output.append(word)
    
    text = output[:]
    output.clear()

    
    ps = PorterStemmer()
    for word in text:
        output.append(ps.stem(word))
    
    return " ".join(output)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("SMS SPAM CLASSIFIER 🚀")


st.sidebar.header("How to use this App:")
st.sidebar.write("""
    - Enter the SMS message in the input box below.
    - Click on **PREDICT** to classify the message as **Spam** or **Not Spam**.
    - The result will be displayed below.
""")


input_sms = st.text_area("Enter the Message:")


if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify!")
    else:
      
        transform_sms = transform_text(input_sms)

        
        vector_input = tfidf.transform([transform_sms])

       
        result = model.predict(vector_input)[0]

       
        if result == 1:
            st.header("🚨 Spam Message 🚨", anchor="spam")
            st.write("This message is classified as spam. Please be cautious!")
            st.markdown("""
                **Spam messages** are often unsolicited and can be harmful. 
                Avoid clicking on any links or responding.
            """)
        else:
            st.header("✅ Not Spam", anchor="not-spam")
            st.write("This message appears safe and not spam.")

        st.markdown("<br>", unsafe_allow_html=True)
