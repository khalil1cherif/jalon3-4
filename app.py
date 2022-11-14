import pickle
import prediction
import streamlit as st

vectoriseur_pickle=open('./vectoriseur_file','rb')
vectoriseur=load(vectoriseur_pickle)

model_pickle=open('./model_file','rb')
model=load(model_pickle)


# Add a title and intro text
st.title('Topics extractor')
st.text('This is a web app that allows Topics extraction from bad restaurant reviews')




with st.form("my_form"):
   
   number = st.number_input("topics number", value=1, min_value=1, max_value=15)
   text=st.text_input("Your review", placeholder="At least 4 characters")
   # Every form must have a submit button.
   submitted = st.form_submit_button("Extract topics")
   if submitted:
      if len(text) < 5:
         st.error("Invalid text: At least 4 characters")
      else:
         pre=prediction.predict_topics(model, vectorizer,number,text)
         if not pre[1]:
            st.info("Polarity score: "+ str(pre[0]))
         else:
            st.info("Polarity score: "+ str(pre[0]))
            st.info("Topics: "+ str(pre[1]))
