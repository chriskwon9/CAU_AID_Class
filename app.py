import streamlit as st
st.title("AID Streamlit Practice")
st.write("This is my new app")
button1 = st.button("Click Me")
if button1:
    st.write("This is some text.")


st.header("start of the radio button example")
animal = st.radio("Choose your favorite animal:", ('Cat', 'Dog', 'Bird'))
if animal == 'Cat':
    st.write("You selected Cat.")


# streamlit run app.py