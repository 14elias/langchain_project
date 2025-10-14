import langchain_helper as lch
import streamlit as st

st.title("Pet's name Generator")

user_animal_name = st.sidebar.selectbox(
    "Select an animal:",
    ["Cat", "Dog", "Elephant", "Lion", "Tiger", "Bear", "Wolf", "Giraffe", "Zebra", "Monkey"]
)

if user_animal_name :
    pet_color = st.sidebar.text_area(label="Input your pet's color", max_chars=15)

if pet_color:
    response = lch.generate_pet(user_animal_name, pet_color)
    st.write(response['text'])