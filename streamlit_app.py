import torch
import transformers
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# @st.cache_resource(allow_output_mutation=True)

def get_model():
    tokenizer = AutoTokenizer.from_pretrained("long292/bartpho-syllable-base-applied-backtranslation")
    model = AutoModelForSeq2SeqLM.from_pretrained("long292/bartpho-syllable-base-applied-backtranslation")
    return tokenizer, model

tokenizer, model = get_model()
user_input = st.text_area('Nhap cau ban muon dich vao day nha')
button = st.button("Dich nghia")

if user_input and button:
    input_ids = tokenizer([user_input], return_tensors="pt").input_ids
    output =  model.generate(input_ids)
    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write(pred)

