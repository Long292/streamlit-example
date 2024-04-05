# import torch
# import transformers
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# @st.cache(allow_output_mutation=True)

# def get_model():
#     tokenizer = AutoTokenizer.from_pretrained("long292/bartpho-syllable-base-applied-backtranslation")
#     model = AutoModelForSeq2SeqLM.from_pretrained("long292/bartpho-syllable-base-applied-backtranslation")
#     return tokenizer, model

# tokenizer, model = get_model()
# user_input = st.text_area('Nhap cau ban muon dich vao day nha')
# button = st.button("Dich nghia")
# text = st.text_area('Enter some text')

# if user_input and button:
#     input_ids = tokenizer([user_input], return_tensors="pt").input_ids
#     output =  model.generate(input_ids)
#     pred = tokenizer_7.decode(output_ids[0], skip_special_tokens=True)
#     st.write(pred)

import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("pnichite/YTFineTuneBert")
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
    
  1:'Toxic',
  0:'Non Toxic'
}

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])
