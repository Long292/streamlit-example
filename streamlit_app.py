import torch
import transformers
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# @st.cache_resource(allow_output_mutation=True)
st.cache()
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("long292/bartpho-syllable-base-applied-backtranslation")
    model = AutoModelForSeq2SeqLM.from_pretrained("long292/bartpho-syllable-base-applied-backtranslation")
    return tokenizer, model
# Title and Textbox
st.title("Dịch Hán Văn")
tokenizer, model = get_model()
user_input = st.text_area('Nhập chữ vào bên dưới')
button = st.button("Dịch nghĩa")

if user_input and button:
    input_ids = tokenizer([user_input], return_tensors="pt").input_ids
    output =  model.generate(input_ids)
    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    st.success(f"{pred}")
    # st.write(pred)

# Add background color using markdown with unsafe_allow_html
background_color = "#FFFFFF"  # Change this to your desired color code
st.markdown(f"""<style>
.reportview-container {{
  background-color: {background_color};
}}
</style>""", unsafe_allow_html=True)

# import streamlit as st
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

# @st.cache(allow_output_mutation=True)
# def get_model():
#     tokenizer = AutoTokenizer.from_pretrained("long292/bartpho-syllable-base-applied-backtranslation")
#     model = AutoModelForSeq2SeqLM.from_pretrained("long292/bartpho-syllable-base-applied-backtranslation")
#     return tokenizer, model


# tokenizer,model = get_model()

# user_input = st.text_area('Enter Text to Analyze')
# button = st.button("Analyze")

# d = {
    
#   1:'Toxic',
#   0:'Non Toxic'
# }

# if user_input and button :
#     test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
#     # test_sample
#     output = model(**test_sample)
#     st.write("Logits: ",output.logits)
#     y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
#     st.write("Prediction: ",d[y_pred[0]])
