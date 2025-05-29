import streamlit as st
from transformers import pipeline

# Set up the Streamlit page
st.set_page_config(page_title="T5 Text Summarizer", layout="centered")

# Add background image (update the URL to your preferred image)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://media.istockphoto.com/id/1623303770/photo/creative-background-image-is-blurred-evening-city-lights-and-light-snowfall.webp?s=1024x1024&w=is&k=20&c=d9LTc-sVJNNwujJKhYGV6Gcaei5otu2PdZwtqUKGrbY=");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📝 T5 Text Summarizer")
st.markdown("""Enter a paragraph of text below, and this app will generate a summary using the T5 model.
You can adjust the length of the summary using the slider.

""")

# Load the summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small", tokenizer="t5-small")

summarizer = load_summarizer()

# Text input
input_text = st.text_area("Enter the text to summarize:", height=250)

# Length slider
max_length = st.slider("Select the maximum length of the summary:", min_value=20, max_value=150, value=50)

# Summarize button
if st.button("Summarize"):
    if input_text.strip():
        with st.spinner("Summarizing..."):
            input_text_prefixed = "summarize: " + input_text.strip()
            summary = summarizer(input_text_prefixed, max_length=max_length, min_length=10, do_sample=False, truncation=True)
            st.subheader("🔍 Summary")
            st.success(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text to summarize.")
