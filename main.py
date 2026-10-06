import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
from docx import Document


# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="AI Notes Summarizer",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Notes Summarizer")
st.write("Upload your PDF, DOCX or TXT notes and generate an AI summary.")


# --------------------------------
# Load AI Model
# --------------------------------
@st.cache_resource
def load_model():

    model_name = "facebook/bart-large-cnn"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


# --------------------------------
# Extract PDF Text
# --------------------------------
def read_pdf(file):

    text = ""

    reader = PdfReader(file)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------
# Extract DOCX Text
# --------------------------------
def read_docx(file):

    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:

        text += paragraph.text + "\n"

    return text


# --------------------------------
# Extract File Text
# --------------------------------
def extract_text(file):

    extension = file.name.split(".")[-1].lower()

    if extension == "txt":

        return file.read().decode("utf-8")

    elif extension == "pdf":

        return read_pdf(file)

    elif extension == "docx":

        return read_docx(file)

    return ""


# --------------------------------
# Split Text
# --------------------------------
def split_text(text, max_words=400):

    words = text.split()

    chunks = []

    for i in range(0, len(words), max_words):

        chunk = " ".join(words[i:i + max_words])

        chunks.append(chunk)

    return chunks


# --------------------------------
# AI Summarization
# --------------------------------
def summarize_text(text, tokenizer, model):

    chunks = split_text(text)

    summaries = []

    for chunk in chunks:

        if len(chunk.split()) < 30:

            summaries.append(chunk)

            continue

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        summaries.append(summary)

    return " ".join(summaries)


# --------------------------------
# Upload File
# --------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload your notes",
    type=["pdf", "docx", "txt"]
)


# --------------------------------
# Generate Summary
# --------------------------------
if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button("🤖 Generate Summary"):

        with st.spinner("Loading AI model..."):

            tokenizer, model = load_model()

        with st.spinner("Reading your notes..."):

            text = extract_text(uploaded_file)

        if not text.strip():

            st.error(
                "❌ Could not extract text from the file."
            )

        else:

            st.subheader("📄 Extracted Notes")

            with st.expander("View Notes"):

                st.write(text)

            with st.spinner(
                "🤖 AI is generating your summary..."
            ):

                summary = summarize_text(
                    text,
                    tokenizer,
                    model
                )

            st.subheader("✨ AI Generated Summary")

            st.write(summary)

            st.download_button(
                "⬇️ Download Summary",
                summary,
                file_name="AI_Notes_Summary.txt",
                mime="text/plain"
            )

            st.success(
                "✅ Summary generated successfully!"
            )