import os
import validators
import streamlit as st

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    UnstructuredURLLoader,
)
from youtube_transcript_api import YouTubeTranscriptApi

from urllib.parse import urlparse, parse_qs

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

st.set_page_config(
    page_title="LangChain: Summarize Text from YT or Website",
    page_icon=":bird:"
)

st.title("LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# --------------------------------------------------
# Get URL
# --------------------------------------------------

generic_url = st.text_input(
    "URL",
    label_visibility="collapsed",
    placeholder="Paste your URL here (YT or Website)"
)

# --------------------------------------------------
# Language Selection
# --------------------------------------------------
with st.sidebar:
    
    languages = {
        "English": "English",
        "Hindi": "Hindi",
        "Marathi": "Marathi",
        "Gujarati": "Gujarati",
        "Bengali": "Bengali",
        "Tamil": "Tamil",
        "Telugu": "Telugu",
        "Kannada": "Kannada",
        "Malayalam": "Malayalam",
        "Punjabi": "Punjabi",
        "Urdu": "Urdu",
        "French": "French",
        "German": "German",
        "Spanish": "Spanish",
        "Italian": "Italian",
        "Portuguese": "Portuguese",
        "Japanese": "Japanese",
        "Korean": "Korean",
        "Chinese": "Chinese"
    }

    selected_language = st.selectbox(
        "Select Summary Language",
        options=list(languages.keys()),
        index=0
    )

# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-120b",
    temperature=0
)

# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt_template = """
Provide a clear and concise summary of the following content
in approximately 300 words.

IMPORTANT:
- Write the entire summary in {language}.
- Do not mix languages.
- Keep the important facts, key points, and main ideas.
- Do not add information that is not present in the content.

Content:
{text}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# --------------------------------------------------
# Chain
# --------------------------------------------------

pipeline = prompt | llm | StrOutputParser()

# --------------------------------------------------
# Summarize Button
# --------------------------------------------------

if st.button("Summarize") and generic_url:

    if not api_key or not api_key.strip():

        st.error("Please provide GROQ_API_KEY.")

    elif not validators.url(generic_url):

        st.error(
            "Please enter a valid URL. "
            "It can be a YouTube URL or Website URL."
        )

    else:

        try:

            with st.spinner("Summarizing......"):

                # --------------------------------------------------
                # YouTube
                # --------------------------------------------------

                if "youtube.com" in generic_url or "youtu.be" in generic_url:

                    # Extract video ID
                    parsed_url = urlparse(generic_url)

                    if "youtube.com" in generic_url:

                        video_id = parse_qs(
                            parsed_url.query
                        ).get("v", [None])[0]

                    else:

                        video_id = parsed_url.path.strip("/")

                    if not video_id:

                        st.error(
                            "Could not extract YouTube video ID."
                        )

                        st.stop()

                    # YouTube Transcript API
                    yt_api = YouTubeTranscriptApi()

                    transcript_list = yt_api.list(video_id)

                    # Try English first
                    try:

                        transcript = transcript_list.find_transcript(
                            ["en"]
                        )

                    except Exception:

                        # If English is not available,
                        # use first available transcript
                        transcript = next(iter(transcript_list))

                    # Fetch transcript
                    transcript_data = transcript.fetch()

                    # Convert transcript to text
                    text = " ".join(
                        snippet.text
                        for snippet in transcript_data
                    )

                # --------------------------------------------------
                # Website
                # --------------------------------------------------

                else:

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 "
                                "(KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        }
                    )

                    docs = loader.load()

                    text = "\n\n".join(
                        doc.page_content
                        for doc in docs
                    )

                # --------------------------------------------------
                # Check Content
                # --------------------------------------------------

                if not text.strip():

                    st.error(
                        "Could not extract any content from the URL."
                    )

                    st.stop()

                # --------------------------------------------------
                # Summarize
                # --------------------------------------------------

                summary = pipeline.invoke({
                    "text": text,
                    "language": languages[selected_language]
                })

                # --------------------------------------------------
                # Display Summary
                # --------------------------------------------------

                st.subheader(
                    f"Summary ({selected_language})"
                )

                st.success(summary)

        except Exception as e:

            st.error(
                f"Error exception: {e}"
            )