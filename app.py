import os
from urllib.parse import urlparse, parse_qs

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
from youtube_transcript_api.proxies import WebshareProxyConfig


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GET SECRETS
# ============================================================

def get_secret(name):
    """
    Get secret from:
    1. Environment variable (.env)
    2. Streamlit Secrets (Streamlit Cloud)
    """

    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets[name]
    except Exception:
        return None


# Groq API Key
api_key = get_secret("GROQ_API_KEY")

# Webshare proxy credentials
webshare_username = get_secret("WEBSHARE_USERNAME")
webshare_password = get_secret("WEBSHARE_PASSWORD")


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="LangChain URL Summarizer",
    page_icon="📝",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📝 LangChain: YouTube & Website Summarizer")

st.write(
    "Summarize YouTube videos or website content "
    "in your preferred language."
)


# ============================================================
# CHECK GROQ API KEY
# ============================================================

if not api_key:

    st.error(
        "GROQ_API_KEY is not configured. "
        "Please add it to your .env file or Streamlit Secrets."
    )

    st.stop()


# ============================================================
# LANGUAGE SELECTION
# ============================================================

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
    "🌐 Select Summary Language",
    list(languages.keys())
)


# ============================================================
# URL INPUT
# ============================================================

generic_url = st.text_input(
    "URL",
    placeholder="Paste YouTube or Website URL here..."
)


# ============================================================
# GROQ LLM
# ============================================================

llm = ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-120b",
    temperature=0
)


# ============================================================
# PROMPT
# ============================================================

prompt_template = """
You are an expert content summarizer.

Summarize the following content in approximately 300 words.

IMPORTANT INSTRUCTIONS:

1. Write the complete summary in {language}.
2. Do not mix languages.
3. Preserve the important facts.
4. Include the main ideas and key points.
5. Remove unnecessary repetition.
6. Do not invent information.
7. Make the summary easy to understand.
8. Use paragraphs and bullet points where appropriate.

Target language:
{language}

Content:
{text}
"""


prompt = ChatPromptTemplate.from_template(
    prompt_template
)


# ============================================================
# LANGCHAIN CHAIN
# ============================================================

pipeline = (
    prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# YOUTUBE VIDEO ID EXTRACTION
# ============================================================

def extract_youtube_video_id(url):

    parsed_url = urlparse(url)

    hostname = parsed_url.hostname

    if not hostname:
        return None

    hostname = hostname.lower()

    # --------------------------------------------------------
    # youtube.com/watch?v=VIDEO_ID
    # --------------------------------------------------------

    if "youtube.com" in hostname:

        query_params = parse_qs(parsed_url.query)

        video_id = query_params.get("v")

        if video_id:
            return video_id[0]

    # --------------------------------------------------------
    # youtu.be/VIDEO_ID
    # --------------------------------------------------------

    if "youtu.be" in hostname:

        video_id = parsed_url.path.strip("/")

        if video_id:
            return video_id.split("/")[0]

    # --------------------------------------------------------
    # youtube.com/embed/VIDEO_ID
    # --------------------------------------------------------

    if "/embed/" in parsed_url.path:

        video_id = parsed_url.path.split("/embed/")[1]

        if video_id:
            return video_id.split("/")[0]

    # --------------------------------------------------------
    # youtube.com/shorts/VIDEO_ID
    # --------------------------------------------------------

    if "/shorts/" in parsed_url.path:

        video_id = parsed_url.path.split("/shorts/")[1]

        if video_id:
            return video_id.split("/")[0]

    return None


# ============================================================
# GET YOUTUBE TRANSCRIPT
# ============================================================

def get_youtube_transcript(video_id):

    try:

        # ----------------------------------------------------
        # Create YouTube API client
        # ----------------------------------------------------

        if webshare_username and webshare_password:

            yt_api = YouTubeTranscriptApi(
                proxy_config=WebshareProxyConfig(
                    proxy_username=webshare_username,
                    proxy_password=webshare_password,
                )
            )

        else:

            st.warning(
                "Webshare proxy credentials were not found. "
                "YouTube may block requests from Streamlit Cloud."
            )

            yt_api = YouTubeTranscriptApi()


        # ----------------------------------------------------
        # Get available transcripts
        # ----------------------------------------------------

        transcript_list = yt_api.list(video_id)


        # ----------------------------------------------------
        # Try languages in preferred order
        # ----------------------------------------------------

        preferred_languages = [
            "en",
            "hi",
            "mr",
            "gu",
            "bn",
            "ta",
            "te",
            "kn",
            "ml",
            "pa",
            "ur"
        ]


        transcript = None


        # ----------------------------------------------------
        # Try preferred language
        # ----------------------------------------------------

        for language in preferred_languages:

            try:

                transcript = transcript_list.find_transcript(
                    [language]
                )

                if transcript:
                    break

            except Exception:
                continue


        # ----------------------------------------------------
        # If no preferred language found,
        # use first available transcript
        # ----------------------------------------------------

        if transcript is None:

            try:

                transcript = next(iter(transcript_list))

            except StopIteration:

                raise Exception(
                    "No transcripts are available for this video."
                )


        # ----------------------------------------------------
        # Fetch transcript
        # ----------------------------------------------------

        transcript_data = transcript.fetch()


        # ----------------------------------------------------
        # Convert transcript to text
        # ----------------------------------------------------

        text_parts = []

        for snippet in transcript_data:

            # New youtube-transcript-api versions
            if hasattr(snippet, "text"):

                text_parts.append(snippet.text)

            # Older versions
            elif isinstance(snippet, dict):

                text_parts.append(
                    snippet.get("text", "")
                )


        text = " ".join(text_parts)


        if not text.strip():

            raise Exception(
                "Transcript was found but contains no text."
            )


        return text


    except Exception as e:

        raise Exception(
            f"Could not retrieve YouTube transcript: {e}"
        )


# ============================================================
# GET WEBSITE CONTENT
# ============================================================

def get_website_content(url):

    try:

        loader = UnstructuredURLLoader(
            urls=[url],
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


        if not docs:

            raise Exception(
                "No content could be extracted from this website."
            )


        text = "\n\n".join(
            doc.page_content
            for doc in docs
        )


        if not text.strip():

            raise Exception(
                "Website content is empty."
            )


        return text


    except Exception as e:

        raise Exception(
            f"Could not retrieve website content: {e}"
        )


# ============================================================
# SUMMARIZE BUTTON
# ============================================================

if st.button(
    "✨ Summarize",
    type="primary"
):

    # --------------------------------------------------------
    # Validate URL
    # --------------------------------------------------------

    if not generic_url:

        st.warning(
            "Please enter a YouTube or Website URL."
        )

        st.stop()


    if not validators.url(generic_url):

        st.error(
            "Please enter a valid URL."
        )

        st.stop()


    # --------------------------------------------------------
    # Start processing
    # --------------------------------------------------------

    try:

        with st.spinner(
            "⏳ Extracting content and generating summary..."
        ):


            # =================================================
            # YOUTUBE
            # =================================================

            if (
                "youtube.com" in generic_url.lower()
                or "youtu.be" in generic_url.lower()
            ):

                st.info(
                    "🎥 YouTube video detected."
                )


                # Extract video ID
                video_id = extract_youtube_video_id(
                    generic_url
                )


                if not video_id:

                    st.error(
                        "Could not extract YouTube video ID."
                    )

                    st.stop()


                # Get transcript
                text = get_youtube_transcript(
                    video_id
                )


            # =================================================
            # WEBSITE
            # =================================================

            else:

                st.info(
                    "🌐 Website detected."
                )


                text = get_website_content(
                    generic_url
                )


            # =================================================
            # CONTENT VALIDATION
            # =================================================

            if not text.strip():

                st.error(
                    "Could not extract content from the URL."
                )

                st.stop()


            # =================================================
            # OPTIONAL: SHOW EXTRACTED CONTENT
            # =================================================

            with st.expander(
                "📄 View extracted content"
            ):

                st.write(text[:10000])


            # =================================================
            # GENERATE SUMMARY
            # =================================================

            summary = pipeline.invoke(
                {
                    "text": text,
                    "language": languages[
                        selected_language
                    ]
                }
            )


            # =================================================
            # DISPLAY SUMMARY
            # =================================================

            st.subheader(
                f"📌 Summary ({selected_language})"
            )


            st.success(summary)


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            f"Error exception: {e}"
        )
