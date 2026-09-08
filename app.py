```python
import os
from urllib.parse import urlparse, parse_qs, quote

import validators
import streamlit as st

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    UnstructuredURLLoader
)

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig


# ============================================================
# LOAD .ENV
# ============================================================

load_dotenv()


# ============================================================
# READ ENVIRONMENT VARIABLES
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

webshare_username = os.getenv("WEBSHARE_USERNAME")
webshare_password = os.getenv("WEBSHARE_PASSWORD")
webshare_host = os.getenv("WEBSHARE_HOST")
webshare_port = os.getenv("WEBSHARE_PORT")


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="YouTube & Website Summarizer",
    page_icon="📝",
    layout="wide"
)


# ============================================================
# CHECK GROQ API KEY
# ============================================================

if not api_key:

    st.error("GROQ_API_KEY not found.")

    st.info(
        "Please add GROQ_API_KEY to your .env file."
    )

    st.stop()


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "📝 LangChain: YouTube & Website Summarizer"
)

st.write(
    "Summarize YouTube videos or website content "
    "in your preferred language."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Summary Settings")


# ============================================================
# LANGUAGE OPTIONS
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


selected_language = st.sidebar.selectbox(
    "🌐 Select Summary Language",
    options=list(languages.keys()),
    index=0
)


st.sidebar.write(
    f"Selected: **{selected_language}**"
)


# ============================================================
# WEBSHARE STATUS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("🔐 YouTube Proxy")


proxy_configured = all([
    webshare_username,
    webshare_password,
    webshare_host,
    webshare_port
])


if proxy_configured:

    st.sidebar.success(
        "Webshare proxy configured"
    )

else:

    st.sidebar.warning(
        "Webshare proxy is not configured"
    )


# ============================================================
# URL INPUT
# ============================================================

generic_url = st.text_input(
    "Enter YouTube or Website URL",
    placeholder="https://www.youtube.com/watch?v=..."
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

1. Write the entire summary in {language}.
2. Do not mix languages.
3. Preserve important facts.
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
# EXTRACT YOUTUBE VIDEO ID
# ============================================================

def extract_youtube_video_id(url):

    try:

        parsed_url = urlparse(url)

        hostname = parsed_url.hostname

        if not hostname:
            return None

        hostname = hostname.lower()

        # ----------------------------------------------------
        # youtube.com/watch?v=VIDEO_ID
        # ----------------------------------------------------

        if "youtube.com" in hostname:

            query_params = parse_qs(
                parsed_url.query
            )

            video_id = query_params.get("v")

            if video_id:

                return video_id[0]


        # ----------------------------------------------------
        # youtu.be/VIDEO_ID
        # ----------------------------------------------------

        if "youtu.be" in hostname:

            video_id = parsed_url.path.strip("/")

            if video_id:

                return video_id.split("/")[0]


        # ----------------------------------------------------
        # youtube.com/embed/VIDEO_ID
        # ----------------------------------------------------

        if "/embed/" in parsed_url.path:

            video_id = parsed_url.path.split(
                "/embed/"
            )[1]

            if video_id:

                return video_id.split("/")[0]


        # ----------------------------------------------------
        # youtube.com/shorts/VIDEO_ID
        # ----------------------------------------------------

        if "/shorts/" in parsed_url.path:

            video_id = parsed_url.path.split(
                "/shorts/"
            )[1]

            if video_id:

                return video_id.split("/")[0]


        return None


    except Exception:

        return None


# ============================================================
# CREATE YOUTUBE CLIENT
# ============================================================

def create_youtube_client():

    # --------------------------------------------------------
    # Check proxy configuration
    # --------------------------------------------------------

    if not proxy_configured:

        return YouTubeTranscriptApi()


    # --------------------------------------------------------
    # URL-encode credentials
    #
    # This protects the proxy URL if username/password contains
    # special characters such as @, :, /, etc.
    # --------------------------------------------------------

    username = quote(
        webshare_username,
        safe=""
    )

    password = quote(
        webshare_password,
        safe=""
    )


    proxy_url = (
        f"http://{username}:{password}"
        f"@{webshare_host}:{webshare_port}"
    )


    # --------------------------------------------------------
    # Generic proxy configuration
    # --------------------------------------------------------

    return YouTubeTranscriptApi(
        proxy_config=GenericProxyConfig(
            http_url=proxy_url,
            https_url=proxy_url
        )
    )


# ============================================================
# GET YOUTUBE TRANSCRIPT
# ============================================================

def get_youtube_transcript(video_id):

    try:

        # ----------------------------------------------------
        # Create client
        # ----------------------------------------------------

        if proxy_configured:

            st.info(
                "🔐 Connecting to YouTube through Webshare..."
            )

        else:

            st.warning(
                "⚠️ Webshare proxy is not configured. "
                "Trying direct YouTube connection."
            )


        yt_api = create_youtube_client()


        # ----------------------------------------------------
        # Get available transcripts
        # ----------------------------------------------------

        transcript_list = yt_api.list(
            video_id
        )


        # ----------------------------------------------------
        # Preferred transcript languages
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
        # Find preferred transcript
        # ----------------------------------------------------

        for language in preferred_languages:

            try:

                transcript = (
                    transcript_list.find_transcript(
                        [language]
                    )
                )

                if transcript:

                    break

            except Exception:

                continue


        # ----------------------------------------------------
        # Fallback to any available transcript
        # ----------------------------------------------------

        if transcript is None:

            try:

                transcript = next(
                    iter(transcript_list)
                )

            except StopIteration:

                raise Exception(
                    "No transcript is available for this video."
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

            if hasattr(
                snippet,
                "text"
            ):

                text_parts.append(
                    snippet.text
                )

            elif isinstance(
                snippet,
                dict
            ):

                text_parts.append(
                    snippet.get(
                        "text",
                        ""
                    )
                )


        text = " ".join(
            text_parts
        )


        # ----------------------------------------------------
        # Validate transcript
        # ----------------------------------------------------

        if not text.strip():

            raise Exception(
                "Transcript was found but contains no text."
            )


        return text


    except Exception as e:

        error_message = str(e)


        # ----------------------------------------------------
        # Proxy authentication error
        # ----------------------------------------------------

        if (
            "407" in error_message
            or
            "Proxy Authentication Required"
            in error_message
        ):

            raise Exception(
                "Webshare proxy authentication failed (407).\n\n"
                "Please verify:\n"
                "• Webshare proxy username\n"
                "• Webshare proxy password\n"
                "• Proxy address/IP\n"
                "• Proxy port\n\n"
                "Make sure the credentials are the proxy "
                "credentials shown by Webshare."
            )


        # ----------------------------------------------------
        # YouTube IP block
        # ----------------------------------------------------

        if (
            "IpBlocked" in error_message
            or
            "RequestBlocked" in error_message
            or
            "YouTube is blocking" in error_message
        ):

            raise Exception(
                "YouTube is blocking the proxy IP.\n\n"
                "Try a different Webshare proxy, preferably "
                "a rotating residential proxy."
            )


        # ----------------------------------------------------
        # Transcript unavailable
        # ----------------------------------------------------

        if (
            "No transcripts were found" in error_message
            or
            "No transcript" in error_message
        ):

            raise Exception(
                "This YouTube video does not have a usable "
                "transcript."
            )


        # ----------------------------------------------------
        # Other error
        # ----------------------------------------------------

        raise Exception(
            f"Could not retrieve YouTube transcript:\n"
            f"{error_message}"
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


    if not validators.url(
        generic_url
    ):

        st.error(
            "Please enter a valid URL."
        )

        st.stop()


    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    try:

        # ====================================================
        # EXTRACT CONTENT
        # ====================================================

        with st.spinner(
            "⏳ Extracting content..."
        ):

            # ------------------------------------------------
            # YouTube
            # ------------------------------------------------

            if (
                "youtube.com" in generic_url.lower()
                or
                "youtu.be" in generic_url.lower()
            ):

                st.info(
                    "🎥 YouTube video detected."
                )


                video_id = (
                    extract_youtube_video_id(
                        generic_url
                    )
                )


                if not video_id:

                    st.error(
                        "Could not extract YouTube video ID."
                    )

                    st.stop()


                text = get_youtube_transcript(
                    video_id
                )


            # ------------------------------------------------
            # Website
            # ------------------------------------------------

            else:

                st.info(
                    "🌐 Website detected."
                )


                text = get_website_content(
                    generic_url
                )


        # ====================================================
        # VALIDATE CONTENT
        # ====================================================

        if not text.strip():

            st.error(
                "Could not extract any content."
            )

            st.stop()


        # ====================================================
        # SHOW EXTRACTED CONTENT
        # ====================================================

        with st.expander(
            "📄 View extracted content"
        ):

            st.write(
                text[:10000]
            )


        # ====================================================
        # GENERATE SUMMARY
        # ====================================================

        with st.spinner(
            f"🤖 Generating {selected_language} summary..."
        ):

            summary = pipeline.invoke(
                {
                    "text": text,
                    "language": languages[
                        selected_language
                    ]
                }
            )


        # ====================================================
        # DISPLAY SUMMARY
        # ====================================================

        st.subheader(
            f"📌 Summary ({selected_language})"
        )


        st.success(
            summary
        )


    except Exception as e:

        st.error(
            f"❌ Error: {e}"
        )
```
