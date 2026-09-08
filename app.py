
import validators
import streamlit as st

from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    UnstructuredURLLoader
)

from youtube_transcript_api import YouTubeTranscriptApi


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="YouTube & Website Summarizer",
    page_icon="📝",
    layout="wide"
)


# ============================================================
# GROQ API KEY
# ============================================================

# For Streamlit Cloud use st.secrets
api_key = st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is missing."
    )

    st.info(
        "Add GROQ_API_KEY in "
        "Streamlit Cloud → Manage app → Settings → Secrets."
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title(
    "📝 YouTube & Website Summarizer"
)

st.write(
    "Summarize YouTube videos and websites "
    "in multiple languages."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Summary Settings")


# ============================================================
# SUMMARY LANGUAGES
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


# ============================================================
# SUMMARY LENGTH
# ============================================================

summary_length = st.sidebar.selectbox(
    "📏 Summary Length",
    options=[
        "150 words",
        "300 words",
        "500 words"
    ],
    index=1
)


# ============================================================
# SOURCE OPTION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("🎥 YouTube")

st.sidebar.info(
    "On Streamlit Cloud free tier, YouTube may block "
    "transcript requests because the app runs from a "
    "cloud-provider IP."
)


# ============================================================
# URL INPUT
# ============================================================

generic_url = st.text_input(
    "🔗 Enter YouTube or Website URL",
    placeholder="https://www.youtube.com/watch?v=..."
)


# ============================================================
# MANUAL TRANSCRIPT FALLBACK
# ============================================================

with st.expander(
    "📝 Paste YouTube transcript manually (free fallback)"
):

    manual_transcript = st.text_area(
        "Paste transcript here",
        height=200,
        placeholder=(
            "If YouTube blocks transcript retrieval, "
            "paste the transcript here and the app will summarize it."
        )
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
# SUMMARY PROMPT
# ============================================================

prompt_template = """
You are an expert content summarizer.

Summarize the following content in approximately
{summary_length}.

IMPORTANT INSTRUCTIONS:

1. Write the entire summary in {language}.
2. Do not mix languages.
3. Preserve important facts.
4. Include the main ideas and key points.
5. Remove unnecessary repetition.
6. Do not invent information.
7. Do not add facts that are not present.
8. Make the summary easy to understand.
9. Use headings and bullet points when useful.

Target language:
{language}

Summary length:
{summary_length}

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
        # youtube.com/shorts/VIDEO_ID
        # ----------------------------------------------------

        if "/shorts/" in parsed_url.path:

            video_id = parsed_url.path.split(
                "/shorts/"
            )[1]

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


        return None


    except Exception:

        return None


# ============================================================
# GET YOUTUBE TRANSCRIPT - DIRECT
# ============================================================

def get_youtube_transcript(video_id):

    try:

        # ----------------------------------------------------
        # Create API client
        # ----------------------------------------------------

        yt_api = YouTubeTranscriptApi()


        # ----------------------------------------------------
        # Get available transcripts
        # ----------------------------------------------------

        transcript_list = yt_api.list(
            video_id
        )


        # ----------------------------------------------------
        # Preferred languages
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
        # Try preferred languages
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
        # If no preferred language exists,
        # use first available transcript
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
        # Validate
        # ----------------------------------------------------

        if not text.strip():

            raise Exception(
                "Transcript was found but contains no text."
            )


        return text


    except Exception as e:

        raise Exception(
            str(e)
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
# MAIN BUTTON
# ============================================================

if st.button(
    "✨ Summarize",
    type="primary"
):

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if not generic_url and not manual_transcript:

        st.warning(
            "Please enter a URL or paste a transcript."
        )

        st.stop()


    # ========================================================
    # CONTENT VARIABLE
    # ========================================================

    text = ""


    try:

        # ====================================================
        # MANUAL TRANSCRIPT HAS PRIORITY
        # ====================================================

        if manual_transcript.strip():

            st.info(
                "📝 Using manually provided transcript."
            )

            text = manual_transcript.strip()


        # ====================================================
        # URL PROCESSING
        # ====================================================

        elif generic_url:

            # ------------------------------------------------
            # Validate URL
            # ------------------------------------------------

            if not validators.url(
                generic_url
            ):

                st.error(
                    "Please enter a valid URL."
                )

                st.stop()


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


                # ------------------------------------------------
                # Try direct transcript
                # ------------------------------------------------

                with st.spinner(
                    "⏳ Trying to retrieve YouTube transcript..."
                ):

                    try:

                        text = get_youtube_transcript(
                            video_id
                        )

                    except Exception as youtube_error:

                        st.error(
                            "❌ YouTube transcript could not "
                            "be retrieved."
                        )

                        st.warning(
                            "Streamlit Cloud free tier may be "
                            "blocked by YouTube because the app "
                            "runs from a cloud-provider IP."
                        )

                        st.info(
                            "Please use the 'Paste YouTube "
                            "transcript manually' box above."
                        )

                        st.caption(
                            f"Technical error: {youtube_error}"
                        )

                        st.stop()


            # ------------------------------------------------
            # Website
            # ------------------------------------------------

            else:

                st.info(
                    "🌐 Website detected."
                )


                with st.spinner(
                    "⏳ Extracting website content..."
                ):

                    text = get_website_content(
                        generic_url
                    )


        # ====================================================
        # VALIDATE CONTENT
        # ====================================================

        if not text.strip():

            st.error(
                "No content was extracted."
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
                    ],
                    "summary_length": summary_length
                }
            )


        # ====================================================
        # DISPLAY RESULT
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

