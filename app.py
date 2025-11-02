import streamlit as st
import base64
import traceback
from src.resume_parser import to_parse_the_resume
from src.model import JobRecommender
from src.preprocess import to_preprocess_and_to_evaluate

#  CACHE HANDLER 
@st.cache_data(show_spinner=False)
def load_dataset(sample_limit):
    """Cached data loader to preprocess job dataset."""
    return to_preprocess_and_to_evaluate(no_of_rows_max=sample_limit)


#  PAGE CONFIG 
st.set_page_config(
    page_title="AI Resume Analyzer & Job Recommender",
    page_icon="🏢",
    layout="wide",
)


#  BACKGROUND IMAGE 
def to_get_img_base64(path_of_the_img):
    try:
        with open(path_of_the_img, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        # Silently skip if image not found
        return None


path_of_the_img = "data/bgapp.JPG"
img_base64 = to_get_img_base64(path_of_the_img)

style_of_the_background = (
    f"""
    background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                url("data:image/jpg;base64,{img_base64}");
    """
    if img_base64
    else "background: linear-gradient(180deg, #1e293b, #0f172a);"
)

st.markdown(
    f"""
    <style>
    .stApp {{
        {style_of_the_background}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f1f5f9;
        font-family: 'Segoe UI', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background: rgba(20, 20, 20, 0.88);
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff !important;
    }}
    .card {{
        background: rgba(255, 255, 255, 0.12);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        border-radius: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        backdrop-filter: blur(8px);
        color: #ffffff;
        transition: all 0.2s ease;
    }}
    .card:hover {{
        transform: scale(1.02);
        box-shadow: 0 6px 18px rgba(0,0,0,0.5);
    }}
    h1, h2, h3, h4 {{
        color: #f8fafc;
    }}
    p, small {{
        color: #e2e8f0;
    }}
    [data-testid="stFileUploader"] {{
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 0.5rem;
        color: #111827 !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }}
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] * {{
        color: #111827 !important;
    }}
    [data-testid="stFileUploader"] div[role="button"] {{
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        color: #111827 !important;
    }}
    [data-testid="stFileUploader"] div[role="button"]:hover {{
        background-color: #e2e8f0 !important;
        color: #111827 !important;
    }}
    [data-testid="stFileUploader"] a,
    [data-testid="stFileUploader"] span {{
        color: #111827 !important;
        font-weight: 500;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


#  HEADER 
st.title("🏢 AI Resume Analyzer & Job Recommender")
st.markdown("""
Discover job roles that best fit your **skills**, **experience**, and **career goals**
""")


#  SIDEBAR 
st.sidebar.header("⚙️ Configuration")
sample_limit = st.sidebar.slider("📊 Dataset size (rows)", 200, 2000, 800, step=100)
top_k = st.sidebar.slider("🎯 Number of job recommendations", 3, 10, 5)
st.sidebar.markdown("---")
st.sidebar.info("💡 Increase dataset size for broader results (might be slower than the default size).")


#  MAIN LOGIC 
file_that_is_uploaded = st.file_uploader("📄 Upload your Resume", type=["pdf", "docx", "txt"])

if file_that_is_uploaded is not None:
    try:
        temp_path = f"data/{file_that_is_uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(file_that_is_uploaded.read())

        txt_in_the_resume = to_parse_the_resume(temp_path)

        if not txt_in_the_resume.strip():
            st.error("❌ Could not extract text. Please upload a text-based PDF or DOCX.")
        else:
            df_cleaned = load_dataset(sample_limit)
            recommender = JobRecommender(df_cleaned, max_rows=sample_limit)
            results = recommender.recommend(txt_in_the_resume, top_k=top_k)

            if not results.empty:
                st.success("🤝 Resume analyzed successfully!")
            else:
                st.warning("⚠️ No strong job matches found. Try increasing dataset size.")

            tab1, tab2, tab3 = st.tabs(
                ["📄 Preview of the Resume", "🎯 Job Recommendations", "💡 Career Insights"]
            )

            # Tab 1 - Resume Text
            with tab1:
                st.subheader("📄 Preview of the Resume")
                st.write(txt_in_the_resume[:2000] + ("..." if len(txt_in_the_resume) > 2000 else ""))

            # Tab 2 - Job Recommendations
            with tab2:
                if not results.empty:
                    st.subheader("🎯 Top Job Recommendations")
                    for _, row in results.iterrows():
                        title = row.get("JobTitle", "N/A")
                        company = row.get("Company Name", "Not Specified")
                        contact = row.get("Contact Person", "Not Provided")
                        skills = row.get("Skills", "Not Mentioned")
                        portal = row.get("Job Portal", "Not Specified")
                        score = row.get("similarity", 0.0)

                        if len(skills.split()) > 25:
                            skills = " ".join(skills.split()[:25]) + "..."

                        st.markdown(
                            f"""
                            <div class="card">
                                <h4>🏢 {title}</h4>
                                <p><b>📊 Match Score:</b> {score:.2f}%</p>
                                <p><b>🏢 Company:</b> {company}</p>
                                <p><b>👤 Contact:</b> {contact}</p>
                                <p><b>🧠 Skills:</b> {skills}</p>
                                <p><b>🌐 Job Portal:</b> {portal}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning("❌ No matching jobs found.")

            # Tab 3 - Resume Tips
            with tab3:
                st.subheader("🛠 Suggestions to Improve Resume")
                for s in recommender.improvements_suggested(txt_in_the_resume, results):
                    st.markdown(f"- {s}")

                st.subheader("💬 Preparation Tips for the Interview")
                for t in recommender.tips_for_the_interview(results):
                    st.markdown(f"- {t}")

    except Exception:
        st.error("🚨 An unexpected error occurred while processing your resume.")
        st.code(traceback.format_exc())
else:
    st.info("↗️ Upload your resume to start the analysis.")


#  FOOTER 
st.markdown(
    """
    <hr><center>
    <p style="color:#cbd5e1;font-size:0.9em;">
    © 2025 Gisma University | GH1033737<br>
    Developed by <b>Devi Varshitha Pagadala</b> 🤵‍♀️
    </p></center>
    """,
    unsafe_allow_html=True,
)