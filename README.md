# Resume analyzer and job recommendation

It is a lightweight application based on NLP that helps in resume analysis, extraction of skills from the resume, predicts the categories of the jobs, and gives recommendations based on jobs that are suitable for the roles using TF-IDF similarity. It is built with Streamlit, Amazon Web Services (AWS) S3, PyPDF2, pdfminer, and scikit-learn.


Features:

Text extraction from resume (supports PDF, DOCX, TXT)
Cleans the text and preprocesses it in an automatic way
Loads the job data from AWS S3 on deployment or local CSV
Computation of TF-IDF embeddings plus cosine similarity
Job domains are detected, such as QA, Data, Hr, Finance, Software, etc.
Gives top job recommendations with match scores (job match percentage)
Gives tips for improving the resume and suggestions for the interview
An interactive Streamlit UI with a themed background

Highlights:

Multi-engine parsing of a resume (PyPDF2, pdfminer, python-docx)
Job filtering that is domain-aware
Dataset handling is dynamic 
Clean UI which is user-friendly with tabs and options in the sidebar

Installation:

Clone the repo: git clone <https://github.com/Devipagadala/Business-project-in-bigdata-and-AI/>
cd <Business-project-in-bigdata-and-AI>
Install dependencies: pip install -r requirements.txt


AWS S3 Configuration:

Create: streamlit/secrets.toml
Add your values:
AWS_ACCESS_KEY_ID="your_key"
AWS_SECRET_ACCESS_KEY="your_secret"
AWS_REGION="your-region"
S3_BUCKET_NAME="your-bucket"
S3_OBJECT_KEY="job_descriptions.csv.gz"
The application  loads the environment automatically with environment variables either from .env or Streamlit Secrets.

Run the Application: streamlit run app.py

Dependencies:

Libraries that were used mainly:
Streamlit
PyPDF2
pdfminer
python-docx
scikit-learn
boto3
numpy
pandas

Limitations:

Doesn't support scanned pdfs or docxs
Keyword-based TF-IDF not semantic
Required cleaned and compressed dataset, formatted before uploading to S3
Large datasets slows down the process and might breakdown the streamlit

