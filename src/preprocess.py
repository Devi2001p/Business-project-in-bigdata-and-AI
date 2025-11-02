"""
Handles preprocessing, cleaning, and evaluation for the AI Resume Analyzer & Job Recommender.
Performs dataset cleaning, exploratory data analysis (EDA), and evaluates recommender accuracy.
"""

import pandas as pd
import re
import boto3
import pandas as pd
import os
from io import BytesIO
import gzip
from dotenv import load_dotenv

def load_env():
    if os.path.exists(".env"):
        load_dotenv()
        print(".env loaded locally")
    elif hasattr(st, "secrets"):
        for key, value in st.secrets.items():
            os.environ[key] = str(value)
        print("Loaded environment from Streamlit secrets")
    else:
        print("❌No environment variables found")

load_env()

def load_dataset_from_s3():

    bucket = os.getenv("S3_BUCKET_NAME")
    key = os.getenv("S3_OBJECT_KEY")

    if not bucket or not key:
        raise ValueError("❌ s3 bucket name or path is missing")

    print(f"dataset is loading from s3 bucket: {bucket}/{key}")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    obj = s3.get_object(Bucket=bucket, Key=key)
    if key.endswith(".gz"):
        with gzip.GzipFile(fileobj=BytesIO(obj["Body"].read())) as gz:
            df = pd.read_csv(gz, low_memory=False)
    else:
        df = pd.read_csv(BytesIO(obj["Body"].read()), low_memory=False)

    print(f" successful loading of dataset from s3 is done — {len(df)} rows, {len(df.columns)} columns")
    return df
# This function is used in cleaning the text which removes symbols that are unwanted, spaces etc..

def text_that_is_cleaned(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# This is the preprocess function which performs eda and evaluates the accuracy of the recommender

def to_preprocess_and_to_evaluate(csv_path: str = "data/job_descriptions.csv", no_of_rows_max: int = 2000):
    from src.model import JobRecommender, to_detect_the_category_of_resume

    #  Here the dataset is loading from s3 bucket
    try:
      if os.getenv("S3_BUCKET_NAME"):
        df = load_dataset_from_s3()
      else:
        print(f"The dataset is loading from: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"The dataset is loaded with {len(df)} rows & {len(df.columns)} columns")
    except FileNotFoundError:
        print("❌ The dataset was not found in either of the paths, please check again.")
        return None

    # no of rows are limited for performance
    if len(df) > no_of_rows_max:
        df = df.sample(no_of_rows_max, random_state=42).reset_index(drop=True)
        print(f"the function is limited to {no_of_rows_max} rows to process faster")

    # to normalize the columns for renaming
    df.columns = [c.strip().lower().replace("_", " ").replace("-", " ") for c in df.columns]

    collection_of_map = {
        "JobTitle": ["job title", "title", "position"],
        "JobDescription": ["job description", "description", "jd"],
        "Company Name": ["company", "employer name", "organization name", "business name"],
        "Contact Person": ["contact person", "recruiter name", "hiring manager"],
        "Skills": ["skills", "skill", "required skills", "job skills"],
        "Job Portal": ["job portal", "portal", "source", "website"],
    }

    to_rename_the_dictionary = {}
    for standard, variants in collection_of_map.items():
        for col in df.columns:
            if col in variants:
                to_rename_the_dictionary[col] = standard

    df.rename(columns=to_rename_the_dictionary, inplace=True)

    # Clenaing the data
    for col in ["JobTitle", "JobDescription", "Company Name", "Contact Person", "Skills"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    previous = len(df)
    col_of_subset = [c for c in ["JobTitle", "JobDescription", "Skills"] if c in df.columns]
    df.drop_duplicates(subset=col_of_subset, inplace=True, ignore_index=True)
    print(f"🧹 Removed {previous - len(df)} duplicate rows")

    for col in ["JobTitle", "JobDescription", "Skills"]:
        if col in df.columns:
            df[col] = df[col].apply(text_that_is_cleaned)

    print(f"The dataset is ready which is cleaned and in good shape: {df.shape}")

    # Exploratory data analysis

    print("\n....summary of dataset....")
    print(f"No of rows total: {len(df)}")
    print(f"job titles that are unique: {df['JobTitle'].nunique()}")
    if "Company Name" in df.columns:
        print(f"companies that are unique: {df['Company Name'].nunique()}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("values that are missing (non-zero):\n", missing if not missing.empty else "None")

    if "Skills" in df.columns:
        all_skills = " ".join(df["Skills"].dropna().astype(str).tolist()).lower().split()
        print(f"skills words that are unique in total: {len(set(all_skills))}")

   
    # Evaluating the model to test accuracy

    print("\nevaluating the accuracy of the model")
    jr = JobRecommender(df, max_rows=no_of_rows_max)

    resumes_to_test = [
        ("qa", "QA engineer skilled in Selenium, automation testing, and API validation."),
        ("finance", "Financial analyst experienced in audits, budgeting, and Excel dashboards."),
        ("software", "Software developer with skills in Python, React, and API development."),
        ("data", "Data analyst skilled in SQL, Power BI, and data visualization."),
        ("marketing", "Digital marketer experienced in SEO, content strategy, and Google Ads."),
        ("hr", "HR manager with experience in recruitment and payroll systems."),
    ]

    correct = 0
    for true_cat, resume_text in resumes_to_test:
        pred_cat = to_detect_the_category_of_resume(resume_text) or "other"
        results = jr.recommend(resume_text, top_k=3)
        print(f"\ncategory of resume: {true_cat}")
        print(f"category of predictions: {pred_cat}")
        print("Jobs that are recommended on top", results["JobTitle"].head(3).tolist())
        if pred_cat == true_cat:
            correct += 1

    accuracy = round((correct / len(resumes_to_test)) * 100, 2)
    print(f"\n Accuracy of model (matching category): {accuracy}%")
    print("-")
    print(" preprocessing and evaluation done...")
    return df

if __name__ == "__main__":
    to_preprocess_and_to_evaluate()