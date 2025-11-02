"""
This file is an integrated model for the app 'AI Resume Analyzer & Job Recommender'
TF-IDF is used to match the texts, connected with the file preprocess.py for the data that is cleaned
which supports the input of dataframe input which is preprocessed and csv based fallback
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import text_that_is_cleaned 

# To detect the category of jobs to filter the resume by content

Patterns_of_categories: Dict[str, List[str]] = {
    "qa": [r"\bqa\b", r"quality\s+assurance", r"selenium", r"pytest", r"automation\b", r"sdet", r"cypress", r"testing"],
    "finance": [r"\bfinance\b", r"accounting", r"auditing", r"\btax\b", r"fp&a", r"reconciliation", r"budget"],
    "data": [r"\bdata\s+(analyst|scientist|engineer)\b", r"\bml\b", r"\bai\b", r"\bsql\b", r"python", r"pandas", r"power\s*bi", r"tableau"],
    "software": [r"\bsoftware\s+(developer|engineer)\b", r"java|node|react|django|spring|.net", r"\bapi\b"],
    "hr": [r"\bhr\b", r"recruit(ment|er)", r"talent\s+acquisition", r"payroll"],
    "marketing": [r"\bmarketing\b", r"\bseo\b", r"\bsem\b", r"campaign", r"content\s+marketing"],
    "admin": [r"administrative\s+assistant", r"office\s+assistant", r"\bclerical\b", r"documentation"],
}

Filter_for_title_category: Dict[str, List[str]] = {
    "qa": ["qa", "quality", "test", "testing", "automation", "sdet"],
    "finance": ["finance", "account", "audit", "tax", "analyst", "controller"],
    "data": ["data", "ml", "ai", "analytics", "scientist", "engineer"],
    "software": ["software", "developer", "engineer", "programmer", "web", "frontend", "backend"],
    "hr": ["hr", "recruit", "talent", "payroll"],
    "marketing": ["marketing", "seo", "sem", "content", "digital"],
    "admin": ["admin", "assistant", "office"],
}

# This function is used to detect the category of resume like software, qa tec...

def to_detect_the_category_of_resume(text: str) -> Optional[str]:
    text = str(text).lower()
    for cat, pats in Patterns_of_categories.items():
        for p in pats:
            if re.search(p, text):
                return cat
    return None

# This class used to recommend the jobs

class JobRecommender:
    def __init__(self, jobs_csv_path: str | pd.DataFrame = "data/job_descriptions.csv", max_rows: int = 2000):
        print(" Job Recommender is getting initialized (TF-IDF) ")
        self.max_rows = max_rows

        # Inputs for both dataframe and csv are handled
        if isinstance(jobs_csv_path, pd.DataFrame):
            df = jobs_csv_path.copy()
            print(f" Using the dataframe which is preprocessed: {len(df)} rows, {len(df.columns)} columns")
        else:
            df = pd.read_csv(jobs_csv_path, low_memory=False)
            print(f" Dataset which is loaded: {len(df)} rows, {len(df.columns)} columns")

        # Fill and clean
        for col in df.columns:
            df[col] = df[col].astype(str).fillna("")

        for col in ["JobTitle", "JobDescription", "Skills"]:
            if col in df.columns:
                df[col] = df[col].apply(text_that_is_cleaned)

        # dataset is limited for stability
        if len(df) > self.max_rows:
            df = df.sample(self.max_rows, random_state=42).reset_index(drop=True)
            print(f" dataset is limited to {self.max_rows} rows for stability")

        self.jobs_df = df
        self.summary_of_eda(df)

        # To build TF-IDF embeddings
        print(" TF_IDF matrix is getting build skills + job description...")
        texts_for_jobs = self.to_compose_jobcorpus(df)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self.job_matrix = self.vectorizer.fit_transform(texts_for_jobs)
        print(" The model is ready!!!\n")

    # Summary of exploratory data analysis
    def summary_of_eda(self, df: pd.DataFrame):
        print("\n Summary of dataset ")
        print(f"No of rows in total: {len(df)}")
        print(f"Jobs that are unique (titles): {df['JobTitle'].nunique() if 'JobTitle' in df.columns else 'N/A'}")
        print(f"Companies that are unique: {df['Company Name'].nunique() if 'Company Name' in df.columns else 'N/A'}")
        if "Skills" in df.columns:
            all_skills = " ".join(df["Skills"].astype(str)).split()
            print(f"Skills found that are unique: {len(set(all_skills))}")

    # This function helps in composing job corpus which combines skills, title and description into one text which can be searched
    def to_compose_jobcorpus(self, df: pd.DataFrame) -> List[str]:
        titles = df.get("JobTitle", "").astype(str)
        skills = df.get("Skills", "").astype(str)
        descs = df.get("JobDescription", "").astype(str)
        return (titles + " || " + skills + " || " + descs).apply(text_that_is_cleaned).tolist()

    # This function is used in filtering the resume by category
    def to_filter_the_jobs_basedon_category(self, df: pd.DataFrame, resume_text: str) -> Tuple[pd.DataFrame, Optional[str]]:
        cat = to_detect_the_category_of_resume(resume_text)
        if not cat:
            return df, None

        keywords = Filter_for_title_category.get(cat, [])
        if not keywords:
            return df, cat

        mask = (
            df["JobTitle"].str.contains("|".join(keywords), case=False, na=False)
            | df["JobDescription"].str.contains("|".join(keywords), case=False, na=False)
            | df["Skills"].str.contains("|".join(keywords), case=False, na=False)
        )

        filtered = df[mask].copy()
        print(f" Category that is detected: {cat.upper()} | Filtered to {len(filtered)} rows")

        # when no jobs are found this is the fallback
        if len(filtered) < 5:
            print(" Jobs found are few which are relavent, after usage of full dataset")
            return df, cat

        return filtered, cat

    # This is the function to recommend jobs
    def recommend(self, resume_text: str, top_k: int = 5) -> pd.DataFrame:
        jobs_filtered, cat = self.to_filter_the_jobs_basedon_category(self.jobs_df, resume_text)
        resume_vec = self.vectorizer.transform([text_that_is_cleaned(resume_text)])
        job_vecs = self.vectorizer.transform(self.to_compose_jobcorpus(jobs_filtered))
        sim = cosine_similarity(job_vecs, resume_vec).flatten()

        if len(sim) == 0:
            return pd.DataFrame(columns=["JobTitle", "Company Name", "Contact Person", "Skills", "Job Portal", "similarity"])

        top_idx = np.argpartition(-sim, kth=min(top_k - 1, len(sim) - 1))[:top_k]
        top_idx = top_idx[np.argsort(-sim[top_idx])]

        keep_cols = [c for c in ["JobTitle", "Company Name", "Contact Person", "Skills", "Job Portal"] if c in jobs_filtered.columns]
        out = jobs_filtered.iloc[top_idx][keep_cols].copy()
        out["similarity"] = (np.clip(sim[top_idx], 0, 1) * 100).round(2)
        out.reset_index(drop=True, inplace=True)
        return out

    # This part give the suggestions for resume and suggests some tips for interview
    def improvements_suggested(self, resume_text: str, top_results: pd.DataFrame) -> List[str]:
        cat = to_detect_the_category_of_resume(resume_text) or "general"
        base = {
            "qa": ["Add automation tools (Selenium/Cypress)", "Quantify coverage & defects found", "Mention CI/CD tools"],
            "finance": ["Show audit/budget outcomes", "Add Excel or BI tools", "Include certifications like CMA/CPA"],
            "data": ["Add data pipeline examples", "Highlight Power BI or SQL projects", "Include ML model metrics"],
            "software": ["List frameworks used", "Show performance improvements", "Mention CI/CD & testing tools"],
            "general": ["Add measurable achievements", "Use action verbs", "Keep layout clean & ATS-friendly"],
        }
        return base.get(cat, base["general"])

    def tips_for_the_interview(self, top_results: pd.DataFrame) -> List[str]:
        title_blob = " ".join(top_results.get("JobTitle", pd.Series([], dtype=str)).astype(str)).lower()
        if any(x in title_blob for x in ["qa", "test", "quality"]):
            return ["Revise testing concepts", "Be ready with automation tools", "Show debugging workflow"]
        if any(x in title_blob for x in ["finance", "account", "audit", "tax"]):
            return ["Revise accounting principles", "Show Excel/Power BI skills", "Know audit workflows"]
        if "data" in title_blob:
            return ["Explain an ML project", "Revise SQL queries", "Show data visualization storytelling"]
        if any(x in title_blob for x in ["software", "developer", "engineer"]):
            return ["Revise DSA", "Discuss design patterns", "Highlight debugging experience"]
        return ["Be confident", "Research the company", "Show measurable impact in your answers"]



'''# CLI Test (optional standalone run)

if __name__ == "__main__":
    print("🔍 Testing model integration with preprocess...")
    from src.preprocess import to_preprocess_and_to_evaluate

    df_cleaned = to_preprocess_and_to_evaluate()  # Run preprocessing
    jr = JobRecommender(df_cleaned, max_rows=1000)
    resume = "Software engineer skilled in Python, React, and APIs."
    print(jr.recommend(resume))'''