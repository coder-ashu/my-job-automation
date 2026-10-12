import os
import json
import pandas as pd
from jobspy import scrape_jobs
import google.generativeai as genai
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# 1. Setup Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Master Profile Baseline
USER_BASELINE = """
Ashutosh Prasad | IIT Patna B.Tech (Math & Computing, CGPA 8.23).
Skills: C++20, Python, Lock-Free Ring Buffers, Concurrency, Linux Kernel Tuning, Multithreading, AWS, FastAPI, Docker,Machine Learning , Neural Networks ,RAG, AgenticAI , Langraph/Langchain.
Experience: Teesta Investment (HFT SDE Intern - Low Latency Alerting & Kernel Isolation), Husk Power (SDE Intern).
Achievements: LeetCode Guardian (Top 1.4%), Codeforces Specialist (Max 1542).
Target: Product companies , HFTs , AI/ML Engineer roles , no service based company. Salary expectation > 18 LPA CTC / 16 Base.
"""

def fetch_and_filter_jobs():
    print("Scraping job listings...")
    jobs = scrape_jobs(
        site_name=["linkedin", "indeed", "glassdoor","naukri"],
        search_term="Software Engineer C++ Quantitative Developer AI Engineer  ML Enginner",
        location="India",
        results_wanted=20,
        hours_old=24,
        country_indeed='India'
    )
    
    if jobs.empty:
        print("No new jobs found in the last 24 hours.")
        return []

    processed_jobs = []
    
    # Exclude common service-based firms
    EXCLUDED_COMPANIES = ["tcs", "infosys", "wipro", "cognizant", "accenture", "capgemini", "hcltech", "ltimindtree"]

    for _, row in jobs.iterrows():
        company = str(row['company']).strip()
        title = str(row['title']).strip()
        desc = str(row['description'])
        job_url = str(row['job_url'])

        # Filter out service companies
        if any(service_firm in company.lower() for service_firm in EXCLUDED_COMPANIES):
            continue

        # Use Gemini to evaluate role fit and generate a tailored resume narrative
        prompt = f"""
        Analyze this job description for a candidate with the following baseline:
        {USER_BASELINE}

        Job Title: {title}
        Company: {company}
        Job Description: {desc[:2000]}

        Return JSON format ONLY:
        {{
            "is_fit": true/false,
            "estimated_ctc_match": "High/Medium/Low",
            "tailored_resume_bullets": ["Bullet 1 with keywords", "Bullet 2 focused on requirements"]
        }}
        """

        try:
            response = model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text)

            if result.get("is_fit", False):
                processed_jobs.append({
                    "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "Company": company,
                    "Title": title,
                    "Apply Link": job_url,
                    "Estimated CTC Match": result.get("estimated_ctc_match", "N/A"),
                    "Tailored Highlights": "\n".join(result.get("tailored_resume_bullets", [])),
                    "Status": "Pending Application"
                })
        except Exception as e:
            print(f"Error processing {title} at {company}: {e}")

    return processed_jobs

def update_google_sheet(job_list):
    if not job_list:
        return

    # Authenticate with Google Sheets API via Service Account JSON
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    sheet = client.open("Job Search Automation Tracker").sheet1

    for job in job_list:
        row = [
            job["Date"],
            job["Company"],
            job["Title"],
            job["Apply Link"],
            job["Estimated CTC Match"],
            job["Tailored Highlights"],
            job["Status"]
        ]
        sheet.append_row(row)
    print(f"Successfully appended {len(job_list)} job(s) to Google Sheets.")

if __name__ == "__main__":
    job_results = fetch_and_filter_jobs()
    update_google_sheet(job_results)