import os
import json
import time
import pandas as pd
from jobspy import scrape_jobs
from huggingface_hub import InferenceClient
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# 1. Setup Hugging Face Client
client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=os.getenv("HF_TOKEN")
)

USER_BASELINE = """
Ashutosh Prasad | IIT Patna B.Tech (Math & Computing, CGPA 8.23)[cite: 1, 2].
Skills: C++20, Python, Lock-Free Ring Buffers, Concurrency, Linux Kernel Tuning, Multithreading, AWS, FastAPI, Docker, Machine Learning ,Deep Learning, Agentic AI, RAG, Langchain/langraph,Fine Tuning.
Experience: Teesta Investment (HFT SDE Intern - Low Latency Alerting & Kernel Isolation), Husk Power (SDE Intern).
Achievements: LeetCode Guardian (Top 1.4%), Codeforces Specialist (Max 1542)[cite: 1, 2].
Target: Product companies or HFTs only. Salary expectation > 18 LPA CTC / 16 Base.
"""

def fetch_and_filter_jobs():
    print("Scraping job listings...")
    jobs = scrape_jobs(
        site_name=["linkedin", "indeed", "glassdoor"],
        search_term="Software Engineer C++ Quantitative Developer Quant Researcher AI Engineer ML Infrastructure",
        location="India",
        results_wanted=20,
        hours_old=24,
        country_indeed='India'
    )
    
    if jobs.empty:
        print("No new jobs found in the last 24 hours.")
        return []

    processed_jobs = []
    
    EXCLUDED_COMPANIES = ["tcs", "infosys", "wipro", "cognizant", "accenture", "capgemini", "hcltech", "ltimindtree"]

    for _, row in jobs.iterrows():
        company = str(row['company']).strip()
        title = str(row['title']).strip()
        desc = str(row['description'])
        job_url = str(row['job_url'])

        if any(service_firm in company.lower() for service_firm in EXCLUDED_COMPANIES):
            continue

        prompt = f"""
        Analyze this job for a candidate with this baseline:
        {USER_BASELINE}

        Job Title: {title}
        Company: {company}
        Job Description: {desc[:2000]}

        Output ONLY a raw JSON object with no markdown formatting, no backticks, and no extra text. Format exactly like this:
        {{
            "is_fit": true,
            "estimated_ctc_match": "High",
            "tailored_resume_bullets": ["Bullet 1", "Bullet 2"]
        }}
        """

        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1
            )
            
            output_text = response.choices[0].message.content
            output_text = output_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            
            result = json.loads(output_text)

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

        print("Pausing for 5 seconds to respect API rate limits...")
        time.sleep(5) 

    return processed_jobs

def update_google_sheet(job_list):
    if not job_list:
        return
    
    scope = ["[https://spreadsheets.google.com/feeds](https://spreadsheets.google.com/feeds)", "[https://www.googleapis.com/auth/drive](https://www.googleapis.com/auth/drive)"]
    creds_dict = json.loads(os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open("Job Search Automation Tracker").sheet1

    for job in job_list:
        sheet.append_row([job["Date"], job["Company"], job["Title"], job["Apply Link"], job["Estimated CTC Match"], job["Tailored Highlights"], job["Status"]])
    print(f"Successfully appended {len(job_list)} job(s) to Google Sheets.")

if __name__ == "__main__":
    job_results = fetch_and_filter_jobs()
    update_google_sheet(job_results)