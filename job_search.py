import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from jobspy import scrape_jobs
from huggingface_hub import InferenceClient
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# 1. Setup Hugging Face Client
client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct", # Using ungated, fast model to prevent 403 errors
    token=os.getenv("HF_TOKEN")
)

USER_BASELINE = """
Ashutosh Prasad | IIT Patna B.Tech (Math & Computing, CGPA 8.23).
Skills: C++20, Python, Lock-Free Ring Buffers, Concurrency, Linux Kernel Tuning, Multithreading, AWS, FastAPI, Docker ,Machine Learning , Deep Learning,RAG , Lanchain/Langraph, Fine Tuning.
Experience: Teesta Investment (HFT SDE Intern - Low Latency Alerting & Kernel Isolation), Husk Power (SDE Intern).
Achievements: LeetCode Guardian (Top 1.4%), Codeforces Specialist (Max 1542).
Target: Product companies , HFTs , AI/ML roles and ML Infrastructure only. Salary expectation > 18 LPA CTC / 16 Base.
"""

def scrape_telegram_jobs():
    print("Scraping Telegram channels...")
    api_id = int(os.getenv("TELEGRAM_API_ID"))
    api_hash = os.getenv("TELEGRAM_API_HASH")
    session_string = os.getenv("TELEGRAM_STRING_SESSION")
    
    channels = ['https://t.me/jobs_and_internships_updates', 'https://t.me/quantside', 'https://t.me/ml_ds_ai_jobs'] # Replace with your target channels
    keywords = ["C++", "Quant", "SDE", "Backend", "AI Engineer", "ML Infra" ,"AI/ML"]
    telegram_jobs = []
    
    try:
        with TelegramClient(StringSession(session_string), api_id, api_hash) as tg_client:
            for channel in channels:
                try:
                    messages = tg_client.get_messages(channel, limit=50)
                    for msg in messages:
                        if msg.text and msg.date.replace(tzinfo=None) > datetime.now() - timedelta(days=1):
                            if any(kw.lower() in msg.text.lower() for kw in keywords):
                                telegram_jobs.append({
                                    'company': f"Telegram: {channel}",
                                    'title': "Software Engineer (Telegram)",
                                    'description': msg.text[:2000],
                                    'job_url': f"https://t.me/{channel.replace('@', '')}/{msg.id}"
                                })
                except Exception as e:
                    print(f"Error reading channel {channel}: {e}")
    except Exception as e:
        print(f"Failed to authenticate Telegram: {e}")
        
    return pd.DataFrame(telegram_jobs)

def fetch_and_filter_jobs():
    print("Scraping Indeed listings...")
    indeed_jobs = scrape_jobs(
        site_name=["indeed"], # Only using Indeed to bypass 429/403 blocks
        search_term="Software Engineer C++ Quantitative Developer Quant researcher AI Engineer  ML Infrastructure SDE",
        location="India",
        results_wanted=20,
        hours_old=24,
        country_indeed='India'
    )
    
    telegram_df = scrape_telegram_jobs()
    
    # Merge Indeed and Telegram jobs
    if telegram_df.empty and indeed_jobs.empty:
        print("No new jobs found.")
        return []
        
    jobs = pd.concat([indeed_jobs, telegram_df], ignore_index=True)
    processed_jobs = []
    EXCLUDED_COMPANIES = ["tcs", "infosys", "wipro", "cognizant", "accenture", "capgemini", "hcltech", "ltimindtree"]

    for _, row in jobs.iterrows():
        company = str(row.get('company', '')).strip()
        title = str(row.get('title', '')).strip()
        desc = str(row.get('description', ''))
        job_url = str(row.get('job_url', ''))

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

        # Rate limit protection for Hugging Face
        time.sleep(5) 

    return processed_jobs

def update_google_sheet(job_list):
    if not job_list:
        return
    scope = ["[https://spreadsheets.google.com/feeds](https://spreadsheets.google.com/feeds)", "[https://www.googleapis.com/auth/drive](https://www.googleapis.com/auth/drive)"]
    creds_dict = json.loads(os.getenv("GCP_SERVICE_ACCOUNT_JSON"))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    sheet_client = gspread.authorize(creds)
    sheet = sheet_client.open("Job Search Automation Tracker").sheet1

    for job in job_list:
        sheet.append_row([job["Date"], job["Company"], job["Title"], job["Apply Link"], job["Estimated CTC Match"], job["Tailored Highlights"], job["Status"]])
    print(f"Successfully appended {len(job_list)} job(s) to Google Sheets.")

if __name__ == "__main__":
    job_results = fetch_and_filter_jobs()
    update_google_sheet(job_results)