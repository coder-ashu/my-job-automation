import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from jobspy import scrape_jobs
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# NEW: Import standard OpenAI client
from openai import OpenAI

# 1. Setup OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
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
    
    channels = [
        'https://t.me/jobs_and_internships_updates',
        'https://t.me/quantside',
        'https://t.me/ml_ds_ai_jobs'
    ]
    keywords = [
        "c++", "cpp", "quant", "sde", "backend", "software engineer",
        "developer", "ai engineer", "ml engineer", "machine learning",
        "deep learning", "ml infra", "ai/ml", "ai / ml", "llm", "hft",
        "low latency", "python"
    ]
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
    print("Scraping global Indeed listings...")
    
    remote_jobs = scrape_jobs(site_name=["indeed"], search_term="Software Engineer C++ Quantitative Developer", location="", is_remote=True, results_wanted=15, hours_old=24, country_indeed='USA')
    us_jobs = scrape_jobs(site_name=["indeed"], search_term="Software Engineer C++ Quantitative Developer", location="United States", results_wanted=10, hours_old=24, country_indeed='USA')
    sg_jobs = scrape_jobs(site_name=["indeed"], search_term="Software Engineer C++ Quantitative Developer", location="Singapore", results_wanted=10, hours_old=24, country_indeed='Singapore')
    
    telegram_df = scrape_telegram_jobs()
    jobs = pd.concat([remote_jobs, us_jobs, sg_jobs, telegram_df], ignore_index=True)
    
    if jobs.empty:
        print("No new jobs found.")
        return []
        
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

        MODELS_TO_TRY = [
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-exp:free"
        ]

        result = None
        for model_name in MODELS_TO_TRY:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=800
                )
                
                if not response or not getattr(response, "choices", None):
                    err = getattr(response, "error", None) or (response.model_extra.get("error") if hasattr(response, "model_extra") and response.model_extra else None)
                    print(f"[{model_name}] OpenRouter returned no choices: {err}")
                    continue

                choice = response.choices[0]
                if not choice.message or not choice.message.content:
                    print(f"[{model_name}] Empty message content received.")
                    continue

                output_text = choice.message.content.strip()
                start_idx = output_text.find("{")
                end_idx = output_text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    result = json.loads(output_text[start_idx:end_idx + 1])
                    break
                else:
                    print(f"[{model_name}] No JSON block found in output: {output_text[:100]}")
            except Exception as e:
                print(f"Error calling {model_name} for {title} at {company}: {e}")

        if result and result.get("is_fit", False):
            processed_jobs.append({
                "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "Company": company,
                "Title": title,
                "Apply Link": job_url,
                "Estimated CTC Match": result.get("estimated_ctc_match", "N/A"),
                "Tailored Highlights": "\n".join(result.get("tailored_resume_bullets", [])),
                "Status": "Pending Application"
            })

        # Rate limit protection
        time.sleep(3) 

    return processed_jobs

def update_google_sheet(job_list):
    if not job_list:
        return
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
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