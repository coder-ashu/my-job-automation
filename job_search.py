import os
import re
import json
import time
import pandas as pd
from datetime import datetime, timedelta, timezone

# Optional local .env loading for testing
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from jobspy import scrape_jobs
from huggingface_hub import InferenceClient
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# 1. Candidate Profile Baseline
USER_BASELINE = """
Ashutosh Prasad | IIT Patna B.Tech (Math & Computing, CGPA 8.23).
Skills: C++20, Python, Lock-Free Ring Buffers, Concurrency, Linux Kernel Tuning, Multithreading, AWS, FastAPI, Docker ,Machine Learning , Deep Learning,RAG , Lanchain/Langraph, Fine Tuning.
Experience: Teesta Investment (HFT SDE Intern - Low Latency Alerting & Kernel Isolation), Husk Power (SDE Intern).
Achievements: LeetCode Guardian (Top 1.4%), Codeforces Specialist (Max 1542).
Target: Product companies , HFTs , AI/ML roles and ML Infrastructure only. Salary expectation > 18 LPA CTC / 16 Base.
"""

def extract_channel_handle(channel_input: str) -> str:
    """Extract clean channel handle from URL or username string."""
    cleaned = channel_input.strip().rstrip('/')
    if '/' in cleaned:
        cleaned = cleaned.split('/')[-1]
    return cleaned.lstrip('@')

def scrape_telegram_jobs():
    print("\n--- Scraping Telegram Channels ---")
    api_id_val = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    session_string = os.getenv("TELEGRAM_STRING_SESSION")
    
    if not (api_id_val and api_hash and session_string):
        print("Warning: Telegram credentials (TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_STRING_SESSION) are missing. Skipping Telegram.")
        return pd.DataFrame()

    try:
        api_id = int(api_id_val)
    except ValueError:
        print(f"Error: TELEGRAM_API_ID '{api_id_val}' is not a valid integer. Skipping Telegram.")
        return pd.DataFrame()

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
    # 72 hours window to capture infrequent postings (weekends/holidays)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=3)

    try:
        with TelegramClient(StringSession(session_string), api_id, api_hash) as tg_client:
            for ch in channels:
                clean_handle = extract_channel_handle(ch)
                print(f"Reading messages from @{clean_handle}...")
                try:
                    messages = tg_client.get_messages(clean_handle, limit=100)
                    channel_match_count = 0
                    for msg in messages:
                        if not msg.text:
                            continue
                        msg_date = msg.date if msg.date.tzinfo else msg.date.replace(tzinfo=timezone.utc)
                        if msg_date > cutoff_date:
                            msg_lower = msg.text.lower()
                            if any(kw in msg_lower for kw in keywords):
                                telegram_jobs.append({
                                    'company': f"Telegram (@{clean_handle})",
                                    'title': "Software / AI Role (Telegram)",
                                    'description': msg.text[:2000],
                                    'job_url': f"https://t.me/{clean_handle}/{msg.id}"
                                })
                                channel_match_count += 1
                    print(f"  -> Found {channel_match_count} relevant recent posts in @{clean_handle}.")
                except Exception as e:
                    print(f"  -> Error reading channel @{clean_handle}: {e}")
    except Exception as e:
        print(f"Failed to authenticate Telegram client: {e}")

    print(f"Total Telegram jobs found: {len(telegram_jobs)}")
    return pd.DataFrame(telegram_jobs)

def scrape_job_boards():
    print("\n--- Scraping Job Boards ---")
    search_queries = [
        "Software Engineer C++",
        "Quantitative Developer",
        "AI Engineer",
        "Machine Learning Engineer",
        "ML Infrastructure",
        "Backend SDE"
    ]
    
    collected_jobs = []

    for query in search_queries:
        print(f"Searching job boards for '{query}'...")
        try:
            df = scrape_jobs(
                site_name=["indeed", "linkedin"],
                search_term=query,
                location="India",
                results_wanted=10,
                hours_old=72,
                country_indeed='india'
            )
            if not df.empty:
                print(f"  -> Found {len(df)} postings for '{query}'.")
                collected_jobs.append(df)
            else:
                print(f"  -> 0 postings found for '{query}'.")
        except Exception as e:
            print(f"  -> Error scraping '{query}': {e}")
        time.sleep(2)

    if not collected_jobs:
        return pd.DataFrame()

    combined_df = pd.concat(collected_jobs, ignore_index=True)
    if 'job_url' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['job_url'])
    print(f"Total unique job board postings scraped: {len(combined_df)}")
    return combined_df

def fetch_and_filter_jobs():
    board_df = scrape_job_boards()
    telegram_df = scrape_telegram_jobs()

    # Combine all scraped sources
    dfs_to_combine = [df for df in [board_df, telegram_df] if not df.empty]
    if not dfs_to_combine:
        print("\nNo new jobs found from any source.")
        return []

    jobs = pd.concat(dfs_to_combine, ignore_index=True)
    if 'job_url' in jobs.columns:
        jobs = jobs.drop_duplicates(subset=['job_url'])

    print(f"\nEvaluating {len(jobs)} total jobs with AI...")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN is not set. Inference calls may be rate-limited or fail.")

    client = InferenceClient(
        model="meta-llama/Llama-3.3-70B-Instruct",
        token=hf_token
    )

    processed_jobs = []
    EXCLUDED_COMPANIES = ["tcs", "infosys", "wipro", "cognizant", "accenture", "capgemini", "hcltech", "ltimindtree"]

    for _, row in jobs.iterrows():
        company = str(row.get('company', '')).strip()
        title = str(row.get('title', '')).strip()
        desc = str(row.get('description', '')).strip()
        job_url = str(row.get('job_url', '')).strip()

        if any(service_firm in company.lower() for service_firm in EXCLUDED_COMPANIES):
            print(f"Skipping excluded service firm: {company}")
            continue

        prompt = f"""
Analyze this job posting for a candidate with this profile baseline:
{USER_BASELINE}

Job Title: {title}
Company: {company}
Job Description:
{desc[:2500]}

Evaluate if the candidate is a strong fit based on skills, target roles, and target salary expectations.
Extract the real Company Name and Job Title from the description if they are generic (e.g. if company is "Telegram").

Output ONLY a raw JSON object with no markdown formatting, no backticks, and no extra text. Format exactly like this:
{{
    "is_fit": true,
    "company": "{company}",
    "title": "{title}",
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

            output_text = response.choices[0].message.content.strip()

            # Robust JSON extraction to avoid JSONDecodeError
            match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if not match:
                print(f"Failed to parse JSON response for '{title}' at '{company}'")
                continue

            result = json.loads(match.group(0))

            if result.get("is_fit", False):
                final_company = result.get("company", company) or company
                final_title = result.get("title", title) or title
                print(f"  [FIT FOUND] {final_title} @ {final_company}")
                processed_jobs.append({
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Company": final_company,
                    "Title": final_title,
                    "Apply Link": job_url,
                    "Estimated CTC Match": result.get("estimated_ctc_match", "N/A"),
                    "Tailored Highlights": "\n".join(result.get("tailored_resume_bullets", [])),
                    "Status": "Pending Application"
                })
            else:
                print(f"  [NOT A FIT] {title} @ {company}")
        except Exception as e:
            print(f"Error evaluating '{title}' at '{company}': {e}")

        # Rate limit protection for Hugging Face
        time.sleep(3)

    return processed_jobs

def update_google_sheet(job_list):
    if not job_list:
        print("\nNo matching jobs to update in Google Sheets.")
        return

    print(f"\n--- Updating Google Sheets with {len(job_list)} job(s) ---")
    gcp_creds_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if not gcp_creds_json:
        print("Warning: GCP_SERVICE_ACCOUNT_JSON is not set. Skipping Google Sheets update.")
        return

    try:
        # Proper OAuth scopes without markdown formatting
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds_dict = json.loads(gcp_creds_json)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        sheet_client = gspread.authorize(creds)
        sheet = sheet_client.open("Job Search Automation Tracker").sheet1

        # Check existing entries to prevent duplicate rows
        try:
            existing_links = set(sheet.col_values(4))  # Column 4 corresponds to Apply Link
        except Exception:
            existing_links = set()

        new_entries = 0
        for job in job_list:
            if job["Apply Link"] in existing_links:
                print(f"Skipping duplicate already in sheet: {job['Title']} @ {job['Company']}")
                continue
            sheet.append_row([
                job["Date"],
                job["Company"],
                job["Title"],
                job["Apply Link"],
                job["Estimated CTC Match"],
                job["Tailored Highlights"],
                job["Status"]
            ])
            existing_links.add(job["Apply Link"])
            new_entries += 1

        print(f"Successfully appended {new_entries} new job(s) to Google Sheets.")
    except gspread.exceptions.SpreadsheetNotFound:
        print("Error: Spreadsheet 'Job Search Automation Tracker' not found. "
              "Please ensure the sheet exists and is shared with your service account email as Editor.")
    except Exception as e:
        print(f"Error updating Google Sheets: {e}")

if __name__ == "__main__":
    job_results = fetch_and_filter_jobs()
    update_google_sheet(job_results)