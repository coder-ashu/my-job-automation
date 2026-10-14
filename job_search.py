import os
import json
import re
import time
import pandas as pd
from datetime import datetime, timedelta
from jobspy import scrape_jobs
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
import gspread
from oauth2client.service_account import ServiceAccountCredentials
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

def parse_json_from_llm(output_text):
    """
    Robust JSON parser that handles:
    - Thinking/reasoning tags (<think>...</think>, <thought>...</thought>)
    - Markdown code fences (```json ... ```)
    - Preamble/conversational text before or after the JSON block
    - Trailing commas or minor JSON quirks
    """
    if not output_text:
        return None

    # 1. Strip thinking/reasoning tags
    text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL)
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
    text = text.strip()

    # 2. Check for markdown code fence
    fence_matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, flags=re.DOTALL)
    for match in fence_matches:
        cand = match.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "is_fit" in obj:
                return obj
        except Exception:
            cleaned = re.sub(r',\s*([\]}])', r'\1', cand)
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, dict) and "is_fit" in obj:
                    return obj
            except Exception:
                continue

    # 3. Use JSONDecoder to find a dictionary containing "is_fit"
    decoder = json.JSONDecoder()
    for i in range(len(text)):
        if text[i] == '{':
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict) and "is_fit" in obj:
                    return obj
            except Exception:
                pass

    # 4. Fallback: widest outer braces with trailing comma cleaning
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        raw_json = text[start_idx:end_idx + 1]
        try:
            obj = json.loads(raw_json)
            if isinstance(obj, dict):
                return obj
        except Exception:
            cleaned = re.sub(r',\s*([\]}])', r'\1', raw_json)
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    return None


def scrape_telegram_jobs():
    print("Scraping Telegram channels...")
    api_id_val = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    session_string = os.getenv("TELEGRAM_STRING_SESSION")

    if not api_id_val or not api_hash or not session_string:
        print("Telegram credentials not fully configured. Skipping Telegram scraping.")
        return pd.DataFrame()

    try:
        api_id = int(api_id_val)
    except ValueError:
        print("Invalid TELEGRAM_API_ID format. Skipping Telegram scraping.")
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
        print(telegram_jobs)
    except Exception as e:
        print(f"Failed to authenticate Telegram: {e}")

    return pd.DataFrame(telegram_jobs)


def fetch_and_filter_jobs():
    print("Scraping global Indeed listings...")

    indeed_dfs = []
    indeed_queries = [
        {"location": "", "is_remote": True, "results_wanted": 15, "country_indeed": "USA"},
        {"location": "United States", "is_remote": False, "results_wanted": 10, "country_indeed": "USA"},
        {"location": "Singapore", "is_remote": False, "results_wanted": 10, "country_indeed": "Singapore"},
    ]

    for q in indeed_queries:
        try:
            df = scrape_jobs(
                site_name=["indeed"],
                search_term="Software Engineer C++ Quantitative Developer",
                hours_old=24,
                **q
            )
            if df is not None and not df.empty:
                indeed_dfs.append(df)
        except Exception as e:
            print(f"Error scraping Indeed ({q.get('location') or 'Remote'}): {e}")

    telegram_df = scrape_telegram_jobs()
    all_dfs = indeed_dfs + ([telegram_df] if not telegram_df.empty else [])

    if not all_dfs:
        print("No job sources returned any results.")
        return []

    jobs = pd.concat(all_dfs, ignore_index=True)

    if jobs.empty:
        print("No new jobs found.")
        return []

    # Deduplicate by job_url if present, then by company and title
    if 'job_url' in jobs.columns:
        jobs = jobs.drop_duplicates(subset=['job_url'])
    jobs = jobs.drop_duplicates(subset=['company', 'title'])

    print(f"Total unique jobs to process: {len(jobs)}")

    processed_jobs = []
    EXCLUDED_COMPANIES = ["tcs", "infosys", "wipro", "cognizant", "accenture", "capgemini", "hcltech", "ltimindtree"]

    # Free models currently active on OpenRouter (in priority order with smart router fallback)
    MODELS_TO_TRY = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "openrouter/free"
    ]

    for _, row in jobs.iterrows():
        company = str(row.get('company', '')).strip()
        title = str(row.get('title', '')).strip()
        desc = str(row.get('description', ''))
        job_url = str(row.get('job_url', ''))

        if any(service_firm in company.lower() for service_firm in EXCLUDED_COMPANIES):
            print(f"Skipping excluded company: {company}")
            continue

        prompt = f"""Analyze this job for a candidate with this baseline:
{USER_BASELINE}

Job Title: {title}
Company: {company}
Job Description:
{desc[:2500]}

Respond ONLY with a raw JSON object formatted exactly like this:
{{
    "is_fit": true,
    "estimated_ctc_match": "High",
    "tailored_resume_bullets": ["Bullet 1", "Bullet 2"]
}}"""

        result = None
        for model_name in MODELS_TO_TRY:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an automated technical recruiter and job match screener. "
                                "Evaluate the job strictly against the candidate profile. "
                                "Output ONLY a valid JSON object matching the requested schema. "
                                "Do NOT include reasoning, preambles, or markdown formatting."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500
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
                result = parse_json_from_llm(output_text)
                if result:
                    print(f"[{model_name}] Evaluated '{title}' at '{company}': is_fit={result.get('is_fit')}")
                    break
                else:
                    print(f"[{model_name}] No valid JSON found in output: {output_text[:120]}...")
            except Exception as e:
                print(f"Error calling {model_name} for {title} at {company}: {e}")
                time.sleep(1)

        if result and result.get("is_fit", False):
            print(f"MATCH FOUND: '{title}' at '{company}'")
            processed_jobs.append({
                "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "Company": company,
                "Title": title,
                "Apply Link": job_url,
                "Estimated CTC Match": result.get("estimated_ctc_match", "N/A"),
                "Tailored Highlights": "\n".join(result.get("tailored_resume_bullets", [])),
                "Status": "Pending Application"
            })
        elif not result:
            print(f"Failed to evaluate '{title}' at '{company}' across all models.")

        # Rate limit protection between jobs
        time.sleep(2)

    return processed_jobs


def update_google_sheet(job_list):
    if not job_list:
        print("No matching jobs to append to Google Sheets.")
        return

    creds_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        print("GCP_SERVICE_ACCOUNT_JSON secret not set. Skipping sheet update.")
        return

    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = json.loads(creds_json)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        sheet_client = gspread.authorize(creds)
        sheet = sheet_client.open("Job Search Automation Tracker").sheet1

        for job in job_list:
            sheet.append_row([
                job["Date"],
                job["Company"],
                job["Title"],
                job["Apply Link"],
                job["Estimated CTC Match"],
                job["Tailored Highlights"],
                job["Status"]
            ])
        print(f"Successfully appended {len(job_list)} job(s) to Google Sheets.")
    except Exception as e:
        print(f"Error updating Google Sheets: {e}")


if __name__ == "__main__":
    job_results = fetch_and_filter_jobs()
    update_google_sheet(job_results)