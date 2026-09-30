# DataForge — AI-powered local data science tool
# Copyright (C) 2026 Mohammed Farazuddin <farazfiyaz2@gmail.com>
# License: AGPL-3.0 — see LICENSE
"""
Questions asked with no file attached.

General questions ("what is a p-value?") get answered normally. Questions that
need the user's data ("what's the churn rate?") get a clear "attach a file"
reply instead of an invented answer. Telling those apart needs understanding,
so the model decides: it's told to reply with a marker when the data is needed.
"""

from datetime import date

MARKER = "NEEDS_FILE"

RULE = (
    "No dataset or file is loaded in this chat. Answer general questions (statistics, "
    "data science, Python and its libraries such as pandas, software versions, facts, "
    "definitions, or anything else) directly and concisely. Only if the request is about "
    "the user's OWN data (for example \"what's the churn rate in my data\", \"plot the age "
    "column\", \"summarize my dataset\") must you not invent data or answer hypothetically: "
    f"then reply with exactly {MARKER} and nothing else."
)

# Added when web search is allowed (local model with the web_search tool)
WEB_HINT = (
    "You have a web_search tool. Use it for anything current or that you're unsure of "
    "(latest versions, recent events, prices, documentation) instead of guessing, then "
    "answer and name the sites you used."
)

GENERAL_SYSTEM_PROMPT = "You are DataForge, an AI data-science assistant.\n\n" + RULE


def general_system_prompt(web_search: bool) -> str:
    if not web_search:
        return GENERAL_SYSTEM_PROMPT
    # Models assume their training year; without today's date, "the most recent
    # World Cup" became a search for 2022
    today = date.today().strftime("%d %B %Y")
    return f"{GENERAL_SYSTEM_PROMPT}\n\n{WEB_HINT} Today's date is {today}; put the year in searches about recent things."

NEEDS_FILE = ("There's no file attached, and this question needs your data. "
              "Attach a CSV or Excel file and ask again.")
NEEDS_FILE_STALE = ("The file for this chat isn't loaded any more (DataForge was restarted since "
                    "it was attached). Attach it again to keep working with it.")


def needs_file(reply: str) -> bool:
    """Did the model say the question needs the user's data?"""
    return MARKER in (reply or "")
