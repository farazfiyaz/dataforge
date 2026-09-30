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

MARKER = "NEEDS_FILE"

RULE = (
    "No dataset or file is loaded in this chat. Answer general questions (statistics, "
    "data science, Python, concepts, or anything else) directly and concisely. If the "
    "request needs the user's own data or file (for example it asks about their data, a "
    "dataset, specific columns, rows or values, or asks you to analyse, clean, model or plot "
    f"their data), do not invent data or answer hypothetically: reply with exactly {MARKER} "
    "and nothing else."
)

GENERAL_SYSTEM_PROMPT = "You are DataForge, an AI data-science assistant.\n\n" + RULE

NEEDS_FILE = ("There's no file attached, and this question needs your data. "
              "Attach a CSV or Excel file and ask again.")
NEEDS_FILE_STALE = ("The file for this chat isn't loaded any more (DataForge was restarted since "
                    "it was attached). Attach it again to keep working with it.")


def needs_file(reply: str) -> bool:
    """Did the model say the question needs the user's data?"""
    return MARKER in (reply or "")
