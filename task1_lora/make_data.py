"""Generate Task 1 data: short plain-English notes -> fixed-schema JSON.

Why templates: labels are produced by code, so they are exactly right (no hand-labelling errors).
Why disjoint templates: train and test use DIFFERENT sentence templates (and test adds some
item/task words never seen in training), so the test score measures generalisation, not memorisation.

Schema of every label:
  {"type": "expense"|"reminder", "amount": number|null, "currency": "INR"|"USD"|null,
   "category": "food"|"travel"|"rent"|"bills"|"shopping"|null, "date": "YYYY-MM-DD"|null, "title": str}
Conventions (so the labels are unambiguous):
  - reminders always have amount, currency and category = null
  - a note with no date phrase gets date = null
  - all dates are resolved against REF_DATE (Saturday 2026-11-21)
  - "next <weekday>" = first such weekday strictly after REF_DATE; "last <weekday>" = most recent strictly before

Usage:  python make_data.py            -> writes data/train.jsonl and data/test.jsonl
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
REF_DATE = date(2026, 11, 21)
N_TRAIN, N_TEST = 600, 60
P_EXPENSE = 0.6          # share of expense examples; the rest are reminders
P_NO_DATE = 0.2          # share of notes with no date phrase (label date = null)
P_UNSEEN_WORD = 0.3      # in test, share of items/tasks drawn from words never seen in training
OUT_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------- templates
# {amt} amount phrase, {item} what was bought, {task} reminder action, {date} date phrase (may be empty)
TRAIN_EXPENSE = [
    "spent {amt} on {item} {date}",
    "paid {amt} for {item} {date}",
    "{item} {amt} {date}",
    "{date} bought {item} for {amt}",
    "add expense: {item}, {amt}, {date}",
    "{amt} on {item} {date}",
    "{date} I spent {amt} on {item}",
    "expense {item} {amt} {date}",
    "got {item} for {amt} {date}",
    "{item} cost me {amt} {date}",
]
TEST_EXPENSE = [
    "had to pay {amt} for {item} {date}",
    "{date} shelled out {amt} on {item}",
    "log {amt} under {item} {date}",
    "my {item} was {amt} {date}",
    "{date} {item} came to {amt}",
]
TRAIN_REMINDER = [
    "remind me to {task} {date}",
    "{date} remind me to {task}",
    "set a reminder to {task} {date}",
    "reminder: {task} {date}",
    "don't let me forget to {task} {date}",
    "add reminder {task} {date}",
    "please remind me to {task} {date}",
]
TEST_REMINDER = [
    "ping me {date} to {task}",
    "note to self: {task} {date}",
    "I need a reminder to {task} {date}",
    "make sure I {task} {date}",
]

# ---------------------------------------------------------------- slot values
ITEMS = {  # category -> items seen in training
    "food": ["groceries", "pizza", "coffee", "lunch", "dinner", "snacks", "vegetables", "milk", "burger", "tea"],
    "travel": ["uber", "cab", "metro card", "train ticket", "bus ticket", "petrol", "flight ticket", "auto"],
    "rent": ["rent", "house rent", "flat rent", "room rent", "pg rent"],
    "bills": ["electricity bill", "water bill", "phone bill", "wifi bill", "gas bill", "internet bill"],
    "shopping": ["shoes", "t-shirt", "jeans", "headphones", "books", "watch", "bag"],
}
UNSEEN_ITEMS = {  # used ONLY in test
    "food": ["biryani", "dosa", "samosas"],
    "travel": ["ola ride", "toll", "parking"],
    "rent": ["apartment rent", "office rent"],
    "bills": ["mobile recharge", "broadband bill", "dth recharge"],
    "shopping": ["kurta", "sunglasses", "backpack"],
}
# INR amount ranges per category (USD amounts are scaled down below)
INR_RANGE = {"food": (50, 1500), "travel": (30, 5000), "rent": (5000, 30000),
             "bills": (200, 3000), "shopping": (300, 5000)}

TASKS = ["call mom", "renew passport", "submit assignment", "pay rent", "water the plants",
         "buy groceries", "book train tickets", "go to the gym", "send the report", "pick up laundry",
         "call the bank", "recharge phone", "visit the doctor", "return library books", "back up laptop"]
UNSEEN_TASKS = ["book dentist appointment", "renew car insurance", "email the professor",
                "pay credit card bill", "wish Rahul happy birthday"]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def ordinal(n):
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def explicit_date_phrase(rng, d, split):
    """Explicit calendar date. Test also uses one format never used in training ('on December 3rd')."""
    fmts = ["on {ord} {mon}", "on {day} {month}"]
    if split == "test":
        fmts.append("on {month} {ord}")
    fmt = rng.choice(fmts)
    return fmt.format(ord=ordinal(d.day), day=d.day, mon=MONTHS[d.month - 1][:3], month=MONTHS[d.month - 1])


def past_date(rng, split):
    """Date phrase for an expense (things already paid). Returns (phrase, date or None)."""
    if rng.random() < P_NO_DATE:
        return "", None
    kind = rng.choice(["today", "yesterday", "day before yesterday", "last", "explicit"])
    if kind == "today":
        return "today", REF_DATE
    if kind == "yesterday":
        return "yesterday", REF_DATE - timedelta(days=1)
    if kind == "day before yesterday":
        return "day before yesterday", REF_DATE - timedelta(days=2)
    if kind == "last":
        wd = rng.randrange(7)
        back = (REF_DATE.weekday() - wd) % 7 or 7   # strictly before REF_DATE
        return f"last {WEEKDAYS[wd]}", REF_DATE - timedelta(days=back)
    d = date(2026, 10, 1) + timedelta(days=rng.randrange((REF_DATE - date(2026, 10, 1)).days + 1))
    return explicit_date_phrase(rng, d, split), d


def future_date(rng, split):
    """Date phrase for a reminder (things to do). Returns (phrase, date or None)."""
    if rng.random() < P_NO_DATE:
        return "", None
    kind = rng.choice(["today", "tomorrow", "day after tomorrow", "next", "in_days", "explicit"])
    if kind == "today":
        return "today", REF_DATE
    if kind == "tomorrow":
        return "tomorrow", REF_DATE + timedelta(days=1)
    if kind == "day after tomorrow":
        return "day after tomorrow", REF_DATE + timedelta(days=2)
    if kind == "next":
        wd = rng.randrange(7)
        ahead = (wd - REF_DATE.weekday()) % 7 or 7   # strictly after REF_DATE
        return f"next {WEEKDAYS[wd]}", REF_DATE + timedelta(days=ahead)
    if kind == "in_days":
        n = rng.randint(2, 10)
        return f"in {n} days", REF_DATE + timedelta(days=n)
    d = REF_DATE + timedelta(days=rng.randint(1, (date(2026, 12, 31) - REF_DATE).days))
    return explicit_date_phrase(rng, d, split), d


def amount_phrase(rng, category):
    """Returns (phrase, numeric value, currency code). Mixes ₹ / Rs / rupees / INR and $ / dollars / USD."""
    if rng.random() < 0.75:
        lo, hi = INR_RANGE[category]
        value = rng.randint(lo // 10, hi // 10) * 10
        num = f"{value:,}" if value >= 1000 and rng.random() < 0.5 else str(value)
        fmt = rng.choice(["₹{n}", "Rs {n}", "Rs. {n}", "{n} rupees", "INR {n}", "{n} rs"])
        return fmt.format(n=num), value, "INR"
    lo, hi = INR_RANGE[category]
    value = rng.randint(max(1, lo // 80), max(2, hi // 80))
    if rng.random() < 0.3:
        value = value + rng.choice([0.5, 0.99, 0.25])
    num = f"{value:.2f}" if isinstance(value, float) else str(value)
    fmt = rng.choice(["${n}", "{n} dollars", "USD {n}"])
    return fmt.format(n=num), value, "USD"


def clean(text):
    """Remove the gaps left by an empty date slot, e.g. 'spent 50 on tea ' or ', ,'."""
    text = " ".join(text.split())
    return text.replace(" ,", ",").strip(" ,")


def make_example(rng, split):
    templates_e, templates_r = (TRAIN_EXPENSE, TRAIN_REMINDER) if split == "train" else (TEST_EXPENSE, TEST_REMINDER)
    use_unseen = split == "test" and rng.random() < P_UNSEEN_WORD
    if rng.random() < P_EXPENSE:
        category = rng.choice(list(ITEMS))
        item = rng.choice((UNSEEN_ITEMS if use_unseen else ITEMS)[category])
        amt, value, currency = amount_phrase(rng, category)
        date_text, d = past_date(rng, split)
        t_idx = rng.randrange(len(templates_e))
        text = templates_e[t_idx].format(amt=amt, item=item, date=date_text)
        label = {"type": "expense", "amount": value, "currency": currency, "category": category,
                 "date": d.isoformat() if d else None, "title": item}
        template_id = f"{split}_expense_{t_idx}"
    else:
        task = rng.choice(UNSEEN_TASKS if use_unseen else TASKS)
        date_text, d = future_date(rng, split)
        t_idx = rng.randrange(len(templates_r))
        text = templates_r[t_idx].format(task=task, date=date_text)
        label = {"type": "reminder", "amount": None, "currency": None, "category": None,
                 "date": d.isoformat() if d else None, "title": task}
        template_id = f"{split}_reminder_{t_idx}"
    return {"input": clean(text), "output": label, "template_id": template_id, "unseen_word": use_unseen}


def generate(rng, split, n, banned_inputs):
    """Draw examples until n unique inputs; skip anything whose input already exists anywhere."""
    rows, seen = [], set(banned_inputs)
    while len(rows) < n:
        ex = make_example(rng, split)
        key = ex["input"].lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(ex)
    return rows


def main():
    # Guard: the template lists and word lists themselves must be disjoint.
    assert not set(TRAIN_EXPENSE) & set(TEST_EXPENSE) and not set(TRAIN_REMINDER) & set(TEST_REMINDER)
    all_seen_items = {i for v in ITEMS.values() for i in v}
    assert not all_seen_items & {i for v in UNSEEN_ITEMS.values() for i in v}
    assert not set(TASKS) & set(UNSEEN_TASKS)

    rng = random.Random(SEED)
    train = generate(rng, "train", N_TRAIN, banned_inputs=set())
    test = generate(rng, "test", N_TEST, banned_inputs={r["input"].lower() for r in train})

    overlap = {r["input"].lower() for r in train} & {r["input"].lower() for r in test}
    assert not overlap, f"train/test share inputs: {overlap}"

    OUT_DIR.mkdir(exist_ok=True)
    for name, rows in [("train", train), ("test", test)]:
        with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for name, rows in [("train", train), ("test", test)]:
        n_exp = sum(r["output"]["type"] == "expense" for r in rows)
        n_nodate = sum(r["output"]["date"] is None for r in rows)
        n_unseen = sum(r["unseen_word"] for r in rows)
        print(f"{name}: {len(rows)} rows | expense {n_exp}, reminder {len(rows) - n_exp} | "
              f"date=null {n_nodate} | unseen item/task words {n_unseen} | "
              f"templates used {len({r['template_id'] for r in rows})}")
    print(f"exact-duplicate inputs between train and test: {len(overlap)}")


if __name__ == "__main__":
    main()
