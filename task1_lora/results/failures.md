# Task 1: failure cases (fine-tuned model)

52/60 exact matches. All 8 misses are in `predictions.jsonl`; these are the 3 discussed in the report.

## Test #33

- **Input:** yesterday shelled out 1220 rupees on tea
- **Expected:** `{"type": "expense", "amount": 1220, "currency": "INR", "category": "food", "date": "2026-11-20", "title": "tea"}`
- **Got:** `{"type": "expense", "amount": 1220, "currency": "INR", "category": "food", "date": "2026-11-20", "title": "shelled out"}`
- **Why:** "shelled out" only appears in a test template. In training the item usually comes right after the verb or date, so the model copied the unfamiliar verb phrase as the title. Same mistake on #56.

## Test #37

- **Input:** note to self: return library books in 10 days
- **Expected:** `{"type": "reminder", "amount": null, "currency": null, "category": null, "date": "2026-12-01", "title": "return library books"}`
- **Got:** `{"type": "reminder", "amount": null, "currency": null, "category": null, "date": "2026-11-22", "title": "return library books"}`
- **Why:** "in N days" exists in training, but adding N days to a date is arithmetic, which a 0.5B model does unreliably, especially across a month boundary. It output tomorrow's date.

## Test #13

- **Input:** had to pay 1790 rupees for mobile recharge last Saturday
- **Expected:** `{"type": "expense", "amount": 1790, "currency": "INR", "category": "bills", "date": "2026-11-14", "title": "mobile recharge"}`
- **Got:** `{"type": "expense", "amount": 1790, "currency": "INR", "category": "shopping", "date": "2026-11-14", "title": "mobile recharge"}`
- **Why:** "mobile recharge" is a held-out item. Every training bills item ends in "bill" and "recharge" only appeared as a reminder task, so the model fell back to shopping.
