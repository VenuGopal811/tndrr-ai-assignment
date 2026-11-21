# Report

`report.md` -> `report.pdf` (4 pages max). Tables are filled only from results files; all written explanations are Venu's own (`TODO(Venu)` blocks).

```
python report/build_report.py            # refresh AUTO tables in report.md from results files, then write report.pdf
python report/build_report.py --no-pdf   # tables only
```

- Each `<!-- AUTO:name -->` block is regenerated on every run; a missing results file shows "not run". Write prose only outside those blocks.
- PDF = markdown -> HTML -> headless Edge/Chrome print. Set `BROWSER` to the browser path if it isn't found. The script prints the page count and flags anything over 4.
