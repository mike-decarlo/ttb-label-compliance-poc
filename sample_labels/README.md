Synthetic test labels for this project are generated automatically:

```bash
python scripts/generate_test_labels.py
```

This produces matched clean/messy pairs across five label profiles (each
exercising a specific validation path — exact-match baseline, fuzzy
brand-name matching, import/country-of-origin handling, a bad
warning-statement format, and an ABV tolerance failure), plus a
deliberately corrupt file for testing error handling, and a matching
`applications.json`.

Rerun the script any time to regenerate — it overwrites existing files
in this directory. See `scripts/generate_test_labels.py` for the full
list of profiles.