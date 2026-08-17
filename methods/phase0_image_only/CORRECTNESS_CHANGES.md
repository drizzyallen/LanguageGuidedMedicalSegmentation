# Correctness-change log

No correctness changes have been made.

Any future fix that changes training behavior must record the affected method,
old and new behavior, evidence that the old behavior was incorrect, review
decision, and replacement run IDs. All three seeds for that method must then be
rerun; corrected and pre-fix runs must never be pooled.
