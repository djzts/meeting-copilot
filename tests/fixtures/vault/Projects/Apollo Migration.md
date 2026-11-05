---
tags: [project, migration]
owner: Dana Lee
---
# Apollo Migration

Apollo moves the billing database from the on-prem Oracle cluster to managed PostgreSQL.

## Timeline

- Phase 1 (schema conversion) finished on 2026-09-15.
- Phase 2 (dual writes) runs through October 2026.
- Cut-over is planned for **2026-11-20**, with a rollback window of 72 hours.

## Risks

1. Stored procedures in PL/SQL need a rewrite; 14 of 38 are done.
2. The nightly reconciliation job is slower on PostgreSQL (45 min vs 20 min).
3. Vendor support for the old cluster ends in January 2027.

```bash
# this is a comment inside a code block, not a heading
psql -c "select 1"
```

## Budget

Total budget is $420k; $310k spent as of October. Remaining budget: **$110k**, mostly for the cut-over weekend and contractor hours.
