# 10013 Linux deployment

The deployment runs two restartable Docker services:

- `factorfactory-10013-postgres`: PostgreSQL 16, host-only port `127.0.0.1:55433`.
- `factorfactory-10013-app`: full-LLM FactorFactory, LAN port `10013`.

Runtime secrets are generated on the target host in `.postgres.env` and
`.runtime.env` with mode `0600`; neither file belongs in source control.
Market panels are mounted read-only from `../../data`. Full-panel evaluation
and first-load expansion are each limited to one concurrent slot for the
7 GiB host.

From the project root on the target host:

```bash
docker compose -f deploy/10013-linux/compose.yml ps
docker compose -f deploy/10013-linux/compose.yml logs -f app
docker compose -f deploy/10013-linux/compose.yml restart app
docker compose -f deploy/10013-linux/compose.yml stop app
docker compose -f deploy/10013-linux/compose.yml start app
```

The HTTP control plane has no authentication. Expose port 10013 only on a
trusted isolated LAN.
