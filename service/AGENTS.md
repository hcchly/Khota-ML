# service/ — analysis worker (version 2)

Connects the pipeline to the app's Firebase. Follow the root AGENTS.md section 11 and `docs/APP_INTEGRATION.md`.

- Never commit credentials. The service-account key path comes from the environment variable
  `GOOGLE_APPLICATION_CREDENTIALS`; `service/*.json` keys are ignored by git.
- Write only the fields in `docs/APP_INTEGRATION.md`. Never change `reportStatus`, user data or images.
- Never save, log or upload report images anywhere else. Delete temporary image files after each report.
- Test against a separate test account / test reports before touching real reports.
