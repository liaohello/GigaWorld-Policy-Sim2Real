# Repository instructions

- Keep this repository focused on GigaWorld-Policy research, reproducibility, data/action contracts, simulation evaluation, and later real-robot adaptation.
- Read the relevant upstream README and license before integrating external code. Record exact commit/version and local patches. Keep upstream repositories outside tracked Git content.
- Store checkpoints, datasets, videos, caches, and environments outside the repository. Use environment variables for local paths; commit only sanitized examples.
- Never commit passwords, tokens, private keys, public IPs, SSH commands, personal paths, or machine-specific identifiers. Review the staged diff before every commit.
- Keep simulation and policy dependencies isolated. Do not alter a shared base environment for this project.
- Mark evidence accurately as measured, documented, inferred, or not yet verified. A process starting is not an inference or simulation pass.
- For RoboTwin native validation, preserve the 14-D Aloha-AgileX action contract. Label expert replay and policy control distinctly; never use a substitute action stream to claim policy success.
- Keep the workbench platform separate; connect through explicit observation, policy, and action interfaces.
