# Set up GitHub Actions

1. Unpack the ZIP. Copy its contents into your existing repository root. Keep
   your existing README. Include files and folders beginning with a dot.
2. Commit pyproject.toml, uv.lock, .python-version, .gitattributes,
   .github/workflows/ci.yml, src/, tests/, docs/, and notebooks/. Do not commit
   virtual environments, run outputs, credentials, or checkpoints.
3. Push your dev branch to GitHub.
4. Open **Actions**. If needed, enable Actions in **Settings → Actions → General**
   and allow the official checkout and setup-uv actions under your account policy.
5. Open **CI**. Inspect **Ruff and mypy** and each **CPU tests** job. A green check
   means every command completed successfully.
6. To rerun manually, select **Actions → CI → Run workflow** and choose the branch.
7. Open a pull request into dev or main. Checks run automatically on updates;
   an older run for the same branch is canceled.
8. Optionally require passing checks in **Settings → Branches** before merging.
   Run CI once first so GitHub knows the check names.

## Checks

- Ruff linting, import ordering, and formatting, including notebook code cells.
- mypy for Python modules and tests, with narrow allowances for incompletely
  typed third-party model/trainer boundaries.
- Scripted game transitions and target-leakage checks.
- Mocked HTTP retries, schema failures, 404 handling, and persistent caching.
- Native GRPO loss masking and safe equal-return advantages.
- Real LoRA updates, adapter reload, and interrupted CPU checkpoint recovery.
- Checkpoint checksums and the mounted-directory backup route.

The tiny Transformer is created locally. No pretrained model download or paid
request is part of CI, and no Actions secrets are needed. Test files are temporary;
models are not uploaded as Actions artifacts. GPU and live API checks run separately.

## Subscription

GitHub Free supports this CPU workflow. Standard public-repository runners are
free. Private repositories consume the owner's included allowance; as checked
2026-11-28, Free includes $12 of compute/month and 500 MB of artifact storage.
The three-OS matrix costs more than Linux alone. Keep only ubuntu-latest initially
if desired.

Private-repository branch protection requires GitHub Pro for a personal account
or GitHub Team for an organization. Running CI itself does not require upgrading.

Sources: [Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions),
[branch protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches),
[uv integration](https://docs.astral.sh/uv/guides/integration/github/).
