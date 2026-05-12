### Contributing

If you want to submit a bug or have a feature request create an issue at https://github.com/Erlemar/pytorch_tempest/issues

Contributing is done using pull requests (direct commits into master branch are disabled).

## To create a pull request:
1. Fork the repository.
2. Clone it.
3. Set up the environment and install pre-commit hooks.

Recommended (with [uv](https://docs.astral.sh/uv/)):
```shell
uv sync                          # installs deps from pyproject.toml + uv.lock
uv run pre-commit install
```

Alternative (pip):
```shell
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pre-commit
pre-commit install
```

4. Make changes to the code.
5. Run tests:

```shell
uv run pytest        # or just `pytest` if your venv is activated
```

6. Push code to your forked repo and create a pull request.

## Managing dependencies

The source of truth is `pyproject.toml`. `uv.lock` is the pinned resolution and is committed.

- Add a dep: `uv add <package>`
- Remove a dep: `uv remove <package>`
- Update lockfile after editing `pyproject.toml` manually: `uv lock`
- Regenerate `requirements.txt` from `pyproject.toml` (for non-uv users): `uv pip compile pyproject.toml -o requirements.txt`
