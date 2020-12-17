### Contributing

If you want to submit a bug or have a feature request create an issue at https://github.com/Erlemar/pytorch_tempest/issues

Contributing is done using pull requests (direct commits into master branch are disabled).

## To create a pull request:
1. Fork the repository.
2. Clone it.
3. Install pre-commit hook, initialize it and install requirements:

```shell
pip install pre-commit
pip install -r requirements.txt
pre-commit install
```

4. Make  changes to the code.
5. Run tests:

```shell
pytest
```

6. Push code to your forked repo and create a pull request
