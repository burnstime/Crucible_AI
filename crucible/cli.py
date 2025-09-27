import click


@click.group()
def cli():
    """Crucible AI CLI."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
@click.option("--model-tag", default="stable")
def serve(host, port, model_tag):
    """Launch FastAPI server."""
    import uvicorn
    from crucible.api.main import app

    app.state.model_tag = model_tag
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.argument("args", nargs=-1)
def train(args):
    """Run training script with args."""
    import subprocess

    subprocess.run(["python", "-m", "crucible.trainer.train", *args])


@cli.command()
def clean():
    """Run data cleaning pipeline."""
    from crucible.tools.data_cleaning import clean_logs

    clean_logs()


@cli.command()
def test():
    """Run safety tests."""
    import subprocess

    subprocess.run(["pytest", "crucible/tests/test_safety.py"])


@cli.command()
@click.option("--candidate-tag", required=True)
@click.option("--promote", is_flag=True)
def deploy(candidate_tag, promote):
    """Promote candidate to stable."""
    from crucible.tools.storage import promote_candidate

    promote_candidate(candidate_tag, promote)


if __name__ == "__main__":
    cli()
