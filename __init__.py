from subprocess import check_call
from sys import executable
from pathlib import Path

requirements = Path(__file__).resolve().parent.joinpath("requirements.txt")

check_call([executable, "-m", "pip", "install", "--quiet", "-r",
           str(requirements)])  # TODO make pip quiet
