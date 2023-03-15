from pathlib import Path
from setuptools import find_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# Define our package
setup(
    name="Textile Classification",
    version=0.1,
    description="Classification of clothes into valid or invalid group using deep learning",
    author="Sajad Rahmatian",
    author_email="sajad.rhmtn@gmail.com",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=required_packages,
)
