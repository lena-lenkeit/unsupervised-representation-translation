import os.path
import tarfile

if __name__ == "__main__":
    output_filename = "repository.tar.gz"

    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add("src")
        tar.add("train-arae.py")
        tar.add("eval-arae.py")
        tar.add("pyproject.toml")
