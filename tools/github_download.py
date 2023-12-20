from argparse import ArgumentParser
from pathlib import Path

import requests
from github import ContentFile, Github, Repository
from loguru import logger


class GithubDownloader:
    def __init__(self, repo: Repository) -> None:
        self.github_client = Github()
        self.repo = self.github_client.get_repo(repo)

    def download(self, content: ContentFile, out: str):
        r = requests.get(content.download_url)
        output_path = Path(out) / content.path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            logger.info(f"Downloading {content.path} to {output_path}")
            f.write(r.content)

    def download_folder(self, folder: str, out: str, recursive: bool = True):
        contents = self.repo.get_contents(folder)
        for content in contents:
            if content.download_url is None:
                if recursive:
                    self.download_folder(content.path, out, recursive)
                continue
            self.download(content, out)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--repo", help="The repo where the file or folder is stored")
    parser.add_argument("--path", help="The folder or file you want to download")
    parser.add_argument(
        "-o",
        "--out",
        default="./",
        required=False,
        help="Path to folder you want to download "
        "to. Default is current folder + "
        "'downloads'",
    )
    parser.add_argument(
        "-f",
        "--file",
        action="store_true",
        help="Set flag to download a single file, instead of a " "folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    github_downloader = GithubDownloader(repo=args.repo)
    github_downloader.download_folder(args.path, args.out)
