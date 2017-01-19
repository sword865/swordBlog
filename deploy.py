#!/usr/bin/env python
# coding:utf-8

import os

GIT_HUB_REPO = 'git@github.com:sword865/sword865.github.io.git'


class ChDir:
    """Context manager for changing the current working directory"""

    def __init__(self, new_path):
        self.newPath = os.path.expanduser(new_path)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, exception_type, exception_value, traceback):
        os.chdir(self.savedPath)


def main():
    os.system('hugo --baseUrl="http://blog.sword865.com/"')
    with ChDir("public"):
        os.system('git add --all')
        os.system('git commit -m "commit"')
        os.system('git push -u origin master')


if __name__ == '__main__':
    main()
