#!/usr/bin/env python
# coding:utf-8

import os
import sys
import glob
import shutil
import subprocess


GIT_HUB_REPO = 'git@github.com:luosha865/luosha865.github.io.git'

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
    os.system('hugo --theme=hyde --baseUrl="http://luosha865.github.io/')
    with ChDir("public"):
        os.system('git init')
        os.system('git remote add origin git@github.com:luosha865/luosha865.github.io.git')
        os.system('git add -A')
        os.system('git commit -m "commit"')
        os.system('git push -u origin master')

if __name__ == '__main__':
    main()

