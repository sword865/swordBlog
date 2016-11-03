#!/usr/bin/env python
# coding:utf-8

import os
import shutil

def main():
	os.system('hugo server --buildDrafts --watch')
	#'hugo server -w -t rapid')


if __name__ == '__main__':
    main()
