#!/bin/bash
cp -r /home/ki/workspace /home/ki/github/workspace
cd /home/ki/github/workspace

git add *
git commit -m "daily save"
git push origin master
