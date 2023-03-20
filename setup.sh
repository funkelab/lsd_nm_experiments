#!/bin/bash

wget -O lsd_legacy_packages https://www.dropbox.com/sh/gtc8zj08qoqq286/AAAy0LiA2LBAo1gsq7rBD3oDa\?dl\=0
unzip lsd_legacy_packages
rm lsd_legacy_packages

make
