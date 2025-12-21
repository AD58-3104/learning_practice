#!/bin/bash

# Remove [ and ] characters from the file
sed -i 's/\[//g; s/\]//g' "$1"

sed -i 's/,$//' "$1"