#!/bin/bash
find src/ -name '*.py' -exec wc -l {} + | tail -1
