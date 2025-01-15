#!/usr/bin/env sh

set -o errexit
set -o nounset

pyclean () {
  find . | grep -E '(__pycache__|\.py[cod]$)' | xargs rm -rf
}

run_test () {
  pytest "$@"
}

pyclean

trap pyclean EXIT INT TERM

run_test "$@"
