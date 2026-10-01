#!/usr/bin/env bash

##########################
##    Your settings     ##
##########################

NODE="ws://127.0.0.1:9944"
OPTIONS=""

##########################
## End of your settings ##
##########################

cd "$(dirname "$0")"
exec ./numen-miner --url "$NODE" $OPTIONS "$@"
