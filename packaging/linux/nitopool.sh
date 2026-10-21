#!/usr/bin/env bash

##########################
##    Your settings     ##
##########################

# replace the wallet addresses with your own
WALLET="nu3Numen1pYQqAfigZh4LiK2VF2b1cRBggkKB4gQcLFpBDhGp"
WORKER="rig1"
POOL="stratum+ssl://stratum.nitopool.online:3027"
OPTIONS=""

##########################
## End of your settings ##
##########################

cd "$(dirname "$0")"
exec ./numen-miner -o "$POOL" -u "$WALLET.$WORKER" $OPTIONS "$@"
