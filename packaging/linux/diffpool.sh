#!/usr/bin/env bash

##########################
##    Your settings     ##
##########################

# replace the wallet addresses with your own
WALLET="nu3Numen1pYQqAfigZh4LiK2VF2b1cRBggkKB4gQcLFpBDhGp"
WORKER="rig1"
POOL="http://stratum.diffpool.xyz:8010"
OPTIONS=""

##########################
## End of your settings ##
##########################

cd "$(dirname "$0")"
exec ./numen-miner --url "$POOL/rpc/$WALLET/$WORKER" $OPTIONS "$@"
