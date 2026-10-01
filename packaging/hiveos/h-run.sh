#!/usr/bin/env bash
. h-manifest.conf
mkdir -p "${CUSTOM_LOG_BASENAME%/*}"
./numen-miner $(< "$CUSTOM_CONFIG_FILENAME") 2>&1 | tee --append "$CUSTOM_LOG_BASENAME.log"
