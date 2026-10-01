#!/usr/bin/env bash
conf=""
for url in $CUSTOM_URL; do
    conf+=" --url $url"
done
[[ -n $CUSTOM_TEMPLATE ]] && conf+=" --user $CUSTOM_TEMPLATE"
[[ -n $CUSTOM_PASS ]] && conf+=" --pass $CUSTOM_PASS"
echo "$conf $CUSTOM_USER_CONFIG" > "$CUSTOM_CONFIG_FILENAME"
