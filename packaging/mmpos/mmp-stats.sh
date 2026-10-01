#!/usr/bin/env bash
curl -s --max-time 2 http://127.0.0.1:4068/summary | jq -c '
    def hex: explode | map(if . >= 97 then . - 87 else . - 48 end) | reduce .[] as $digit (0; . * 16 + $digit);
    {
        busid: [.devices[] | if .bus == "" then "cpu" else .bus | split(":")[0] | hex end],
        hash: [.devices[].hashrate],
        units: "hs",
        air: [(.accepted | tostring), "0", (.rejected | tostring)],
        shares: {accepted: [.devices[].accepted], rejected: [.devices[].rejected], invalid: [.devices[] | 0]},
        miner_name: "numen-miner",
        miner_version: .version
    }'
