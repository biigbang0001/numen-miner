#!/usr/bin/env bash
summary=$(curl -s --max-time 2 http://127.0.0.1:4068/summary)
khs=$(jq '.hashrate / 1000' <<< "$summary")
stats=$(jq -c --slurpfile gpu "$GPU_STATS_JSON" '
    def hex: explode | map(if . >= 97 then . - 87 else . - 48 end) | reduce .[] as $digit (0; . * 16 + $digit);
    ($gpu[0] | [.busids, .temp, .fan] | transpose | map({key: .[0], value: {temp: .[1], fan: .[2]}}) | from_entries) as $hive
    | [.devices[] | select(.bus != "")] as $cards
    | {
        hs: [$cards[].hashrate],
        hs_units: "hs",
        temp: [$cards[] | $hive[.bus].temp],
        fan: [$cards[] | $hive[.bus].fan],
        uptime,
        ver: .version,
        ar: [.accepted, .rejected],
        algo: "poscan",
        bus_numbers: [$cards[] | .bus | split(":")[0] | hex]
    }' <<< "$summary")
[[ -z $khs ]] && khs=0
[[ -z $stats ]] && stats=null
