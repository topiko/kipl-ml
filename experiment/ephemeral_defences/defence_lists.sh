#!/bin/bash
#

ephemeral_bottle=("ephemeral-pad-bottle-sc0.5" "ephemeral-block-bottle-sc0.75")
ephemeral_inf=("ephemeral-pad-inf-sc0.75" "ephemeral-block-inf-sc0.75")
other_bottle=("regulator-bottleneck" "front-bottleneck" "interspace" "breakpad" "tamaraw-bottleneck")
other_inf=("regulator-infinite" "front-infinite" "interspace" "breakpad" "tamaraw-infinite")


defences_bottle=("${other_bottle[@]}" "${ephemeral_bottle[@]}")
defences_inf=("${other_inf[@]}" "${ephemeral_inf[@]}")
