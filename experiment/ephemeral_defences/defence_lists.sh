#!/bin/bash
#

ephemeral_bottle=("ephemeral-pad-bottle-sc0.5" "ephemeral-block-bottle-sc0.75")
ephemeral_inf=("ephemeral-pad-inf-sc0.75" "ephemeral-block-inf-sc0.75")
other_bottle=("front-bottleneck" "interspace" "breakpad" "requlator-bottleneck" "tamaraw-bottleneck")
other_inf=("front-infinite" "interspace" "breakpad" "requlator-infinite" "tamaraw-infinite")


defences_bottle=("${ephemeral_bottle[@]}" "${other_bottle[@]}")
defences_inf=("${ephemeral_inf[@]}" "${other_inf[@]}")
