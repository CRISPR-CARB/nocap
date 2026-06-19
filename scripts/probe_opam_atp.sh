#!/usr/bin/env bash
eval $(/opt/homebrew/bin/opam env)
opam list 2>/dev/null | grep -E "z3|eprover|vampire|cvc" > /tmp/opam_atp_list.txt 2>&1
cat /tmp/opam_atp_list.txt
