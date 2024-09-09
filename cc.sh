#!/bin/bash

cirqs=('rotatedsurface_3_3_ph')
decoders=('naive_bp' 'naive_bposd')

for cirq in "${cirqs[@]}"; do
    for decoder in "${decoders[@]}"; do
        python3 simulate.py -c "$cirq" -dec "$decoder" -s 100
    done
done