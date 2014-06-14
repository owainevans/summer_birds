#!/bin/sh

avconv -y -r 4 -b 1800 -i $1%2d.png movie.mp4 && totem movie.mp4
