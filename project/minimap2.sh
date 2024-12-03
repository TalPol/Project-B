#!/bin/bash

curl -L https://github.com/lh3/minimap2/releases/download/v2.28/minimap2-2.28_x64-linux.tar.bz2 | tar -jxvf -
./minimap2-2.28_x64-linux/minimap2
wget -O- https://github.com/attractivechaos/k8/releases/download/v1.2/k8-1.2.tar.bz2 | tar -jxf -
sudo install k8-1.2/k8-x86_64-Linux /usr/local/bin/k8
# hope this works