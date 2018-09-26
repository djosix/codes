#!/bin/bash

[ "$1" ] && [ "$2" ] || echo '$1 = thread_url, $2 = save_path' && exit
curl -s $1 | grep zoomfile | egrep -o 'https?\:/\/[^"]+' | grep -v thumb | sort -u | wget -i - -P $2
