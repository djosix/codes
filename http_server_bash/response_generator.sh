#!/bin/bash

base=`realpath .`
path=`realpath .$REQUEST_PATH`
[[ ${REQUEST_PATH: -1} = / && -f $path/index.html ]] && path=$path/index.html
[[ $path = $base* ]] && cat $path || echo '404 not found'
