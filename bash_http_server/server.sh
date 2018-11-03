#!/bin/bash

echo Serving...
ncat -c './request_handler.sh' -lk 8080
