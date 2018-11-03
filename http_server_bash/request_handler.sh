#!/bin/bash

urldecode() { : "${*//+/ }"; echo -e "${_//%/\\x}"; }

read -r -a line
export REQUEST_METHOD=${line[0]}
export REQUEST_PATH=$(urldecode "${line[@]:1:$((${#line[@]}-2))}")
export REQUEST_VERSION=${line[@]: -1}

echo "[$(date)] $NCAT_REMOTE_ADDR:$NCAT_REMOTE_PORT - $REQUEST_METHOD $REQUEST_PATH $REQUEST_VERSION" >&2

while IFS=$' \t\r\n' read -r -a line; do
    [[ ${#line[@]} -ge 2 ]] || break
    key="$(echo "${line[0]}" | tr '-' '_' | tr a-z A-Z)"
    key=HTTP_${key%:}
    value="$(urldecode "${line[@]:1}")"
    # echo $key $value > /dev/stderr
    export "$key=$value"
done

response="$(./response_generator.sh)"
length=$(echo "$response" | wc -c | tr -d ' ')

echo -en 'HTTP/1.1 200 OK\r\n'
echo -en 'Content-Type: text/html\r\n'
echo -en "Content-Length: $length\r\n"
echo -en 'Server: WTF Server 0.0.0.0.1\r\n'
echo -en '\r\n'
echo "$response"
