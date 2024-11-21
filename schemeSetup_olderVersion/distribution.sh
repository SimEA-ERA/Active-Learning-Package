#!/bin/bash
find data -type f -name "runlist.txt" -exec grep -E -o 'Ag[2-9]_|Ag1[0-9]_|Ag2[0-9]_|Ag3[0-9]_' {} \; | sort | uniq -c | awk '{print $1/6, $2}'
