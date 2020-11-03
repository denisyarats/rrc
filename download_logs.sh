#!/bin/bash

# Download all logs from the specified job


# expects output directory (to save downloaded files) as argument
if (( $# != 1 ))
then
    echo "Invalid number of arguments."
    echo "Usage:  $0 <job_id>"
    exit 1
fi

job_id=$1
output_directory="robot_logs"

if ! [ -d "${output_directory}" ]
then
    echo "${output_directory} does not exist or is not a directory"
    exit 1
fi

# prompt for username and password (to avoid having user credentials in the
# bash history)
username="hushedtomatoe"
password="BuntApDaz5"



# URL to the webserver at which the recorded data can be downloaded
base_url=https://robots.real-robot-challenge.com/output/${username}/data


# Check if the file/path given as argument exists in the "data" directory of
# the user
function curl_check_if_exists()
{
    local filename=$1

    http_code=$(curl -sI --user ${username}:${password} -o /dev/null -w "%{http_code}" ${base_url}/${filename})

    case "$http_code" in
        200|301) return 0;;
        *) return 1;;
    esac
}


echo "Check ${job_id}"
if curl_check_if_exists ${job_id}
then
    # create directory for this job
    job_dir="${output_directory}/${job_id}"
    mkdir "${job_dir}"

    echo "Download data to ${job_dir}"

    # Download data.  Here only the report file is downloaded as example.  Add
    # equivalent commands for other files as needed.
    for file in report.json info.json robot_data.dat camera_data.dat camera60.yml camera180.yml camera300.yml
    do
        curl --user ${username}:${password} -o "${job_dir}/${file}" ${base_url}/${job_id}/${file}
    done
else
    echo "No data for ${job_id} found."
    exit 1
fi

echo "Copy goal and policy"
cp rrc_package/goal.json ${base_url}/${job_id}
cp rrc_package/scripts/policy_0.pt ${base_url}/${job_id}/