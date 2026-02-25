#!/bin/bash
# Bash script for rewriting .conf file for parameter optimalization
WORKDIR="$1"
shift
rm -rf "$WORKDIR"
cp -a drutes_temp "$WORKDIR"
cd "$WORKDIR"

###########################################
# READ PARAMETERS
###########################################

# evap module
b1_org=$1
b2_org=$2
b3_org=$3
b1_min=$4
b2_min=$5
b3_min=$6

albedo=$7

# water module
alpha_org=$8
n_org=$9
m_org=$10
K_org=${11}

alpha_min=${12}
n_min=${13}
m_min=${14}
K_min=${15}

S_max=${16}

###########################################
# CREATE evap.conf
###########################################
sed \
    -e "s/!b1_org/$b1_org/g" \
    -e "s/!b2_org/$b2_org/g" \
    -e "s/!b3_org/$b3_org/g" \
    -e "s/!b1_min/$b1_min/g" \
    -e "s/!b2_min/$b2_min/g" \
    -e "s/!b3_min/$b3_min/g" \
    drutes.conf/evaporation/evap.conf.temp > drutes.conf/evaporation/evap.conf

###########################################
# CREATE albedo.dat
###########################################
sed \ 
	-e "s/!albedo/$albedo/g" \
	drutes.conf/evaporation/albedo.dat.temp > drutes.conf/evaporation/albedo.dat

###########################################
# CREATE water.conf
###########################################
sed \
    -e "s/!alpha_org/$alpha_org/g" \
    -e "s/!n_org/$n_org/g" \
    -e "s/!m_org/$m_org/g" \
    -e "s/!K_org/$K_org/g" \
    -e "s/!alpha_min/$alpha_min/g" \
    -e "s/!n_min/$n_min/g" \
    -e "s/!m_min/$m_min/g" \
    -e "s/!K_min/$K_min/g" \
    drutes.conf/water.conf/matrix.conf.temp > drutes.conf/water.conf/matrix.conf

###########################################
# CREATE root4uptake.conf
###########################################
sed \ 
	-e "s/!S_max/$S_max/g" \
    drutes.conf/water.conf/root4uptake.conf.temp > drutes.conf/water.conf/root4uptake.cong

###########################################
# RUN DRUTES
###########################################
bin/drutes

# run drutes simulation and discard the terminal output
#bin/drutes > /dev/null


