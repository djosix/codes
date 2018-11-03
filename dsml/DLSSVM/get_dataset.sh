#!/bin/bash

# Available datasets:
# Basketball, Biker, Bird1, BlurBody, BlurCar2, BlurFace, BlurOwl,
# Bolt, Box, Car1, Car4, CarDark, CarScale, ClifBar, Couple, Crowds,
# David, Deer, Diving, DragonBaby, Dudek, Football, Freeman4, Girl,
# Human3, Human4, Human6, Human9, Ironman, Jump, Jumping, Liquor,
# Matrix, MotorRolling, Panda, RedTeam, Shaking, Singer2, Skating1,
# Skating2, Skiing, Soccer, Surfer, Sylvester, Tiger2, Trellis,
# Walking, Walking2, Woman, Bird2, BlurCar1, BlurCar3, BlurCar4,
# Board, Bolt2, Boy, Car2, Car24, Coke, Coupon, Crossing, Dancer,
# Dancer2, David2, David3, Dog, Dog1, Doll, FaceOcc1, FaceOcc2, Fish,
# FleetFace, Football1, Freeman1, Freeman3, Girl2, Gym, Human2, Human5,
# Human7, Human8, Jogging [1,2], KiteSurf, Lemming, Man, Mhyang, MountainBike,
# Rubik, Singer1, Skater, Skater2, Subway, Suv, Tiger1, Toy, Trans, Twinnings,
# Vase, ts

# Reference:
# http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

dataset=Basketball # default
[ "$1" ] && dataset=$1

if [ -f "datasets/$dataset.zip" ]; then
    echo "Dataset $dataset is already downloaded"
    exit
fi

echo "Downloading $dataset"
mkdir "datasets"
cd "datasets" || mkdir "datasets" && cd "datasets"
wget "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/$dataset.zip"
unzip "$dataset.zip"
