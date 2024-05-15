mkdir -p data/ade
wget -O ./data/ade/ADEChallengeData2016.zip http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ./data/ade/ADEChallengeData2016.zip -d ./data/ade
rm ./data/ade/ADEChallengeData2016.zip
echo "Dataset downloaded."
