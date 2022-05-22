mkdir -p Datasets
cd /media/dlr/nd/ # Datasets
# you can also download the Apartment.zip manually through
# link: https://caiyun.139.com/m/i?1A5CvuLuaPdhR  password: kL2G
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Apartment.zip
unzip Apartment.zip
ln -s /media/dlr/nd/Apartment /home/dlr/Project/nice-slam/Datasets/Apartment
