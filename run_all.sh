eb=0.01
DW=1800
DH=1200
h=0.05
eps=0.01

echo "*******get normal compressed data"

# .././test ../small_data/uf_1_4.dat ../small_data/vf_1_4.dat 1800 1200 0.1 0.1 ../small_data/tracepoints.bin ../small_data/index.bin ../small_data/record.bin ../small_data/position out
../test2 ../small_data/uf_1_4.dat ../small_data/vf_1_4.dat ${DW} ${DH} ${eb} normal  # get .out file

echo "============================================="

echo "******calculate trajectories that need to lossless compress"

# .././test ../small_data/uf_1_4.dat ../small_data/vf_1_4.dat 1800 1200 0.1 0.1 ../small_data/tracepoints.bin ../small_data/index.bin ../small_data/record.bin ../small_data/position out
../test ../small_data/uf_1_4.dat ../small_data/vf_1_4.dat ${DW} ${DH} ${h} ${eps} ../small_data/tracepoints.bin ../small_data/index.bin ../small_data/record.bin ../small_data/position baseline
# use the .out file  and write all index that need to be lossless stored
echo "============================================="



echo "*******lossless trajectories compression"
../test2 ../small_data/uf_1_4.dat ../small_data/vf_1_4.dat ${DW} ${DH} ${eb} lossless_trajectory
# read the index file and reconstruct the trajectory and de/compress it, write as .test file

echo "============================================="
echo "*******testing if trajectories are correct"

../test ../small_data/uf_1_4.dat ../small_data/vf_1_4.dat ${DW} ${DH} ${h} ${eps} ../small_data/tracepoints.bin ../small_data/index.bin ../small_data/record.bin ../small_data/position out

echo "============================================="