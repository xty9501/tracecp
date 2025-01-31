# TspSZ: An Efficient Parallel Error-Bounded Lossy Compressor for Topological Skeleton Preservation

## Dependencies
- cmake >= 3.19.4
- gcc >= 9.3
- zstd >= 1.5.0
- ftk >= 0.0.6
- eigen >= 3.4.0
- openmp >=4.0

You may need to modify the paths in CMakeLists.txt for these dependencies.

## Installation
```bash
cd tracecp
mkdir build
cd build
cmake ..
make -j8
```

## Dataset
In our work, we utilized four datasets: ocean (2d),CBA(2d), hurricane, and nek5000

Datasets available online:
- [Ocean](https://github.com/szcompressor/cpSZ/tree/main/data) (uf.dat and vf.dat)
- [CBA](https://cgl.ethz.ch/research/visualization/data.php) (Heated Cylinder with Boussinesq Approximation)
- [Hurricane](https://sdrbench.github.io/) (Uf48.bin.f32,Vf48.bin.f32,Wf48.bin.f32)
- [Nek5000](https://drive.google.com/drive/folders/1JDYp4mLebE0s0EZ2UFWJYBtxdq5km7Rz?usp=sharing) (U.dat,V.dat,W.dat)


## usage

### for 2d: 

./test <u_file> <v_file> <r1> <r2> <step_size> <eps_along_eigenvec> <max_length_rk4> <max_eb> <eb_mode> <num_thread> <threadshold1> <threadshold2> <threadshold3> <next_index_coeff(optional,default=1)> <traj_out_dir(optional)>

eg:

~/tracecp/test ~/data/2d/uf.dat ~/data/2d/vf.dat 3600 2400 0.025 0.01 1000 0.05 abs 100 1.414 5 10

### for 3d

./test_3d <u_file> <v_file> <w_file> <r1> <r2> <r3> <eps_along_eigenvec> <max_length_rk4> <max_eb> <eb_mode> <num_thread> <threadshold1> <threadshold2> <threadshold3> <next_index_coeff(optional,default=1)> <traj_out_dir(optional)>

eg:

~/tracecp/test3d ~/data/nek5000/U.dat ~/data/nek5000/V.dat ~/data/nek5000/W.dat 512 512 512 0.025 0.01 1000 0.05 abs 125 1.414 5 10

<threadshold2> <threadshold3> are placeholder.
if <traj_output_dir> is not specified， then the trajs file will not be wriiten.

To enable/disable OMP for cpSZ, set CPSZ_OMP_FLAG to 0 in main.cpp and main3d.cpp.

Use the SoS method to extract critical points by setting SOS_FLAG to 1.

