# tracecp

## usage

### for 2d: 

./test <u_file> <v_file> <r1> <r2> <step_size> <eps_along_eigenvec> <max_length_rk4> <max_eb> <objective> <num_thread> <traj_output_dir>

eg:

../test ../data/uf.dat  ../data/vf.dat 3600 2400 0.05 0.01 2000 0.05 0 64 ~/data/saved_trajs/

### for 3d

./test_3d <u_file> <v_file> <w_file> <r1> <r2> <r3> <eps_along_eigenvec> <max_length_rk4> <max_eb> <objective> <num_thread> <traj_output_dir>

eg:

../test3d ~/data/nek5000/U.dat ~/data/nek5000/V.dat ~/data/nek5000/W.dat 512 512 512 0.05 0.01 2000 0.01 0 64 ~/data/saved_trajs/

