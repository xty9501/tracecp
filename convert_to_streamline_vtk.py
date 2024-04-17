import numpy as np
import vtkmodules.all as vtk
def write_traj_to_vtk(dataset, input):
    coordinates = np.fromfile(f'/Users/mingzexia/Documents/Github/tracecp/{dataset}/tracepoints.bin{input}', dtype=np.float64).reshape(-1, 2)
    index_array = np.fromfile(f'/Users/mingzexia/Documents/Github/tracecp/{dataset}/index.bin{input}', dtype=np.int32)

    # Create a vtkPoints object to store the coordinates
    points = vtk.vtkPoints()

    # Create a vtkCellArray object to define lines
    lines = vtk.vtkCellArray()

    # Adjust the point IDs for skipped points
    point_id_adjustment = np.zeros(coordinates.shape[0], dtype=np.int64)
    adjusted_id_counter = 0

    current_index = 0
    for length in index_array:
        # Skip single-point segments or segments that become single-point due to skipping (-1, -1)
        if length < 2:
            current_index += length
            continue

        # Initialize the line
        line = vtk.vtkPolyLine()
        line_point_ids = []

        for i in range(current_index, current_index + length):
            coord = coordinates[i]
            # Check if the coordinate is (-1, -1) and skip if true
            if not (coord[0] == -1 and coord[1] == -1):
                points.InsertNextPoint(coord[0], coord[1], 0.0)  # Z-coordinate is set to 0
                line_point_ids.append(adjusted_id_counter)
                adjusted_id_counter += 1
            point_id_adjustment[i] = adjusted_id_counter

        # Add points to the line considering skipped points
        for pid in line_point_ids:
            line.GetPointIds().InsertNextId(pid)
        
        # Check if the line is valid after potentially skipping points
        if line.GetNumberOfPoints() >= 2:
            lines.InsertNextCell(line)

        current_index += length

    # Create vtkPolyData to store points and lines
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    # Save vtkPolyData as a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileTypeToBinary()
    writer.SetFileName(f"/Users/mingzexia/Documents/Github/tracecp/{dataset}/dat{input}.vtk")
    writer.SetInputData(polydata)
    writer.Write()

write_traj_to_vtk("small_data", "")
print("done")
write_traj_to_vtk("small_data", ".out")
print("done")


# create vtk file that contains the difference between the original and the decompressed data trajectory
def write_traj_to_vtk(dataset, cat,input):
    coordinates = np.fromfile(f'/Users/mingzexia/Documents/Github/tracecp/{dataset}/{cat}_{input}.bin', dtype=np.float64).reshape(-1, 2)
    index_array = np.fromfile(f'/Users/mingzexia/Documents/Github/tracecp/{dataset}/index_{cat}_{input}.bin', dtype=np.int32)

    # Create a vtkPoints object to store the coordinates
    points = vtk.vtkPoints()

    # Create a vtkCellArray object to define lines
    lines = vtk.vtkCellArray()

    # Adjust the point IDs for skipped points
    point_id_adjustment = np.zeros(coordinates.shape[0], dtype=np.int64)
    adjusted_id_counter = 0

    current_index = 0
    for length in index_array:
        # Skip single-point segments or segments that become single-point due to skipping (-1, -1)
        if length < 2:
            current_index += length
            continue

        # Initialize the line
        line = vtk.vtkPolyLine()
        line_point_ids = []

        for i in range(current_index, current_index + length):
            coord = coordinates[i]
            # Check if the coordinate is (-1, -1) and skip if true
            if not (coord[0] == -1 and coord[1] == -1):
                points.InsertNextPoint(coord[0], coord[1], 0.0)  # Z-coordinate is set to 0
                line_point_ids.append(adjusted_id_counter)
                adjusted_id_counter += 1
            point_id_adjustment[i] = adjusted_id_counter

        # Add points to the line considering skipped points
        for pid in line_point_ids:
            line.GetPointIds().InsertNextId(pid)
        
        # Check if the line is valid after potentially skipping points
        if line.GetNumberOfPoints() >= 2:
            lines.InsertNextCell(line)

        current_index += length

    # Create vtkPolyData to store points and lines
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    # Save vtkPolyData as a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileTypeToBinary()
    writer.SetFileName(f"/Users/mingzexia/Documents/Github/tracecp/{dataset}/{cat}_{input}.vtk")
    writer.SetInputData(polydata)
    writer.Write()

write_traj_to_vtk("small_data","ori_found_dec_found_diff", "ORI")
print("done")
write_traj_to_vtk("small_data","ori_found_dec_found_diff" ,"DEC")
print("done")
write_traj_to_vtk("small_data","ori_found_dec_not_found", "ORI")
print("done")
write_traj_to_vtk("small_data","ori_found_dec_not_found" ,"DEC")
print("done")