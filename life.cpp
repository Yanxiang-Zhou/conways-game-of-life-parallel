#include <stdio.h> /*for printf()*/
#include <stdlib.h>/*for rand(),malloc(),free()*/
#include <mpi.h>
#include <omp.h>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <unistd.h>


#define TAG 0

template <typename T>
std::string vec_to_str(const std::vector<T>& vec,
                       const std::string& delim = ",");

template <typename T>
std::string vec_to_str(const std::vector<T>& vec, const std::string& delim) {
  std::ostringstream oss;
  if (!vec.empty()) {
    std::copy(vec.begin(), vec.end() - 1,
              std::ostream_iterator<T>(oss, delim.c_str()));
    oss << vec.back();
  }
  return oss.str();
}


/*helper functions*/
int linear_index(int m, int n, int row, int col);
void update_state(int m, int n, const int* in_grid, int* out_grid);
void read_data(const std::string& input_filename, int m, int n, std::vector<int>& output_data);
void write_data(const std::string& output_filename, int m, int n, const std::vector<int>& output_data);
int coord_to_index(int * coords, int proc_num_row, int proc_num_col);
void allocate_row_col(int* cell_num_row, int* cell_num_col, int size, int m, int n, int proc_num_row, int proc_num_col);
void allocate_disp(int* disps, int* cell_num_row, int* cell_num_col, int n, int proc_num_row, int proc_num_col);
void allocate_extent(int* extent,int* cell_num_col,int proc_num_row,int proc_num_col);
int** allocate_memory(int rows,int columns);
void create_datatype(MPI_Datatype* derivedtype, int start1,int start2,int subsize1,int subsize2,int local_num_row, int local_num_col);
void find_next_state(int i,int j,int sum,int** &local_matrix, int** &next_gen);
void find_neighbourhood_sum(int current_i,int current_j,int *sum, int** &local_matrix);
void calculate_inner_matrix(int local_num_row, int local_num_col, int** &local_matrix, int** &next_gen);
void find_neighbours(MPI_Comm comm_2D,int my_rank,int NPROWS,int NPCOLS,int* left,int* right,int* top,int* bottom,int* topleft,int* topright,int* bottomleft,int* bottomright);
void game(MPI_Comm comm_2D,int my_rank,int NPROWS,int NPCOLS,int MAX_GENS,int local_num_row, int local_num_col, int** &local_matrix);


std::vector<int> local_cells;
int changed;

int main(int argc, char** argv){
	MPI_Init(&argc, &argv);

	const int m = std::stoi(argv[1]);
	const int n = std::stoi(argv[2]);
	const int gen = std::stoi(argv[3]);
	const std::string input_file(argv[4]);
	const std::string output_file(argv[5]);

	int rank;
	int size;
	double local_start,local_finish,local_elapsed,elapsed;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	std::vector<int> global_data;
	std::vector<int> output_data;

	/* On root, read in the data */
	if (rank == 0)
		read_data(input_file, m, n, global_data);

	if (size == 1) { /* If serial, use the serial code */
		/* Allocate the output data buffer */
		local_start = MPI_Wtime();
		printf("%.3f\n", local_start);


		output_data.reserve(global_data.size());

		/* For each generation update the state */
		for (int i = 0; i < gen; i++) {
			update_state(m, n, global_data.data(), output_data.data());

			/* Swap the input and output */
			if (i < gen - 1) {
				std::swap(global_data, output_data);
			}
		}

		local_finish = MPI_Wtime();
		printf("%.3f\n", local_finish);
		// local_elapsed = local_finish - local_start;
		local_finish -= local_start;

		printf("Elapsed time (sequential): %.4f seconds\n\n",local_finish);

    } else {
   
		int dims[2] = {0,0};
		MPI_Dims_create(size, 2, dims);

		int periods[2] = {0,0};
		int my_coords[2];
		MPI_Comm comm_2D;
		MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_2D);
		MPI_Cart_coords(comm_2D, rank, 2, my_coords);

		const int proc_num_row = dims[0]; /* Number of processor rows */
		const int proc_num_col = dims[1]; /* Number of processor cols */

		int local_num_row, local_num_col;

		int *cell_num_row; /* Number of rows for the i-th process [local_num_rows]*/
		int *cell_num_col; /* Number of columns for the i-th process [local_num_cols]*/
		int *disps; /*Displacement for the i-th process [local_disp]*/

		std::vector<int> data_for_scatter;
		std::vector<int> data_for_receive;

		std::vector<int> scatter_sendcnts;
		std::vector<int> scatter_disps;

		int recv_cnt;

		int max_num_row, max_num_col, max_block_size;

		MPI_Barrier(comm_2D);


		/**Calculate number of cells spread on the processors**/
		if (rank == 0) {
			local_start = MPI_Wtime();
			printf("%.3f\n", local_start);


			cell_num_row = (int *) malloc(size * sizeof(int));
			cell_num_col = (int *) malloc(size * sizeof(int));
			allocate_row_col(cell_num_row, cell_num_col, size, m, n, proc_num_row, proc_num_col);
			disps = (int *) malloc(size * sizeof(int));
			allocate_disp(disps, cell_num_row, cell_num_col, n, proc_num_row, proc_num_col);

			max_num_row = cell_num_row[0];
			max_num_col = cell_num_col[0];
			max_block_size = max_num_row * max_num_col;

			/**prepare initial data for each processor**/

			data_for_scatter.reserve(size * max_block_size);

			scatter_sendcnts.reserve(size);
			scatter_disps.reserve(size);

			for (int proc_index = 0; proc_index < size; proc_index ++){
				int local_pos, global_pos;
				int col_size, row_size, first_disp, row_disp;

				col_size = cell_num_col[proc_index];
				row_size = cell_num_row[proc_index];
				first_disp = disps[proc_index];

				for (int i = 0; i < row_size; i++){
					row_disp = i * n;

					for (int j = 0; j < col_size; j++){
						local_pos = i * col_size + j + proc_index * max_block_size;
						global_pos = first_disp + row_disp + j;
						data_for_scatter[local_pos] = global_data[global_pos];
					}
				}
				scatter_disps[proc_index] = proc_index * max_block_size;
				scatter_sendcnts[proc_index] = max_block_size;
			}
		}

		/**send initial data from master (rank=0)**/

		MPI_Bcast(&max_block_size, 1, MPI_INT, 0, comm_2D);
		MPI_Bcast(&max_num_row, 1, MPI_INT, 0, comm_2D);
		MPI_Bcast(&max_num_col, 1, MPI_INT, 0, comm_2D);

		MPI_Scatter(cell_num_row, 1, MPI_INT, &local_num_row, 1, MPI_INT, 0, comm_2D);
		MPI_Scatter(cell_num_col, 1, MPI_INT, &local_num_col, 1, MPI_INT, 0, comm_2D);

		data_for_receive.reserve(max_block_size);

		MPI_Request req;
		MPI_Status stat;

		if (rank == 0) {
			for (int i = 0; i < size; i ++) {
				MPI_Isend(&scatter_sendcnts[i], 1, MPI_INT, i, 0, comm_2D, &req);
			}
			MPI_Wait(&req, &stat);

			for (int i = 0; i < size; i ++) {
				MPI_Isend(&data_for_scatter[scatter_disps[i]], scatter_sendcnts[i], MPI_INT, i, 1, comm_2D, &req);
			}
			MPI_Wait(&req, &stat);
		}

		MPI_Recv(&recv_cnt, 1, MPI_INT, 0, 0, comm_2D, &stat);
    		MPI_Recv(&data_for_receive[0], recv_cnt, MPI_INT, 0, 1, comm_2D, &stat);


		std::vector<int> local_cells((local_num_col+2) * (local_num_row+2));
		int ori_index, padding_index;
		for (int i = 0; i < local_num_row; i++) {
			for (int j = 0; j < local_num_col; j++) {
				ori_index = i * local_num_col + j;
				padding_index = (i + 1) * (local_num_col + 2) + j + 1;
				local_cells[padding_index] = data_for_receive[ori_index];
			}
		}


		/**create local 2D matrix for updating**/
		int** local_matrix;
		int** next_gen;

		local_matrix=allocate_memory(local_num_row+2,local_num_col+2);

		for (int i = 0; i < local_cells.size(); i++) {
			int row = i / (local_num_col+2);
			int col = i % (local_num_col+2);
			local_matrix[row][col] = local_cells[i];
		}

		/**main update function**/

		game(comm_2D, rank, proc_num_row, proc_num_col, gen, local_num_row, local_num_col, local_matrix);

		for (int i = 0; i < local_num_row + 2; ++i) {
			for (int j = 0; j < local_num_col + 2; ++j) {
				local_cells[i * (local_num_col + 2) + j] = local_matrix[i][j];
			}
		}

		MPI_Reduce(&local_elapsed,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,comm_2D);

		/**prepare to receive update date from processors**/
		std::vector<int> data_to_gather(max_block_size);
		std::vector<int> data_gathered;
		std::vector<int> data_from_procs(max_block_size);
		std::vector<int> gather_recvcnts;
		std::vector<int> gather_disps;

		if (rank == 0) {
			data_gathered.reserve(size * max_block_size);
			gather_disps.reserve(size);
			gather_recvcnts.reserve(size);

			for (int i = 0; i < size; i ++) {
				gather_recvcnts[i] = max_block_size;
				gather_disps[i] = i*max_block_size;
			}
		}


		for (int i = 0; i < local_num_row; i++) {
			for (int j = 0; j < local_num_col; j++) {
				ori_index = i * local_num_col + j;
				padding_index = (i+1) * (local_num_col+2) + j + 1;
				data_to_gather[ori_index] = local_cells[padding_index];
			}
		}

		MPI_Gatherv(&data_to_gather[0], max_block_size, MPI_INT,
						&data_gathered[0], &gather_recvcnts[0], &gather_disps[0], MPI_INT, 0, comm_2D);

		local_cells.clear();

		/**collect the gathered data on master**/
		if (rank == 0){
			output_data.reserve(m*n);

			for (int proc_index = 0; proc_index < size; proc_index ++){
				int local_pos, global_pos;
				int col_size, row_size, first_disp, row_disp;

				col_size = cell_num_col[proc_index];
				row_size = cell_num_row[proc_index];
				first_disp = disps[proc_index];

				for (int i = 0; i < row_size; i++){
					row_disp = i * n;

					for (int j = 0; j < col_size; j++){
						local_pos = i * col_size + j + proc_index * max_block_size;
						global_pos = first_disp + row_disp + j;

						output_data[global_pos] = data_gathered[local_pos];
					}
				}

			}

		}

		if(rank == 0){
			free(cell_num_row);
			free(cell_num_col);
			free(disps);

			local_finish = MPI_Wtime();
			printf("%.3f\n", local_finish);
			local_finish -= local_start;
			printf("Elapsed time (parallel): %.3f seconds\n\n",local_finish);
		}
	}

  	/* On root, output the data */
	if (rank == 0){
		write_data(output_file, m, n, output_data);
	}

	MPI_Finalize();
}

/*helper functions*/

/*sequential helper functions*/
int linear_index(int m, int n, int row, int col) {
  return row * n + col;
}

/* A sequential function to update a grid,
 * uses LDA (leading dimension) to allow for subgrid considerations */
void update_state(int m, int n, const int* in_grid, int* out_grid) {
  for (int i = 0; i < m; i++) { // For each row
    for (int j = 0; j < n; j++) { // For each column
      //Consider a single element
      int lin_loc = linear_index(m, n, i, j); //This is the linear index of the element

      int alive = 0;

      /* Look at each neighbor */
      for (int k = -1; k < 2; k++) {
        for (int l = -1; l < 2; l++) {
          /* Figure out the index associated with each neighbor */
          int y_loc = i + k;
          int x_loc = j + l;
          int neighbor_lin_loc = linear_index(m, n, y_loc, x_loc);

          /* Ensure the considered neighbor is in bounds */
          if ((x_loc >= 0) && (y_loc >= 0) && (y_loc < m) && (x_loc < n)) {
            /* Check that the neighbor is actually a neighbor */
            if (!(k == 0 && l == 0)) {
              /* If it is alive, count it as alive */
              if (in_grid[neighbor_lin_loc]) {
                alive++;
              }
            }
          }
        }
      }

      /* Based on the number of alive neighbors, update the output accordingly */
      if (in_grid[lin_loc]) {
        if (alive < 2) {
          out_grid[lin_loc] = 0;
        } else if (alive > 3) {
          out_grid[lin_loc] = 0;
        } else {
          out_grid[lin_loc] = 1;
        }
      } else {
        if (alive == 3) {
          out_grid[lin_loc] = 1;
        } else {
          out_grid[lin_loc] = 0;
        }
      }
    }
  }
}

/* Read in the data from an input txt file */
void read_data(const std::string& input_filename, int m, int n,
               std::vector<int>& output_data) {
  output_data.reserve(m * n);
  std::ifstream input_file(input_filename, std::ios::in);
  for (int i = 0; i < m * n; i++) {
    std::string mystring;
    input_file >> mystring;
    output_data.push_back(std::stoi(mystring));
  }
}

/* Write output data to a file */
/*parallel helper functions*/
void write_data(const std::string& output_filename, int m, int n,
                const std::vector<int>& output_data) {
  std::ofstream output_file(output_filename, std::ios::out);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      output_file << output_data[linear_index(m, n, i, j)];
      if (j < n - 1) {
        output_file << " ";
      }
    }
    output_file << "\n";
  }
}

int coord_to_index(int * coords, int proc_num_row, int proc_num_col){
	int index = coords[0] * proc_num_row + coords[1];
	return index;
}

void allocate_row_col(int* cell_num_row, int* cell_num_col, int size, int m, int n, int proc_num_row, int proc_num_col){
	/*
		m:9
		n:9
		proc_num_row:4
		proc_num_col:4
	*/
	for (auto i = 0; i < size; i++) {
		cell_num_row[i] = m / proc_num_row;
		cell_num_col[i] = n / proc_num_col;
	}

	for (auto i = 0; i < m % proc_num_row; i++) {
		for (auto j = 0; j < proc_num_col; j++) {
			cell_num_row[i*proc_num_col+j]++;
		}
	}

	for (auto i = 0; i < n % proc_num_col; i++) {
		for (auto j = 0; j < proc_num_row; j++) {
			cell_num_col[i+proc_num_row*j]++;
		}
	}
}

void allocate_disp(int* disps, int* cell_num_row, int* cell_num_col, int n, int proc_num_row, int proc_num_col){
	int proc_index;
	for (auto i = 0; i < proc_num_row; i++)
		for (auto j = 0; j < proc_num_col; j++){
			proc_index = i*proc_num_col + j;

			if (j == 0){
				// int row;
				disps[proc_index] = 0;

				/*add up previous rows*/
				if (i != 0){
					disps[proc_index] += disps[proc_index - 1] + cell_num_col[proc_index - 1] + (cell_num_row[proc_index - 1] - 1) * n;
				}
				// printf("index: %d, disp: %d\n", proc_index, disps[proc_index]);
			} else{
				/*Just add cell_num_col of the left process to its displacement*/
				disps[proc_index] = disps[proc_index - 1] + cell_num_col[proc_index - 1];
				// printf("index: %d, disp: %d\n", proc_index, disps[proc_index]);
			}
		}
}

void allocate_extent(int* extent,int* cell_num_col,int proc_num_row,int proc_num_col){
	for (auto i = 0; i < proc_num_row; i++)
		for (auto j = 0; j < proc_num_col; j++){

			int current = i*proc_num_col+j;
			int block;

			/*Add newline to the extent*/
			/*Add num_cols of this process*/
			extent[current] = 1 + cell_num_col[current];

			/*For all blocks on the left of me*/
			for (block = i*proc_num_col; block < current; block++)
				extent[current] += cell_num_col[block];
			/*For all blocks on the right */
			for (block = current+1; block < (i+1)*proc_num_col; block++)
				extent[current] += cell_num_col[block];
		}
}

int** allocate_memory(int rows,int columns) {
	int *data = (int *)malloc(rows*columns*sizeof(int));
    int **ans = (int **)malloc(rows*sizeof(int *));
    for (auto i = 0; i < rows; i++)
        ans[i] = &(data[i*columns]);

	return ans;
}

// void create_datatype(MPI_Datatype* derivedtype, int local_num_row, int local_num_col, int start1,int start2,int subsize1,int subsize2)
void create_datatype(MPI_Datatype* derivedtype, int start1, int start2, int subsize1, int subsize2, int local_num_row, int local_num_col) {
	int array_of_bigsizes[2] = {local_num_row+2, local_num_col+2};
	int array_of_subsizes[2] = {subsize1, subsize2};
	int array_of_starts[2] = {start1, start2};

	MPI_Type_create_subarray(2,array_of_bigsizes,array_of_subsizes,array_of_starts,MPI_ORDER_C, MPI_INT, derivedtype);
	MPI_Type_commit(derivedtype);
}

void find_next_state(int i,int j,int sum, int** &local_matrix, int** &next_gen) {
	/*If cell is alive*/
	if (local_matrix[i][j]==1) {
		if(sum == 0 || sum == 1) {
			next_gen[i][j] = 0;
			changed = 1;
		} else if(sum == 2 || sum == 3)
			next_gen[i][j] = 1;
		else if(sum>=4 && sum<=8) {
			next_gen[i][j] = 0;
			changed = 1;
		}
	}
	/*If cell is not alive and has 3 active neighbours*/
	else if (sum == 3) {
		next_gen[i][j] = 1;
		changed = 1;
	}
	/*If cell is not alive but it has less than 3 active neighbours*/
	else
		next_gen[i][j] = 0;
}

void find_neighbourhood_sum(int current_i,int current_j, int *sum, int** &local_matrix) {
	*sum = 0;
	/*For all my 8 neighbours*/
	for(auto i = -1; i <= 1; ++i) {
		for(auto j = -1; j <= 1; ++j) {
			if(i || j)
				/*If neighbour is alive,add it to sum*/
				if(local_matrix[current_i+i][current_j+j]==1)
					(*sum)++;
		}
	}
}

void calculate_inner_matrix(int local_num_row, int local_num_col, int** &local_matrix, int** &next_gen) {
	int sum;

	/*For all cells that require no communication at all*/
	for(auto i = 1;i <= local_num_row; ++i)
		for(auto j = 1; j <= local_num_col; ++j) {
			find_neighbourhood_sum(i, j, &sum, local_matrix);
			find_next_state(i, j, sum, local_matrix, next_gen);
		}
}

void find_neighbours(MPI_Comm comm_2D,int my_rank,int NPROWS,int NPCOLS,int* left,int* right,int* top,int* bottom,int* topleft,int* topright,int* bottomleft,int* bottomright) {
	int source,dest,disp=1;
	int my_coords[2];
	int corner_coords[2];
	int corner_rank;

	/*Finding top/bottom neighbours*/
	MPI_Cart_shift(comm_2D,0,disp,top,bottom);

	/*Finding left/right neighbours*/
	MPI_Cart_shift(comm_2D,1,disp,left,right);

	/*Finding top-right corner*/
	MPI_Cart_coords(comm_2D,my_rank,2,my_coords);
	corner_coords[0] = my_coords[0] - 1;
	corner_coords[1] = my_coords[1] + 1;
	if(corner_coords[0] < 0)
		*topright = MPI_PROC_NULL;
	else if (my_coords[1] + 1 >= NPCOLS)
		*topright = MPI_PROC_NULL;
	else
		MPI_Cart_rank(comm_2D,corner_coords,topright);

	/*Finding top-left corner*/
	MPI_Cart_coords(comm_2D,my_rank,2,my_coords);
	corner_coords[0] = my_coords[0] - 1;
	corner_coords[1] = my_coords[1] - 1;
	if(corner_coords[0]<0)
		*topleft = MPI_PROC_NULL;
	else if (corner_coords[1]<0)
		*topleft = MPI_PROC_NULL;
	else
		MPI_Cart_rank(comm_2D,corner_coords,topleft);

	/*Finding bottom-right corner*/
	MPI_Cart_coords(comm_2D,my_rank,2,my_coords);
	corner_coords[0] = my_coords[0] + 1;
	corner_coords[1] = my_coords[1] + 1;
	if (corner_coords[0] >= NPROWS)
		*bottomright = MPI_PROC_NULL;
	else if (corner_coords[1] >= NPCOLS)
		*bottomright = MPI_PROC_NULL;
	else
		MPI_Cart_rank(comm_2D,corner_coords,bottomright);

	/*Finding bottom-left corner*/
	MPI_Cart_coords(comm_2D,my_rank,2,my_coords);
	corner_coords[0] = my_coords[0] + 1;
	corner_coords[1] = my_coords[1] - 1;
	if (corner_coords[1] < 0)
		*bottomleft = MPI_PROC_NULL;
	else if (corner_coords[0] >= NPROWS)
		*bottomleft = MPI_PROC_NULL;
	else
		MPI_Cart_rank(comm_2D,corner_coords,bottomleft);

}

void game(MPI_Comm comm_2D,int my_rank,int NPROWS,int NPCOLS,int MAX_GENS,int local_num_row, int local_num_col, int** &local_matrix) {
	int i,j;
	int **next_gen = allocate_memory(local_num_row + 2, local_num_col + 2);

	/*Create 4 datatypes for sending*/
	MPI_Datatype firstcolumn_send,firstrow_send,lastcolumn_send,lastrow_send;
	create_datatype(&firstcolumn_send, 1,1,local_num_row,1,local_num_row,local_num_col);
	create_datatype(&firstrow_send, 1,1,1,local_num_col,local_num_row,local_num_col);
	create_datatype(&lastcolumn_send, 1,local_num_col,local_num_row,1,local_num_row,local_num_col);
	create_datatype(&lastrow_send,local_num_row,1,1,local_num_col,local_num_row,local_num_col);

	/*Create 4 datatypes for receiving*/
	MPI_Datatype firstcolumn_recv,firstrow_recv,lastcolumn_recv,lastrow_recv;
	create_datatype(&firstcolumn_recv,1,0,local_num_row,1,local_num_row,local_num_col);
	create_datatype(&firstrow_recv,0,1,1,local_num_col,local_num_row,local_num_col);
	create_datatype(&lastcolumn_recv,1,local_num_col+1,local_num_row,1,local_num_row,local_num_col);
	create_datatype(&lastrow_recv,local_num_row+1,1,1,local_num_col,local_num_row,local_num_col);

	/*Find ranks of my 8 neighbours*/
	int left,right,bottom,top,topleft,topright,bottomleft,bottomright;
	find_neighbours(comm_2D,my_rank,NPROWS,NPCOLS,&left,&right,&top,&bottom,&topleft,&topright,&bottomleft,&bottomright);

	/*16 requests , 16 statuses */
	MPI_Request array_of_requests[16];
	MPI_Status array_of_statuses[16];

	MPI_Send_init(*local_matrix, 1, firstcolumn_send, left, TAG, comm_2D, &array_of_requests[0]);
	MPI_Send_init(*local_matrix, 1, firstrow_send, top, TAG, comm_2D, &array_of_requests[1]);
	MPI_Send_init(*local_matrix, 1, lastcolumn_send, right, TAG,comm_2D, &array_of_requests[2]);
	MPI_Send_init(*local_matrix, 1, lastrow_send, bottom, TAG,comm_2D, &array_of_requests[3]);
	MPI_Send_init(&(local_matrix[1][1]), 1, MPI_INT, topleft, TAG,comm_2D, &array_of_requests[4]);
	MPI_Send_init(&(local_matrix[1][local_num_col]), 1, MPI_INT, topright, TAG, comm_2D, &array_of_requests[5]);
	MPI_Send_init(&(local_matrix[local_num_row][local_num_col]), 1, MPI_INT, bottomright, TAG, comm_2D, &array_of_requests[6]);
	MPI_Send_init(&(local_matrix[local_num_row][1]), 1, MPI_INT, bottomleft, TAG, comm_2D, &array_of_requests[7]);

	MPI_Recv_init(*local_matrix,1,firstcolumn_recv,left,TAG,comm_2D,&array_of_requests[8]);
	MPI_Recv_init(*local_matrix,1,firstrow_recv,top,TAG,comm_2D,&array_of_requests[9]);
	MPI_Recv_init(*local_matrix,1,lastcolumn_recv,right,TAG,comm_2D,&array_of_requests[10]);
	MPI_Recv_init(*local_matrix,1,lastrow_recv,	bottom,TAG,comm_2D,&array_of_requests[11]);
	MPI_Recv_init(&(local_matrix[0][0]),1,MPI_INT,topleft,TAG,comm_2D,&array_of_requests[12]);
	MPI_Recv_init(&(local_matrix[0][local_num_col+1]),1,MPI_INT,topright,TAG,comm_2D,&array_of_requests[13]);
	MPI_Recv_init(&(local_matrix[local_num_row+1][local_num_col+1]),1,MPI_INT,bottomright,TAG,comm_2D,&array_of_requests[14]);
	MPI_Recv_init(&(local_matrix[local_num_row+1][0]),1,MPI_INT,bottomleft,TAG,comm_2D,&array_of_requests[15]);

	for(auto gen = 0; gen < MAX_GENS; gen++) {
		//changed = 0;

		/*Start all requests [8 sends + 8 receives]*/
		// MPI_Startall(16, array_of_requests);
		MPI_Start(&array_of_requests[0]);
		MPI_Start(&array_of_requests[1]);
		MPI_Start(&array_of_requests[2]);
		MPI_Start(&array_of_requests[3]);
		MPI_Start(&array_of_requests[4]);
		MPI_Start(&array_of_requests[5]);
		MPI_Start(&array_of_requests[6]);
		MPI_Start(&array_of_requests[7]);
		MPI_Start(&array_of_requests[8]);
		MPI_Start(&array_of_requests[9]);
		MPI_Start(&array_of_requests[10]);
		MPI_Start(&array_of_requests[11]);
		MPI_Start(&array_of_requests[12]);
		MPI_Start(&array_of_requests[13]);
		MPI_Start(&array_of_requests[14]);
		MPI_Start(&array_of_requests[15]);
		/*Make sure all requests are completed*/
		// MPI_Waitall(16, array_of_requests, array_of_statuses);
		MPI_Wait(&array_of_requests[0],&array_of_statuses[0]);
		MPI_Wait(&array_of_requests[1],&array_of_statuses[1]);
		MPI_Wait(&array_of_requests[2],&array_of_statuses[2]);
		MPI_Wait(&array_of_requests[3],&array_of_statuses[3]);
		MPI_Wait(&array_of_requests[4],&array_of_statuses[4]);
		MPI_Wait(&array_of_requests[5],&array_of_statuses[5]);
		MPI_Wait(&array_of_requests[6],&array_of_statuses[6]);
		MPI_Wait(&array_of_requests[7],&array_of_statuses[7]);
		MPI_Wait(&array_of_requests[8],&array_of_statuses[8]);
		MPI_Wait(&array_of_requests[9],&array_of_statuses[9]);
		MPI_Wait(&array_of_requests[10],&array_of_statuses[10]);
		MPI_Wait(&array_of_requests[11],&array_of_statuses[11]);
		MPI_Wait(&array_of_requests[12],&array_of_statuses[12]);
		MPI_Wait(&array_of_requests[13],&array_of_statuses[13]);
		MPI_Wait(&array_of_requests[14],&array_of_statuses[14]);
		MPI_Wait(&array_of_requests[15],&array_of_statuses[15]);
		// if (!my_rank) {
		// 	for (int i = 0; i < local_num_row+2; ++i) {
		// 		for (int j = 0; j < local_num_col + 2; ++j) {
		// 			std::cout<< local_matrix[i][j] << " ";
		// 		}
		// 		std::cout<<std::endl;
		// 	}
		// 	std::cout<<std::endl;
		// }

		for (int i = 0; i < local_num_row + 2; ++i) {
			for (int j = 0; j < local_num_col + 2; ++j) {
				next_gen[i][j] = local_matrix[i][j];
			}
		}

		/*Overlap communication [calculating inner matrix]*/
		calculate_inner_matrix(local_num_row, local_num_col, local_matrix, next_gen);

		for (int i = 0; i < local_num_row + 2; ++i) {
			for (int j = 0; j < local_num_col + 2; ++j) {
				local_matrix[i][j] = next_gen[i][j];
			}
		}

		//std::swap(local_matrix, next_gen);
	}

	MPI_Request_free(&array_of_requests[0]);
	MPI_Request_free(&array_of_requests[1]);
	MPI_Request_free(&array_of_requests[2]);
	MPI_Request_free(&array_of_requests[3]);
	MPI_Request_free(&array_of_requests[4]);
	MPI_Request_free(&array_of_requests[5]);
	MPI_Request_free(&array_of_requests[6]);
	MPI_Request_free(&array_of_requests[7]);
	MPI_Request_free(&array_of_requests[8]);
	MPI_Request_free(&array_of_requests[9]);
	MPI_Request_free(&array_of_requests[10]);
	MPI_Request_free(&array_of_requests[11]);
	MPI_Request_free(&array_of_requests[12]);
	MPI_Request_free(&array_of_requests[13]);
	MPI_Request_free(&array_of_requests[14]);
	MPI_Request_free(&array_of_requests[15]);

	MPI_Type_free(&firstcolumn_send);
	MPI_Type_free(&firstrow_send);
	MPI_Type_free(&lastcolumn_send);
	MPI_Type_free(&lastrow_send);

	MPI_Type_free(&firstcolumn_recv);
	MPI_Type_free(&firstrow_recv);
	MPI_Type_free(&lastcolumn_recv);
	MPI_Type_free(&lastrow_recv);

	// for (auto i = 0; i < local_num_row + 2; ++i)
	// 	free(next_gen[i]);
	// free(next_gen);
}
/*end of helper functions*/
