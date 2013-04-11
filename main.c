#include <stdio.h>

#include <math.h>
#include <string.h>
#include "/opt/mpich/include/mpi.h"

#define N 50
#define EPSILON 0.01
#define INITIALIZE 4
#define FINISHED 5
#define NEIGHBOUR_UP_MSG 2
#define NEIGHBOUR_DOWN_MSG 3
#define NO_NEIGHBOUR -1
#define MAX_IT 100

//xor swap
#define SWAP(a, b) ((a)^=(b),(b)^=(a),(a)^=(b))

void print_data(double u[2][N][N], char *file, int z)
{
	FILE *f = fopen(file, "w");
	int i, j;
	for(i = 0; i < N; ++i)
	{
		for(j = 0; j < N; ++i)
		{
			fprintf(f, "%lf\t", u[z][i][j]);
		}
		fprintf(f, "\n");
	}
}

int main(int argc, char** argv)
{
    double diff, global_diff, mean;
    int i,j;
    double u[2][N][N]; //3d array for better performance, u[0] - grid for old values, u[1] - grid for new values, final solution in u[0]
    int myid, m, server, rows_per_worker, extra_rows, rows, message, neighbour_up, neighbour_down, offset = 0, z, tmpi, z2, start, end, procs;
    MPI_Status status;

    MPI_Init(&argc, &argv);

    MPI_Comm_size( MPI_COMM_WORLD, &procs );
    MPI_Comm_rank( MPI_COMM_WORLD, &myid );
    server = 0;	// first process is server
    m = procs - 1;
    // Grid initialization

    // BEGIN SERVER PART
    if (myid == server) {
        rows_per_worker = N / m;
        extra_rows = N % m;

		// initialize grid
        for(i = 0; i < N; ++i)
        {
            u[0][i][0] = 100.0;
            u[0][i][N-1] = 0;
            u[0][0][i] = 60.0;
            u[0][N-1][i] = 29.0;
            mean += u[0][1][0] + u[0][1][N-1] + u[0][0][1] + u[0][N-1][1];
        }
        mean /= (4.0 * N);

        for(i = 1; i < N-1; ++i)
            for(j = 1; j < N-1; ++j)
                u[0][i][j] = u[1][i][j] = mean;

        // Initialize workers
        for(i = 1; i <= m; ++i) { // i <-> number of worker
            // Take care of extra rows by distributing them evenly among workers
            if(extra_rows) {
                rows = rows_per_worker + 1;
                --extra_rows;
            }
            else {
                rows = rows_per_worker;
            }

            neighbour_up = i-1;
            neighbour_down = i+1;

			// if first or last workers then one neighbour is missing
            if(i == 1) {
                neighbour_up = NO_NEIGHBOUR;
            }
            else if (i == m) {
                neighbour_down = NO_NEIGHBOUR;
            }

            message = INITIALIZE;

            //MPI_Send(&i, 1, MPI_INT, i, message, MPI_COMM_WORLD);

            //printf("[server] Sending data...\n");

            MPI_Send(&offset, 1, MPI_INT, i, message, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, i, message, MPI_COMM_WORLD);
            MPI_Send(&neighbour_up, 1, MPI_INT, i, message, MPI_COMM_WORLD);
            MPI_Send(&neighbour_down, 1, MPI_INT, i, message, MPI_COMM_WORLD);
            MPI_Send(&u[0][offset][0], rows * N, MPI_DOUBLE, i, message, MPI_COMM_WORLD);

            //printf("[server] Data sent!\n");

            offset += rows;
        }

        // Wait for results
        //printf("[server] Waiting for results...\n");
        for(i = 1; i <= m; ++i) {
            message = FINISHED;
            MPI_Recv(&offset, 1, MPI_INT, i, message, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, i, message, MPI_COMM_WORLD, &status);
            MPI_Recv(&u[0][offset][0], rows * N, MPI_DOUBLE, i, message, MPI_COMM_WORLD, &status);
        }
        //printf("[server] Got results!\n");
        
        // Print final result
        for(i = 0; i < N; ++i)
        {
            for(j = 0; j < N; ++j)
                printf("%.3lf\t", u[0][i][j]);
            printf("\n");
        }

    }
    // END OF SERVER PART

    // BEGIN WORKER PART
    if(myid != server) {
        // Initialize worker according to information sent by server
        //printf("[worker] Waiting for initialization...\n");
        message = INITIALIZE;

        MPI_Recv(&offset, 1, MPI_INT, server, message, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, server, message, MPI_COMM_WORLD, &status);
        MPI_Recv(&neighbour_up, 1, MPI_INT, server, message, MPI_COMM_WORLD, &status);
        MPI_Recv(&neighbour_down, 1, MPI_INT, server, message, MPI_COMM_WORLD, &status);
        MPI_Recv(&u[0][offset][0], rows * N, MPI_DOUBLE, server, message, MPI_COMM_WORLD, &status);
        //printf("[worker] Initialized!\n");
        z = 0;
        
        // set iteration start and end
        start = offset == 0 ? 1 : offset;
        end = (offset + rows) == N ? (offset + rows - 1) : (offset + rows);
        
        // begin main loop
        for(tmpi = 0; tmpi < MAX_IT; ++tmpi) {

			// get ghost rows from neighbours
            if(neighbour_up != NO_NEIGHBOUR) {
                //printf("[worker] Sending info to up neighbour... %d\n", neighbour_up);
                message = NEIGHBOUR_DOWN_MSG;
                MPI_Send(&u[z][offset][0], N, MPI_DOUBLE, neighbour_up, message, MPI_COMM_WORLD);
                message = NEIGHBOUR_UP_MSG;
                MPI_Recv(&u[z][offset-1][0], N, MPI_DOUBLE, neighbour_up, message, MPI_COMM_WORLD, &status);
                //printf("[worker] Up neighbour data sent!\n");
            }
            if(neighbour_down != NO_NEIGHBOUR) {
                //printf("[worker] Sending info to down neighbour... %d\n", neighbour_down);
                message = NEIGHBOUR_UP_MSG;
                MPI_Send(&u[z][offset + rows - 1][0], N, MPI_DOUBLE, neighbour_down, message, MPI_COMM_WORLD);
                message = NEIGHBOUR_DOWN_MSG;
                MPI_Recv(&u[z][offset + rows][0], N, MPI_DOUBLE, neighbour_down, message, MPI_COMM_WORLD, &status);
                //printf("[worker] Down neighbour data sent!\n");
            }
            
            // update grid values
            for(i = start; i < end; ++i)
            {
                for(j = 1; j < N-1; ++j)
                {
                    u[1-z][i][j] = (u[z][i-1][j] + u[z][i+1][j] + u[z][i][j-1] + u[z][i][j+1]) / 4.0;
//                    if(fabs(w[i][j] - u[i][j]) > diff)
//                        diff = fabs(w[i][j] - u[i][j]);
                }
            }
            for(i = start; i < end; ++i)
            {
                for(j = 1; j < N-1; ++j)
                {
                    u[z][i][j] = u[1-z][i][j];
                }
            }
            
        }
        
        // Work finished, send results to server
        //printf("[worker] Job done, sending results... \n");
        message = FINISHED;
        //char file[50];
        //sprintf(file, "worker%d.txt", myid);
        //print_data(u, file, 0);
        
        MPI_Send(&offset, 1, MPI_INT, server, message, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, server, message, MPI_COMM_WORLD);
        MPI_Send(&u[z][offset][0], rows * N, MPI_DOUBLE, server, message, MPI_COMM_WORLD);
//        if(myid == 1)
//        {
//                    for(i = 0; i < N; ++i)
//                    {
//                        for(j = 0; j < N; ++j)
//                            printf("%.3lf\t", u[0][i][j]);
//                        printf("\n");
//                    }
//        }
        //printf("[worker] Results sent!\n");
    }

    MPI_Finalize();
}
