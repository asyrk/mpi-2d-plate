

/* ======================================================================== */
/*   mc_pi_serv.c                                                           */
/*   MPI program for calculating Pi by Monte Carlo estimation               */
/* ======================================================================== */
/***
   * Fatcat MPICH compilation: 
   *  /opt/nfs/mpich-3.0.2/bin/mpicc.mpich -o mcpi mcpi.c -lm
   * Fatcat MPICH example execution:
   *  /opt/nfs/mpich-3.0.2/bin/mpiexec -n 4 ./mcpi 0.000001
   */

#include <stdio.h>
#include <math.h>
#include "mpi.h"
#define CHUNKSIZE 10000
/* We'd like a value that gives the maximum value returned by the function
   random, but no such value is *portable*.  RAND_MAX is available on many 
   systems but is not always the correct value for random (it isn't for 
   Solaris).  The value ((unsigned(1)<<31)-1) is common but not guaranteed */
#define INT_MAX   1000000000
#define THROW_MAX 100000000
#define PI        3.141592653589793238462643
/* message tags */
#define REQUEST   1
#define REPLY     2
#define DONE	  3

int main( int argc, char *argv[] )
{
	srand(time(NULL));
    int numprocs, myid, server, workerid, ranks[1], 
        request, i, iter, ix, iy, done;
    long rands[CHUNKSIZE], max, in, out, totalin, totalout;
    double x, y, Pi, error, epsilon;
    MPI_Comm world, workers;
    MPI_Group world_group, worker_group;
    MPI_Status status;

    MPI_Init( &argc, &argv );
    
    world  = MPI_COMM_WORLD;
    MPI_Comm_size( world, &numprocs );
    MPI_Comm_rank( world, &myid );
    server = numprocs-1;	// Last process is a random server 

/***
   * Now Master should read epsilon from command line
   * and distribute it to all processes.
   */

   /* ....Fill in, please.... */

	if(myid == 0)
	{
		printf("Epsilon: \n");
		scanf("%lf", &epsilon);
	}
	MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, world);

/***
   * Create new process group called world_group containing all 
   * processes and its communicator called world
   * and a group called worker_group containing all processes
   * except the last one (called here server) 
   * and its communicator called workers.
   */

   /* ....Fill in, please.... */


    MPI_Comm_group(world, &world_group );
    ranks[0] = server;
    MPI_Group_excl(world_group, 1, ranks, &worker_group);
    MPI_Comm_create(world, worker_group, &workers);
    MPI_Group_free(&worker_group);

/***
   * Server part
   *
   * Server should loop until request code is 0, in each iteration:
   * - receiving request code from any slave
   * - generating a vector of CHUNKSIZE randoms <= INT_MAX
   * - sending vector back to slave 
   */
    if (myid == server) {
		do
		{
			MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, world, &status);
			if(request)
			{
				for(i = 0; i < CHUNKSIZE; ++i)
				{
					rands[i] = rand() % INT_MAX + 1;
				}
			
				MPI_Send (rands, CHUNKSIZE, MPI_INT, status.MPI_SOURCE, REPLY, world);
			}
		} while(request > 0);
		printf("OUT WHILE\n");
		MPI_Recv(&Pi, 1, MPI_DOUBLE, 0, DONE, world, &status);
		printf("SERVER: Pi: %lf\t PI: %lf\n", Pi, PI);
				
   /* ....Fill in, please.... */

/*
	    MPI_Recv( &request, ... );
	    
	    ...
	    
	    MPI_Send( rands, ... );
*/

    }
/***
   * Workers (including Master) part
   *
   * Worker should send initial request to server.
   * Later, in a loop worker should:
   * - receive vector of randoms 								OK
   * - compute x,y point inside unit square						OK
   * - check (and count result) if point is inside/outside 		OK
   *   unit circle
   * - sum both counts over all workers							OK
   * - calculate Pi and its error (from "exact" value) in all workers	OK
   * - test if error is within epsilon limit					
   * - test continuation condition (error and max. points limit)
   * - print Pi by master only
   * - send a request to server (all if more or master only if finish)
   * Before finishing workers should free their communicator.
   */ 
    else {			// I am a worker process
		max = INT_MAX;
		//MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
		MPI_Comm_rank(workers, &workerid);
		iter = 0;
		while (!done || (totalin + totalout) > THROW_MAX)
		{
			++iter;
			request = !done;
			done = in = out = 0;
			MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
			MPI_Recv(rands, CHUNKSIZE, MPI_INT, server, REPLY, world, &status);
			for(i = 0; i < CHUNKSIZE / 2; i += 2)
			{
				x = (double)rands[i] / max;
				y = (double)rands[i+1] / max;
				if(sqrt(x * x + y * y) <= 1.)
					++in;
				else
					++out;
			}
			MPI_Allreduce(&in, &totalin, 1, MPI_INT, MPI_SUM, workers);
			MPI_Allreduce(&out, &totalout, 1, MPI_INT, MPI_SUM, workers);
			
			Pi = 4. * (double)totalin / (double)(totalin + totalout);
			printf("in: %d\tout: %d\n", totalin, totalout);
			if(fabs(Pi - PI) < epsilon)
			{
				done = 1;
			}
		}
		request = 0;
		if(myid == 0) {
			MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
			MPI_Send(&Pi, 1, MPI_DOUBLE, server, DONE, world);
		}

	}
    /* ....Fill in, please.... */
    
/*
            MPI_Send( &request, ... );
                        
            ...
                                                
            MPI_Recv( rands, ... );

        ... throw number of darts 
        ... calculate Pi globally
        ... test epsilon condition
                                                            
	MPI_Comm_free( ... );
*/




/***
   * Master should print final point counts.
   */


/* ....Fill in, please.... */


    MPI_Finalize();
}

