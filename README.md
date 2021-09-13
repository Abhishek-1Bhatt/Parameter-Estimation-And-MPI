# Parameter-Estimation-And-MPI

This Project aims at exploring the Dormand Prince 5 method for finding the solution of non-stiff  ODEs. The DP5 method is a 5th order Runge-Kutta method. The file [here](https://github.com/Abhishek-1Bhatt/Parameter-Estimation-And-MPI/blob/main/DP_lv.jl) presents a very simple implementation of the DP5 method with a fixed timestep and attempts to solve the Lotka-Volterra equations for ecological dynamics. The Lotka-Volterra equations describe the interaction between predator and prey populations in a habitat with the help of certain parameters. They are given as:

    dx/dt = αx − βxy
    dy/dt = −γy + δxy
    
Where x and y denote the prey and predator populations respectively. Now solving these two ODEs gives the population change as the time progresses and thus gives the information whether the two species can coexist without one driving the other to extinction. Here the prey grows at an exponential rate but has a term that reduces its population by being eaten by the predator. The predator's growth is dependent on the available food (the amount of prey) and has a decay rate due to old age. On solving the ODEs with parameters α=1.5 , β=1.0, γ=3.0, and δ=1.0 with the OrdinaryDiffEq.jl Package in Julia gives:

![image](https://user-images.githubusercontent.com/46929125/132629174-3672fb80-2197-4fee-843b-01b91d995ca2.png)

### Output of The Fixed Time Step Solver

The Lotka-Volterra equations when solved with the DP5 implementation done [here](https://github.com/Abhishek-1Bhatt/Parameter-Estimation-And-MPI/blob/main/DP_lv.jl) with the same parameters gives:

![lotka](https://user-images.githubusercontent.com/46929125/132630472-301552bb-39e3-4abb-9625-7de165cbf7c7.png)

As we can see the DP5 implementation here is able to approximate the solution to some accuracy but it fails at time steps where the values are changing fast, i.e. , at the peaks. Incorporating certain error control metrics in the code, i.e. , including a variable time step to reduce the error in each time step with the help of a sixth order calculation with coefficients provided in the Dormand Prince tableau and adjusting the time step with the error rate can provide a more accurate result. Hence, next I will be looking into the same and in future I will be updating the results here.

### Distributed Computing with MPI.jl

MPI.jl is the julia interface for utilising the Message Passing Interface standard in a High Performance Cluster. It allows the same program to be run on different nodes of a High Performance Cluster(Distributed SPMD). But it can also be used in a single node(like a pc) to communicate between multiple processes. This is what has been shown in this [script](https://github.com/Abhishek-1Bhatt/Parameter-Estimation-And-MPI/blob/main/mpass.jl) 
where I have used two processes to generate the Bandwidth vs Message Size Plot by having two processes send each other 2^n length arrays(n=1:25) of Int8(1 byte) values. Send-Receive is one of the many types of communications which can be done through MPI called Point-to-Point communication. Other methods of communication are Gather, Scatter, Reduce and Broadcast.

The two processes give two plots as output, as shown,

![4 mpass0](https://user-images.githubusercontent.com/46929125/132634459-8bd7ee89-5397-48eb-a37a-31c42290c4f7.png)

![4 mpass1](https://user-images.githubusercontent.com/46929125/132634498-af28eee3-0071-4d49-8b1b-44b81b798d3b.png)

where we have the Bandwidth(MB/s) vertically and message size(Bytes) horizontally.

#### Still in development

[Here](https://github.com/Abhishek-1Bhatt/Parameter-Estimation-And-MPI/blob/main/DP_AD.jl) I am trying to develop a method to estimate the parameters for the lotka-volterra model using the ode solver above. The idea is to have the solution of Lotka-Volterra model shown above as the given data and start with some incorrect estimate of the parameters α=1.2, β=0.8, γ=2.8, and δ=0.8. Then we can compute the output of the model with the off parameters and compute an L2 norm as the loss function to perform gradient descent and perturb the off parameters towards a close approximation of the correct parameters. A key tool in implementing this is to use Forward Mode Accumulation/Automatic Differentiation/Algorithmic Differentiation with the chain rule to calculate the gradient of the output w.r.t. the parameters.
