# canvas_algorithm
canvas cubesat development!


this code is to run the canvas data processing algorthim that will be loaded onto the FPGA onboard CANVAS
this code was written in python because I know python

we used this code base to test the algorithm and run input data through this code and through the FPGA simulation, and see where there were differences

it helped us understand how the FPGA does these computations and what differences we might expect

we are now going to be using this to generate input data and run it through the physical FPGA and this code and see how they compare



fpgamodel.py includes all the functions that are used

first, test data is generated (either from table mtn or just fake data)
then, a windowing function is applied that is designed to match the FPGA window operation

next, the fft is taken. we found there are some differences in the fft between the fpga fft and the python fft, but we noted these and kept going

power values are found from the resulting fft data, and then the biggest part of the algorithm happens where the resulting fbins are averaged into log-spaced bins 
and the power values are accummulated over time and averaged every second

the result is compressed before it comes into the final science packets
