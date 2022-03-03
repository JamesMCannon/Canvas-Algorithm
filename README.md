# canvas_algorithm
canvas cubesat development!

this code is to run the canvas data processing algorthim that will be loaded onto the FPGA onboard CANVAS

we used this code base to test the algorithm and run input data through this code and through the FPGA simulation, and see where there were differences

it helped us understand how the FPGA does these computations and what differences we might expect.

```fpgamodel.py``` includes all the functions that are used and outlines the steps of the data processing.

for more detail, contact the CANVAS team for payload documentation

to install this repo into your computer: 
```
git clone https://github.com/JamesMCannon/Canvas-Algorithm.git
```

move into your newly cloned repo:
```
cd canvas_algorithm
```

and create a virtual environment:
```
python3 -m venv canalg_env
```

activate that environment:
Unix/Mac:```
source canalg_env/bin/activate
```
Windows:```
.\canalg_env\Scripts\activate
```

and finally install packages
```
pip install -r requirements.txt
```

to run a test of the canvas alrgorithm, navigate to the ```fpgamodel.py``` and run that script with the desired input settings
