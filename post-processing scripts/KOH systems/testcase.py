from mpi_python import Prot_Hop

print(r"Running Python Now", flush=True)

pwd = r"/home/jelle/simulations/RPBE_Production/6m/AIMD/"
s = r"/i_"
r = r"/part_"

for i in range(1, 6):
    for j in range(1, 8):
        try:
            Traj = Prot_Hop(pwd + s + str(i) + r + str(j), dt=0.5)
            print("Success with file:",  s + str(i) + r + str(j), flush=True)
        except:
            print("Problem with file:", s + str(i) + r + str(j), flush=True)