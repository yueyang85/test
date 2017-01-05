
import os


project_dir = "D:\\svn\\k450\\design\\analog\project\\"
cell_path = "D:\\svn\\k450\\design\\analog\\libraries\\acr_lib\\acr_afe_top\\acr_afe_top.asc"

print(cell_path)

with open(cell_path) as f:
    #lines = [line.rstrip() for line in open(cell_path)]
    lines = [line.rstrip() for line in f.readlines()]
    


for line in lines:
    print(line)
    
    
