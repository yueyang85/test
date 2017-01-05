
import os

class test1(object):
    def __init__(self):
        self.a = 1
        
    def __enter__(self):
        print ("enter")
        return self.a
    
    def __exit__(self,type,value,TypeError):
        print ("exit")
        return True

project_dir = "D:\\svn\\k450\\design\\analog\project\\"
cell_path = "D:\\svn\\k450\\design\\analog\\libraries\\acr_lib\\acr_afe_top\\acr_afe_top.asc"

print(cell_path)

with open(cell_path) as f:
    #lines = [line.rstrip() for line in open(cell_path)]
    lines = [line.rstrip() for line in f.readlines()]
    

print (len(lines))
for line in lines:
    print(line)
    
with test1() as t:
    print(t)      
