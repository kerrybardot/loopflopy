import sys
import os

class Project:
    def __init__(self, name, workspace, results, figures, triexe):
        self.name = name
        self.workspace = '../modelfiles/'
        self.results = '../results/'
        self.figures = '../figures/'
        self.triexe = '../exe/triangle.exe'
        
        if not os.path.isdir(self.workspace): os.makedirs(self.workspace, exist_ok=True)
        if not os.path.isdir(self.results):   os.makedirs(self.results, exist_ok=True)
        if not os.path.isdir(self.figures):   os.makedirs(self.figures, exist_ok=True)

