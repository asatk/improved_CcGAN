import asyncio
import json
import main
import multiprocessing as mp
import os
from PIL import Image, ImageTk
import re
import subprocess
from typing import Any, Callable, ItemsView

import defs
from Run import Run

class AppModel():
    
    def __init__(self):
        self.current_train_pars = {}
        self.current_analysis_pars = {}
        self.train_pars_list = list()
        self.analysis_pars_list = list()

        self.processes: list[mp.Process] = []

        self.display_pars_cmd = None

    def start(self):
        mp.set_start_method('fork')
        self.current_train_pars = self.load_train_pars()

    def quit(self):
        for p in self.processes:
            p.kill()
        #rm shared mem?

    def load_analysis_pars(self, filename: str) -> ItemsView:
        '''
        filename: .json file that lists all training (hyper)parameters
        '''
        return dict(json.load(open(filename))).items()

    def load_train_pars(self) -> list[tuple[str, Any]]:
        # f = lambda v: self.set_train_par(p[0])
        self.current_train_pars = dict((k, v) for k, v in defs.__dict__.items() if k[0] != '_')
        # self.current_train_pars = [list(p) for p in defs.__dict__.items() if p[0][0] != '_']
        # for p in self.current_train_pars:
        #     p.append(lambda v, p=p: self.set_train_par(p[0], v))
        # print(self.current_train_pars)
        return self.current_train_pars

    def update_train_pars(self, key: str, val: str):
        if re.match("^\d+$", val):
            val = int(val)
        elif re.match("^\d+\.e*\-*\d+$", val):
            val = float(val)
        self.current_train_pars[key] = val
        exec("defs.%s = %s"%(key, val))
        print(eval("defs.%s"%(key)))

    def set_train_par(self, key: str, val: any):
        self.current_train_pars[key] = val

    def get_current_train_pars(self):
        return self.current_train_pars

    def get_train_pars_list(self):
        return self.train_pars_list

    def set_display_pars_cmd(self, cmd: Callable):
        self.display_pars_cmd = cmd

    def train(self):
        self.display_pars_cmd(len(self.train_pars_list), self.current_train_pars)
        self.train_pars_list.append(self.current_train_pars)
        # p = subprocess.Popen(['python', 'main.py'], bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # p.wait()
        # stdout, stderr = p.communicate()
        # print(stdout)
        # print(stderr)

        run = Run(self.current_train_pars)

        p = mp.Process(target=main.main, daemon=True, args=[run])
        self.processes.append(p)
        p.start()
        # p.join()
        
        # pid = os.fork()
        # if pid == 0:
        #     print("child process")
        # else:
        #     print("parent process - child has pid: %i"%(pid))
            # subprocess.run(["python", "main.py"])
        # p = Process()

        

