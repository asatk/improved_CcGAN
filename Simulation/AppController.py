#!/home/asatk/miniconda3/envs/py3CCGAN/bin/python

import sys

from AppModel import AppModel
from AppView import AppView
import tkinter as tk

class TMController():

    def __init__(self, mode: str):
        self.__root = tk.Tk()
        self.__root.withdraw()
        self.mode = mode
        self.model = AppModel()
        self.view = AppView(self.__root)
    
    def start(self):
        self.model.start()
        self.init_cmds()
        self.init_display()
        self.view.start()
        
        self.__root.mainloop()
        
    def init_cmds(self):
        view = self.view
        model = self.model

        #define commands that get event from view, process data in model, and update view
        quit_cmd = view.quit
        train_cmd = lambda: model.train() #needs to be async

        # set buttons to corresponding commands
        view.button_cmd("quit", quit_cmd)
        view.button_cmd("Train CCGAN", train_cmd)

        view.set_update_train_pars_cmd(model.update_train_pars) #needs to be async
        model.set_display_pars_cmd(view.display_train) #needs to be async

    def init_display(self):
        view = self.view
        model = self.model
        view.display_train_params(model.get_current_train_pars())

if __name__ == '__main__':
    #the mode is a base-10 num representing the bit-string for which apps (bits) one wishes to run
    #3 -> 11 (both); 2 -> 10 (analysis app); 1 -> 01 (main app)
    mode = ("{:02b}".format(int(sys.argv[1])) if len(sys.argv) > 1 else bin(3))
    controller = TMController(mode)
    controller.start()