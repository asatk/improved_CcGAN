import asyncio

from AppModel import AppModel
from AppView import AppView
import tkinter as tk

class TMController():

    def __init__(self):
        self.__root = tk.Tk()
        self.__root.withdraw()
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
        train_cmd = model.train #needs to be async

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
    controller = TMController()
    controller.start()