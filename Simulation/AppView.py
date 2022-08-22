import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Literal

from PIL import Image, ImageTk

class AppView(tk.Toplevel):

    def __init__(self, root: tk.Tk, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.__frames: dict[str,ttk.Frame] = {}
        self.__labels: dict[str,ttk.Label] = {}
        self.__buttons: dict[str,ttk.Button] = {}

        self.update_train_pars_cmd = None

        self.initGUI()

    def initGUI(self):
        self.title("TM-ML")
        self.geometry("700x500")

        # main frame
        self.__create_frame("main")
        self.__pack_frame("main", fill=tk.BOTH, expand=True, side='left', padx=5)
        self.__create_label("MAIN", self.get_frame("main"))
        self.__pack_label("MAIN",side='top')

        # control frame
        self.__create_frame("control")
        self.__pack_frame("control", fill=tk.Y,side='right', padx=5, expand=True)
        self.__create_label("CONTROL", self.get_frame("control"))
        self.__pack_label("CONTROL")

        self.__create_button("quit",self.get_frame("control"))
        self.__pack_button("quit",fill=tk.X)

        # tabs
        self.tabs = ttk.Notebook(self.get_frame("main"))
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self.tabs.bind('<Control-w>', lambda e: self.tabs.forget("current") if self.tabs.index("end") > 0 else self.quit())

        # training tab
        self.__create_frame("tab1", self.tabs)
        self.__pack_frame("tab1", fill=tk.BOTH, expand=True)
        self.__create_label("Train", self.get_frame("tab1"))
        self.__pack_label("Train")
        self.__create_button("Train CCGAN", self.get_frame("tab1"))
        self.__pack_button("Train CCGAN")

        self.pcanv = tk.Canvas(self.get_frame("tab1"))
        self.pcanv.pack(fill=tk.BOTH, expand=True)

        self.scroll_train_params = ttk.Scrollbar(self.pcanv, command=self.pcanv.yview)
        self.scroll_train_params.pack(side=tk.RIGHT, fill=tk.Y)

        self.pcanv.configure(yscrollcommand=self.scroll_train_params.set)

        
        self.__create_frame("param_frame", self.pcanv)
        self.__pack_frame("param_frame", fill=tk.BOTH, expand=True)
        
        # self.get_frame("param_frame").configure()
        
        # scroll_train_params.pack(side='right', fill=tk.Y)
        

        # scroll_train_params.configure(command=train_params.yview)
        

        # analysis tab
        self.__create_frame("tab2", self.tabs)
        self.__pack_frame("tab2", fill=tk.BOTH, expand=True)
        self.__create_label("Analyze", self.get_frame("tab2"))
        self.__pack_label("Analyze")

        # self.__create_frame("tabplus", self.tabs)
        # self.__pack_frame("tabplus", fill=tk.X, expand=True)

        self.tabs.add(self.get_frame("tab1"), state='normal', text="train", underline=0)
        self.tabs.add(self.get_frame("tab2"), state='normal', text="analyze", underline=0)
        self.tabs.enable_traversal()

    def start(self):
        self.bind('<Control-c>', lambda e: self.quit())
        
        img_IP = Image.open("./status_orange.png")
        img_IP = img_IP.resize((20,20), Image.ANTIALIAS)
        self.status_IP = ImageTk.PhotoImage(img_IP)
        
        img_DONE = Image.open("./status_green.png")
        img_DONE = img_DONE.resize((20,20), Image.ANTIALIAS)
        self.status_DONE = ImageTk.PhotoImage(img_DONE)

        img_TERM = Image.open("./status_red.png")
        img_TERM = img_TERM.resize((20,20), Image.ANTIALIAS)
        self.status_TERM = ImageTk.PhotoImage(img_TERM)


    # general widget creation methods used internally (model will not be adding any components)

    def __create_frame(self, name: str, parent: tk.Frame=None, *args, **kwargs):
        if parent == None:
            parent = self
        self.__frames[name] = ttk.Frame(parent, *args, **kwargs)

    def __create_label(self, name: str, parent: tk.Frame=None, text=None, *args, **kwargs):
        if parent == None:
            parent = self
        if text == None:
            text = name
        self.__labels[name] = ttk.Label(parent,*args, text=text, **kwargs)

    def __create_button(self, name: str, parent: tk.Frame=None, *args, **kwargs):
        if parent == None:
            parent = self
        self.__buttons[name] = ttk.Button(parent, *args, text=name, **kwargs)

    def __pack_frame(self, name: str, **kwargs):
        self.__frames[name].pack(**kwargs)

    def __pack_label(self, name: str, **kwargs):
        self.__labels[name].pack(**kwargs)

    def __pack_button(self, name: str, **kwargs):
        self.__buttons[name].pack(**kwargs)

    def get_frame(self, name:str) -> ttk.Frame:
        '''
        Return the frame with the given name
        '''
        return self.__frames[name]

    def button_cmd(self, name: str, cmd: Callable):
        '''
        Set the Callable cmd to be the action of the button named name
        '''
        self.__buttons[name].configure(command=cmd)

    def set_update_train_pars_cmd(self, cmd):
        '''
        Set the Callable cmd to the be action of train parameter Entry widgets
        '''
        self.update_train_pars_cmd = cmd

    def display_train_params(self, params_dict: dict[str, Any]):
        for i, p in enumerate(params_dict.items()):
            k = p[0]
            v = p[1]

            key_label = ttk.Label(self.get_frame("param_frame"), text=k, width=30, background='#FFFFFF')
            key_label.grid(row=i, column=0, sticky='w')
            
            s = tk.StringVar(self, v)
            s.trace("w", lambda name, index, mode, s=s, k=k: self.update_train_pars_cmd(k, s.get()))
            val_entry = ttk.Entry(self.get_frame("param_frame"), textvariable=s)
            val_entry.grid(row=i, column=1)

    def display_params(self, run: int, params_dict: dict[str, Any]):
        newtab_name = "run %i"%(run + 1)
        self.__create_frame(newtab_name, self.tabs)
        self.__pack_frame(newtab_name, fill=tk.BOTH, expand=True)
        self.tabs.add(self.get_frame(newtab_name), state="normal", text="params %i"%(run), underline=7)

        for i, p in enumerate(params_dict.items()):
            key_frame = tk.Frame(self.get_frame(newtab_name), background="#CCCCCC")
            key_frame.grid(row=i, column=0, sticky='w')
            key_label = ttk.Label(key_frame, text=p[0], width=35, background="#FFFFFF")
            key_label.pack(padx=1, pady=1)

            val_frame = tk.Frame(self.get_frame(newtab_name), background="#CCCCCC")
            val_frame.grid(row=i, column=2, sticky='w')
            val_label = ttk.Label(val_frame, text=p[1], width=35, background="#FFFFFF")
            val_label.pack(padx=1, pady=1)

            sep = ttk.Separator(self.get_frame(newtab_name), orient=tk.VERTICAL)
            sep.grid(row=i, column=1, sticky='ns')

    def display_run_control(self, run: int, params_dict: dict[str, Any]):
        run_frame_name = "run %i control"%(run)
        img_lbl_name = run_frame_name + "_img"
        txt_lbl_name = run_frame_name + "_txt"

        self.__create_frame(run_frame_name, self.get_frame("control"))
        self.__pack_frame(run_frame_name, side=tk.TOP, expand=False, fill=tk.X,padx=5)
        
        self.__create_label(img_lbl_name, self.get_frame(run_frame_name), text='')
        self.__pack_label(img_lbl_name, fill=tk.NONE, expand=False, side='left', padx=5, pady=5, anchor='w')

        self.__create_label(txt_lbl_name, self.get_frame(run_frame_name), text="run %i"%(run))
        self.__pack_label(txt_lbl_name, fill=tk.NONE, expand=False, side='left', padx=5, pady=5, anchor='w')
        self.__labels[txt_lbl_name].bind("<1>", lambda e: self.display_params(run, params_dict))

    def display_train(self, run: int, params_dict: dict[str, Any]):
        self.display_params(run, params_dict)
        self.display_run_control(run, params_dict)

    def set_run_status(self, run: int, status: Literal['IP', 'DONE', 'TERM']):
        status = status.upper()
        if status == 'IP':
            img = self.status_IP
        elif status == 'DONE':
            img = self.status_DONE
        elif status == 'TERM':
            img = self.status_TERM
        
        lbl = self.__labels["run %i"%(run)]
        lbl.image = img
        lbl.configure(image=img)