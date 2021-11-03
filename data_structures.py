'''
objects to hold experiments
'''
import tkinter as tk
import pandas as pd
from tkinter import filedialog
from importer_snippets import ask_path

Tk = tk.Tk()
Tk.withdraw()

class experiment:
    '''Write me'''
    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.save_as = ask_path()


    pass