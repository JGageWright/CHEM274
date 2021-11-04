'''
objects to hold experiments
'''
import tkinter as tk
import pandas as pd
from tkinter import filedialog

Tk = tk.Tk()
Tk.withdraw()

class experiment:
    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.filepath = None

    def data(self):
        return self.data

    def params(self):
        return self.params

    def to_excel(self):
        self.filepath = filedialog.asksaveasfile(mode='wb', filetypes=[('Excel Worksheet', '.xlsx')],
                                                     defaultextension='.xlsx')
        self.filepath = self.filepath.name

        if self.filepath[-5:] != '.xlsx':
            self.filepath = str(self.filepath + '.xlsx')

        with pd.ExcelWriter(self.filepath) as writer:
            self.data.to_excel(writer, engine='openpyxl', sheet_name='data')
            self.params.to_excel(writer, engine='openpyxl', sheet_name='params')

    def to_csv(self):
        self.filepath = filedialog.asksaveasfile(mode='wb', filetypes=[('CSV', '.csv')],
                                                     defaultextension='.csv')
        self.filepath = self.filepath.name

        if self.filepath[-4:] != '.csv':
            self.filepath = str(self.filepath + '.csv')

        with pd.ExcelWriter(self.filepath) as writer:
            self.data.to_excel(writer, engine='openpyxl', sheet_name='data')
            self.params.to_excel(writer, engine='openpyxl', sheet_name='params')