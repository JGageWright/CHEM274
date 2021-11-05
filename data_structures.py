'''
objects to hold experiments
'''
import tkinter as tk
import pandas as pd
from tkinter import filedialog

Tk = tk.Tk()
Tk.withdraw()

class experiment:
    def __init__(self, data: pd.DataFrame, params: pd.DataFrame, opt=None):
        '''
        :param data: Dataframe holding main data
        :param params: Dataframe holding parameters
        :param opt: Iterable of Dataframes holding optional data

        Initializes the experiment object
        '''
        self.data = data
        self.data.name = 'data'
        self.params = params
        self.params.name = 'params'
        self.filepath = None
        self.opt = opt
        # Rename opt dataframes
        # for df in self.opt():
        #     i = 0
        #     df.name = 'opt' + str(i)
        #     i += 1

    def data(self):
        return self.data

    def params(self):
        return self.params

    def opt(self):
        return self.opt

    def to_excel(self):
        self.filepath = filedialog.asksaveasfile(mode='wb', filetypes=[('Excel Worksheet', '.xlsx')],
                                                     defaultextension='.xlsx')
        self.filepath = self.filepath.name

        if self.filepath[-5:] != '.xlsx':
            self.filepath = str(self.filepath + '.xlsx')

        with pd.ExcelWriter(self.filepath) as writer:
            self.data.to_excel(writer, engine='openpyxl', sheet_name='data')
            self.params.to_excel(writer, engine='openpyxl', sheet_name='params')
            if self.opt != None:
                for i in range(len(self.opt)):
                    self.opt[i].to_excel(writer, engine='openpyxl', sheet_name='opt'+str(i))

    def to_csv(self):
        self.filepath = filedialog.asksaveasfile(mode='wb', filetypes=[('CSV', '.csv')],
                                                     defaultextension='.csv')
        self.filepath = self.filepath.name

        if self.filepath[-4:] != '.csv':
            self.filepath = str(self.filepath + '.csv')
        self.data.to_csv(self.filepath)
