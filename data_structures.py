'''
objects to hold experiments
'''
import tkinter as tk
import pandas as pd
from tkinter import filedialog
import time, datetime
import os

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
        self.opt = opt


    def data(self):
        return self.data

    def params(self):
        return self.params

    def opt(self):
        return self.opt

    def to_excel(self, filepath: str = None):
        """
        :param filepath: String of .xlsx file name.
        :return: No return statements

        Save experiment object to a single .xlsx file.
        WARNING: There are size and read/write speed limitations inherent to .xlsx.
        """
        if filepath is None:
            filepath = filedialog.asksaveasfile(mode='wb', filetypes=[('Excel Worksheet', '.xlsx')],
                                                defaultextension='.xlsx')
            with filepath as f:
                filepath = f.name
        else:
            filepath = filepath
        if filepath[-5:] != '.xlsx':
            filepath = str(filepath + '.xlsx')

        with pd.ExcelWriter(filepath) as writer:
            self.data.to_excel(writer, engine='openpyxl', sheet_name='data')
            self.params.to_excel(writer, engine='openpyxl', sheet_name='params')
            if self.opt is not None:
                for i in range(len(self.opt)):
                    self.opt[i].to_excel(writer, engine='openpyxl', sheet_name='opt'+str(i))

    def to_csv(self, parentdir: str = None, dirname: str = None):
        """
        :param parentdir: Parent directory to which to save directory of CSVs.
        :param dirname: Name of directory of CSVs.
        :return: No return statements

        Creates a new directory in the selected directory
        """
        if parentdir is None:
            parentdir = filedialog.askdirectory(title='Select the Parent Directory')
        if dirname is None:
            ts = time.time()
            dirname = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

        os.mkdir(parentdir + '/' + dirname)

        self.data.to_csv(parentdir + '/' + dirname + '/data.csv')
        self.params.to_csv(parentdir + '/' + dirname + '/params.csv')
        if self.opt is not None:
            for i in range(len(self.opt)):
                self.opt[i].to_csv(parentdir + '/' + dirname + '/opt' + str(i) + '.csv')
