import numpy as np
import pandas as pd

class PolyReg:
    """
    Polynomial fitting routine based on numpy.polyfit
    Methods return more and more useful information with simpler syntax

    No parameters are fixed.
    Object to hold all information associated with a polynomial model.
    Polynomial coefficients in order of decreasing degree are in coef[i].
    Note that ss_yy is commonly named ss_tot in other implementations.

    coef = ndarray of fitting parameters in order of decreasing degree
    ss_res = sum of squares of residuals.
    std_err = ndarray of standard errors of the fitting parameters in order of decreasing degree.
    These are calculated as the square root of diagonal elements in the covariance matrix
    """
    def __init__(self, xdata, ydata, degree):
        self.xdata = xdata
        self.ydata = ydata
        self.degree = degree
        self.coef, self.cov = np.polyfit(xdata, ydata, degree, cov=True)
        self.residuals = ydata - np.polyval(self.coef, xdata)
        self.ss_res = np.sum(self.residuals**2)
        self.ss_yy = np.sum((ydata - np.mean(ydata)) ** 2)
        self.ss_xx = np.sum((xdata - np.mean(xdata)) ** 2)
        self.ss_xy = np.sum((xdata - np.mean(xdata))*(ydata - np.mean(ydata)))
        self.r_squared = 1 - (self.ss_res / self.ss_yy)
        self.std_err = np.sqrt(np.diag(self.cov))
        self.s_y = np.sqrt(self.ss_res / (len(ydata) - 1 - self.degree))
        self.roots = np.roots(self.coef)
        
    def report(self):
        '''
        Returns a table of quantities
        '''
        params = {}
        for i in range(len(self.coef)):
            params['coef_deg' + str(len(self.coef) - i -1)] = self.coef[i]
            params['std_err_deg' + str(len(self.coef) - i -1)] = self.std_err[i]
            
        params['r_squared'] = self.r_squared
        params['s_y'] = self.s_y
        series = pd.Series(params)
        return series
    
    def eval(self, x):
        return np.polyval(self.coef, x)