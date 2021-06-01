# -*- coding: utf-8 -*-
"""
Created on Sat May 22 10:02:16 2021

@author: andrea
"""

from ScopeFoundry import BaseMicroscopeApp

class HexSimAnalysisApp(BaseMicroscopeApp):

    # this is the name of the microscope that ScopeFoundry uses 
    # when storing data
    name = 'hex_sim_app'
    
    # You must define a setup function that adds all the 
    #capablities of the microscope and sets default settings
    def setup(self):
        
        from HexSimAnalyser_measurement import HexSimAnalysis
        self.add_measurement(HexSimAnalysis)
        
        # show ui
        self.ui.show()
        self.ui.activateWindow()


if __name__ == '__main__':
    import sys
    
    app = HexSimAnalysisApp(sys.argv)
    #app.settings_load_ini(".\\Settings\\HexSIM_Analysis.ini")
    
    sys.exit(app.exec_())