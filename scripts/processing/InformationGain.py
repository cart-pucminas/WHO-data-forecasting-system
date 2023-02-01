import pandas as pd
from info_gain import info_gain

def informationGain(firtDf, nameFirtColumn, secondDf, nameSecondColumn):
    if ((not firtDf.empty) and (not secondDf.empty)):
        try:
            mergeDf = pd.merge(secondDf, firtDf, how='left', on=['year', 'countryName'])        
            ig  = info_gain.info_gain(mergeDf[nameSecondColumn], mergeDf[nameFirtColumn])
            iv  = info_gain.intrinsic_value(mergeDf[nameSecondColumn], mergeDf[nameFirtColumn])
            igr = info_gain.info_gain_ratio(mergeDf[nameSecondColumn], mergeDf[nameFirtColumn])
            return ig, iv, igr
            
        except Exception:
            print('Error during processing Information Gain')
            return 0,0,0
    else:
        return 0,0,0
        