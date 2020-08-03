# import libraries as needed
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from itertools import accumulate
from pylab import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Data Cleanup

## 1. convert all the missing values to nan
## 1.1 create a dictionary of all key value pairs for missing values per attribute
missingValuesDictionary = {
    'AGER_TYP' : ['-1'], # unknown
    'ALTERSKATEGORIE_GROB' : ['-1', '0'], # unknown
    'ALTER_HH' : ['0'], # unknown
    'ANREDE_KZ' : ['-1', '0'], # unknown
    'BALLRAUM' : ['-1'], # unknown
    'BIP_FLAG' : ['-1'], # unknown
    'CAMEO_DEU_2015' : ['-1', 'XX'], # unknown    
    'CAMEO_DEUG_2015' : ['-1', 'X'], # unknown
    'CAMEO_DEUINTL_2015' : ['-1'], # unknown
    'CAMEO_INTL_2015' : ['XX'], # unknown
    'CJT_GESAMTTYP' : ['0'], # unknown
    #'D19_BANKEN_ANZ_12' : ['0'], # not known
    #'D19_BANKEN_ANZ_24' : ['0'], # not known
    #'D19_BANKEN_DATUM' : ['10'], # not known
    #'D19_BANKEN_DIREKT_RZ' : ['0'], # not known
    #'D19_BANKEN_GROSS_RZ' : ['0'], # not known
    #'D19_BANKEN_LOKAL_RZ' : ['0'], # not known
    #'D19_BANKEN_OFFLINE_DATUM' : ['10'], # not known
    #'D19_BANKEN_ONLINE_DATUM' : ['10'], # not known
    #'D19_BANKEN_ONLINE_QUOTE_12' : ['0'], # none
    #'D19_BANKEN_REST_RZ' : ['0'], # not known
    #'D19_BEKLEIDUNG_GEH_RZ' : ['0'], # not known
    #'D19_BEKLEIDUNG_REST_RZ' : ['0'], # not known
    #'D19_BILDUNG_RZ' : ['0'], # not known
    #'D19_BIO_OEKO_RZ' : ['0'], # not known
    #'D19_BUCH_RZ' : ['0'], # not known
    #'D19_DIGIT_SERV_RZ' : ['0'], # not known
    #'D19_DROGERIEARTIKEL_RZ' : ['0'], # not known
    #'D19_ENERGIE_RZ' : ['0'], # not known
    #'D19_FREIZEIT_RZ' : ['0'], # not known
    #'D19_GARTEN_RZ' : ['0'], # not known
    #'D19_GESAMT_ANZ_12' : ['0'], # not known
    #'D19_GESAMT_ANZ_24' : ['0'], # not known
    #'D19_GESAMT_DATUM' : ['10'], # not known
    #'D19_GESAMT_OFFLINE_DATUM' : ['10'], # not known
    #'D19_GESAMT_ONLINE_DATUM' : ['10'], # not known
    #'D19_GESAMT_ONLINE_QUOTE_12' : ['0'], # not known
    #'D19_HANDWERK_RZ' : ['0'], # not known
    #'D19_HAUS_DEKO_RZ' : ['0'], # not known
    #'D19_KINDERARTIKEL_RZ' : ['0'], # not known
    #'D19_KK_KUNDENTYP' : ['-1'], # unknown
    #'D19_KONSUMTYP' : ['9'], #inactive, consumption type
    #'D19_KOSMETIK_RZ' : ['0'], # not known
    #'D19_LEBENSMITTEL_RZ' : ['0'], # not known
    #'D19_LOTTO_RZ' : ['0'], # not known
    #'D19_NAHRUNGSERGAENZUNG_RZ' : ['0'], # not known
    #'D19_RATGEBER_RZ' : ['0'], # not known
    #'D19_REISEN_RZ' : ['0'], # not known
    #'D19_SAMMELARTIKEL_RZ' : ['0'], # not known
    #'D19_SCHUHE_RZ' : ['0'], # not known
    #'D19_SONSTIGE_RZ' : ['0'], # not known
    #'D19_TECHNIK_RZ' : ['0'], # not known
    #'D19_TELKO_ANZ_12' : ['0'], # not known
    #'D19_TELKO_ANZ_24' : ['0'], # not known
    #'D19_TELKO_DATUM' : ['10'], # not known
    #'D19_TELKO_MOBILE_RZ' : ['0'], # not known
    #'D19_TELKO_OFFLINE_DATUM' : ['10'], # not known
    #'D19_TELKO_ONLINE_DATUM' : ['10'], # not known
    #'D19_TELKO_REST_RZ' : ['0'], # not known
    #'D19_TIERARTIKEL_RZ' : ['0'], # not known
    #'D19_VERSAND_ANZ_12' : ['0'], # not known
    #'D19_VERSAND_ANZ_24' : ['0'], # not known
    #'D19_VERSAND_DATUM' : ['10'], # not known
    #'D19_VERSAND_OFFLINE_DATUM' : ['10'], # not known
    #'D19_VERSAND_ONLINE_DATUM' : ['10'], # not known
    #'D19_VERSAND_ONLINE_QUOTE_12' : ['0'], #none
    #'D19_VERSAND_REST_RZ' : ['0'], # not known
    #'D19_VERSICHERUNGEN_RZ' : ['0'], # not known
    #'D19_VERSI_ANZ_12' : ['0'], # not known
    #'D19_VERSI_ANZ_24' : ['0'], # not known
    #'D19_VOLLSORTIMENT_RZ' : ['0'], # not known
    #'D19_WEIN_FEINKOST_RZ' : ['0'], # not known
    'EWDICHTE' : ['-1'], # unknown
    'FINANZTYP' : ['-1'], # unknown
    'FINANZ_ANLEGER' : ['-1'], # unknown
    'FINANZ_HAUSBAUER' : ['-1'], # unknown
    'FINANZ_MINIMALIST' : ['-1'], # unknown
    'FINANZ_SPARER' : ['-1'], # unknown
    'FINANZ_UNAUFFAELLIGER' : ['-1'], # unknown
    'FINANZ_VORSORGER' : ['-1'], # unknown
    'GEBAEUDETYP' : ['-1', '0'], # unknown
    'GEBURTSJAHR' : ['0'],
    'GEOSCORE_KLS7' : ['-1', '0'], # unknown
    'HAUSHALTSSTRUKTUR' : ['-1', '0'], # unknown
    'HEALTH_TYP' : ['-1'], # unknown
    'HH_EINKOMMEN_SCORE' : ['-1', '0'], # unknown
    'INNENSTADT' : ['-1'], # unknown
    'KBA05_ALTER1' : ['-1', '9'], # unknown
    'KBA05_ALTER2' : ['-1', '9'], # unknown
    'KBA05_ALTER3' : ['-1', '9'], # unknown
    'KBA05_ALTER4' : ['-1', '9'], # unknown
    'KBA05_ANHANG' : ['-1', '9'], # unknown
    'KBA05_ANTG1' : ['-1'], # unknown
    'KBA05_ANTG2' : ['-1'], # unknown
    'KBA05_ANTG3' : ['-1'], # unknown
    'KBA05_ANTG4' : ['-1'], # unknown
    'KBA05_AUTOQUOT' : ['-1', '9'], # unknown
    'KBA05_BAUMAX' : ['-1', '0'], # unknown
    'KBA05_CCM1' : ['-1', '9'], # unknown
    'KBA05_CCM2' : ['-1', '9'], # unknown
    'KBA05_CCM3' : ['-1', '9'], # unknown
    'KBA05_CCM4' : ['-1', '9'], # unknown
    'KBA05_DIESEL' : ['-1', '9'], # unknown
    'KBA05_FRAU' : ['-1', '9'], # unknown
    'KBA05_GBZ' : ['-1', '0'], # unknown
    'KBA05_HERST1' : ['-1', '9'], # unknown
    'KBA05_HERST2' : ['-1', '9'], # unknown
    'KBA05_HERST3' : ['-1', '9'], # unknown
    'KBA05_HERST4' : ['-1', '9'], # unknown
    'KBA05_HERST5' : ['-1', '9'], # unknown
    'KBA05_HERSTTEMP' : ['-1', '9'], # unknown
    'KBA05_KRSAQUOT' : ['-1', '9'], # unknown
    'KBA05_KRSHERST1' : ['-1', '9'], # unknown
    'KBA05_KRSHERST2' : ['-1', '9'], # unknown
    'KBA05_KRSHERST3' : ['-1', '9'], # unknown
    'KBA05_KRSKLEIN' : ['-1', '9'], # unknown
    'KBA05_KRSOBER' : ['-1', '9'], # unknown
    'KBA05_KRSVAN' : ['-1', '9'], # unknown
    'KBA05_KRSZUL' : ['-1', '9'], # unknown
    'KBA05_KW1' : ['-1', '9'], # unknown
    'KBA05_KW2' : ['-1', '9'], # unknown
    'KBA05_KW3' : ['-1', '9'], # unknown
    'KBA05_MAXAH' : ['-1', '9'], # unknown
    'KBA05_MAXBJ' : ['-1', '9'], # unknown
    'KBA05_MAXHERST' : ['-1', '9'], # unknown
    'KBA05_MAXSEG' : ['-1', '9'], # unknown
    'KBA05_MAXVORB' : ['-1', '9'], # unknown
    'KBA05_MOD1' : ['-1', '9'], # unknown
    'KBA05_MOD2' : ['-1', '9'], # unknown
    'KBA05_MOD3' : ['-1', '9'], # unknown
    'KBA05_MOD4' : ['-1', '9'], # unknown
    'KBA05_MOD8' : ['-1', '9'], # unknown
    'KBA05_MODTEMP' : ['-1', '9'], # unknown
    'KBA05_MOTOR' : ['-1', '9'], # unknown
    'KBA05_MOTRAD' : ['-1', '9'], # unknown
    'KBA05_SEG1' : ['-1', '9'], # unknown
    'KBA05_SEG10' : ['-1', '9'], # unknown
    'KBA05_SEG2' : ['-1', '9'], # unknown
    'KBA05_SEG3' : ['-1', '9'], # unknown
    'KBA05_SEG4' : ['-1', '9'], # unknown
    'KBA05_SEG5' : ['-1', '9'], # unknown
    'KBA05_SEG6' : ['-1', '9'], # unknown
    'KBA05_SEG7' : ['-1', '9'], # unknown
    'KBA05_SEG8' : ['-1', '9'], # unknown
    'KBA05_SEG9' : ['-1', '9'], # unknown
    'KBA05_VORB0' : ['-1', '9'], # unknown
    'KBA05_VORB1' : ['-1', '9'], # unknown
    'KBA05_VORB2' : ['-1', '9'], # unknown
    'KBA05_ZUL1' : ['-1', '9'], # unknown
    'KBA05_ZUL2' : ['-1', '9'], # unknown
    'KBA05_ZUL3' : ['-1', '9'], # unknown
    'KBA05_ZUL4' : ['-1', '9'], # unknown
    'KBA13_ALTERHALTER_30' : ['-1'], # unknown
    'KBA13_ALTERHALTER_45' : ['-1'], # unknown
    'KBA13_ALTERHALTER_60' : ['-1'], # unknown
    'KBA13_ALTERHALTER_61' : ['-1'], # unknown
    'KBA13_AUDI' : ['-1'], # unknown
    'KBA13_AUTOQUOTE' : ['-1'], # unknown
    'KBA13_BJ_1999' : ['-1'], # unknown
    'KBA13_BJ_2000' : ['-1'], # unknown
    'KBA13_BJ_2004' : ['-1'], # unknown
    'KBA13_BJ_2006' : ['-1'], # unknown
    'KBA13_BJ_2008' : ['-1'], # unknown
    'KBA13_BJ_2009' : ['-1'], # unknown
    'KBA13_BMW' : ['-1'], # unknown
    'KBA13_CCM_0_1400' : ['-1'], # unknown
    'KBA13_CCM_1000' : ['-1'], # unknown
    'KBA13_CCM_1200' : ['-1'], # unknown
    'KBA13_CCM_1400' : ['-1'], # unknown
    'KBA13_CCM_1400_2500' : ['-1'], # unknown
    'KBA13_CCM_1500' : ['-1'], # unknown
    'KBA13_CCM_1600' : ['-1'], # unknown
    'KBA13_CCM_1800' : ['-1'], # unknown
    'KBA13_CCM_2000' : ['-1'], # unknown
    'KBA13_CCM_2500' : ['-1'], # unknown
    'KBA13_CCM_2501' : ['-1'], # unknown
    'KBA13_CCM_3000' : ['-1'], # unknown
    'KBA13_CCM_3001' : ['-1'], # unknown
    'KBA13_FAB_ASIEN' : ['-1'], # unknown
    'KBA13_FAB_SONSTIGE' : ['-1'], # unknown
    'KBA13_FIAT' : ['-1'], # unknown
    'KBA13_FORD' : ['-1'], # unknown
    'KBA13_HALTER_20' : ['-1'], # unknown
    'KBA13_HALTER_25' : ['-1'], # unknown
    'KBA13_HALTER_30' : ['-1'], # unknown
    'KBA13_HALTER_35' : ['-1'], # unknown
    'KBA13_HALTER_40' : ['-1'], # unknown
    'KBA13_HALTER_45' : ['-1'], # unknown
    'KBA13_HALTER_50' : ['-1'], # unknown
    'KBA13_HALTER_55' : ['-1'], # unknown
    'KBA13_HALTER_60' : ['-1'], # unknown
    'KBA13_HALTER_65' : ['-1'], # unknown
    'KBA13_HALTER_66' : ['-1'], # unknown
    'KBA13_HERST_ASIEN' : ['-1'], # unknown
    'KBA13_HERST_AUDI_VW' : ['-1'], # unknown
    'KBA13_HERST_BMW_BENZ' : ['-1'], # unknown
    'KBA13_HERST_EUROPA' : ['-1'], # unknown
    'KBA13_HERST_FORD_OPEL' : ['-1'], # unknown
    'KBA13_HERST_SONST' : ['-1'], # unknown
    'KBA13_KMH_0_140' : ['-1'], # unknown
    'KBA13_KMH_110' : ['-1'], # unknown
    'KBA13_KMH_140' : ['-1'], # unknown
    'KBA13_KMH_140_210' : ['-1'], # unknown
    'KBA13_KMH_180' : ['-1'], # unknown
    'KBA13_KMH_211' : ['-1'], # unknown
    'KBA13_KMH_250' : ['-1'], # unknown
    'KBA13_KMH_251' : ['-1'], # unknown
    'KBA13_KRSAQUOT' : ['-1'], # unknown
    'KBA13_KRSHERST_AUDI_VW' : ['-1'], # unknown
    'KBA13_KRSHERST_BMW_BENZ' : ['-1'], # unknown
    'KBA13_KRSHERST_FORD_OPEL' : ['-1'], # unknown
    'KBA13_KRSSEG_KLEIN' : ['-1'], # unknown
    'KBA13_KRSSEG_OBER' : ['-1'], # unknown
    'KBA13_KRSSEG_VAN' : ['-1'], # unknown
    'KBA13_KRSZUL_NEU' : ['-1'], # unknown
    'KBA13_KW_0_60' : ['-1'], # unknown
    'KBA13_KW_110' : ['-1'], # unknown
    'KBA13_KW_120' : ['-1'], # unknown
    'KBA13_KW_121' : ['-1'], # unknown
    'KBA13_KW_30' : ['-1'], # unknown
    'KBA13_KW_40' : ['-1'], # unknown
    'KBA13_KW_50' : ['-1'], # unknown
    'KBA13_KW_60' : ['-1'], # unknown
    'KBA13_KW_61_120' : ['-1'], # unknown
    'KBA13_KW_70' : ['-1'], # unknown
    'KBA13_KW_80' : ['-1'], # unknown
    'KBA13_KW_90' : ['-1'], # unknown
    'KBA13_MAZDA' : ['-1'], # unknown
    'KBA13_MERCEDES' : ['-1'], # unknown
    'KBA13_MOTOR' : ['-1'], # unknown
    'KBA13_NISSAN' : ['-1'], # unknown
    'KBA13_OPEL' : ['-1'], # unknown
    'KBA13_PEUGEOT' : ['-1'], # unknown
    'KBA13_RENAULT' : ['-1'], # unknown
    'KBA13_SEG_GELAENDEWAGEN' : ['-1'], # unknown
    'KBA13_SEG_GROSSRAUMVANS' : ['-1'], # unknown
    'KBA13_SEG_KLEINST' : ['-1'], # unknown
    'KBA13_SEG_KLEINWAGEN' : ['-1'], # unknown
    'KBA13_SEG_KOMPAKTKLASSE' : ['-1'], # unknown
    'KBA13_SEG_MINIVANS' : ['-1'], # unknown
    'KBA13_SEG_MINIWAGEN' : ['-1'], # unknown
    'KBA13_SEG_MITTELKLASSE' : ['-1'], # unknown
    'KBA13_SEG_OBEREMITTELKLASSE' : ['-1'], # unknown
    'KBA13_SEG_OBERKLASSE' : ['-1'], # unknown
    'KBA13_SEG_SONSTIGE' : ['-1'], # unknown
    'KBA13_SEG_SPORTWAGEN' : ['-1'], # unknown
    'KBA13_SEG_UTILITIES' : ['-1'], # unknown
    'KBA13_SEG_VAN' : ['-1'], # unknown
    'KBA13_SEG_WOHNMOBILE' : ['-1'], # unknown
    'KBA13_SITZE_4' : ['-1'], # unknown
    'KBA13_SITZE_5' : ['-1'], # unknown
    'KBA13_SITZE_6' : ['-1'], # unknown
    'KBA13_TOYOTA' : ['-1'], # unknown
    'KBA13_VORB_0' : ['-1'], # unknown
    'KBA13_VORB_1' : ['-1'], # unknown
    'KBA13_VORB_1_2' : ['-1'], # unknown
    'KBA13_VORB_2' : ['-1'], # unknown
    'KBA13_VORB_3' : ['-1'], # unknown
    'KBA13_VW' : ['-1'], # unknown
    'KKK' : ['-1', '0'], # unknown
    'MOBI_REGIO' : ['0'], # none
    'NATIONALITAET_KZ' : ['-1', '0'], # unknown
    'ONLINE_AFFINITAET' : ['0'], # none
    'ORTSGR_KLS9' : ['-1'], # unknown
    'OST_WEST_KZ' : ['-1'], # unknown
    'PLZ8_ANTG1' : ['-1'], # unknown
    'PLZ8_ANTG2' : ['-1'], # unknown
    'PLZ8_ANTG3' : ['-1'], # unknown
    'PLZ8_ANTG4' : ['-1'], # unknown
    'PLZ8_GBZ' : ['-1'], # unknown
    'PLZ8_HHZ' : ['-1'], # unknown
    'PRAEGENDE_JUGENDJAHRE' : ['-1', '0'], # unknown
    'REGIOTYP' : ['-1', '0'], # unknown
    'RELAT_AB' : ['-1', '9'], # unknown
    'RETOURTYP_BK_S' : ['0'], # unknown
    'SEMIO_DOM' : ['-1', '9'], # unknown
    'SEMIO_ERL' : ['-1', '9'], # unknown
    'SEMIO_FAM' : ['-1', '9'], # unknown
    'SEMIO_KAEM' : ['-1', '9'], # unknown
    'SEMIO_KRIT' : ['-1', '9'], # unknown
    'SEMIO_KULT' : ['-1', '9'], # unknown
    'SEMIO_LUST' : ['-1', '9'], # unknown
    'SEMIO_MAT' : ['-1', '9'], # unknown
    'SEMIO_PFLICHT' : ['-1', '9'], # unknown
    'SEMIO_RAT' : ['-1', '9'], # unknown
    'SEMIO_REL' : ['-1', '9'], # unknown
    'SEMIO_SOZ' : ['-1', '9'], # unknown
    'SEMIO_TRADV' : ['-1', '9'], # unknown
    'SEMIO_VERT' : ['-1', '9'], # unknown
    'SHOPPER_TYP' : ['-1'], # unknown
    'SOHO_FLAG' : ['-1'], # unknown
    'TITEL_KZ' : ['-1', '0'], # unknown
    'VERS_TYP' : ['-1'], # unknown
    'WACHSTUMSGEBIET_NB' : ['-1', '0'], # unknown
    'WOHNDAUER_2008' : ['-1', '0'], # unknown
    'WOHNLAGE' : ['-1'], # unknown
    'W_KEIT_KIND_HH' : ['-1', '0'], # unknown
    'ZABEOTYP' : ['-1', '9'] # unknown
}

def CleanDataBymarkingNaN(df):
    '''
    Clean the dataframe by swapping unknown values for NaN 
    
    INPUT: 
        df: Dataframe to clean
        
    OUTPUT:
        Cleaned dataframe
    '''
    print('CleanDataBymarkingNaN')
    ## 1.2 Swap unknown values with Nan
    i = 0
    for (columnName, columnValues) in missingValuesDictionary.items():
        i += 1
        print(i, 'column header: ', columnName)
        for value in columnValues:
            try:
                df.loc[df[columnName] == int(value), columnName] = np.nan
            except:
                print('value not found: ', value)
    return df

## transform every value to int for two mixed value columns

def CleanupCameo2015(df):
    '''
    Clean the dataframe by swapping the values 'XX' and 'X' with NaN and converting all values to integer for the columns
    'CAMEO_INTL_2015' and 'CAMEO_DEUG_2015'
    
    INPUT: 
        df: Dataframe to clean
        
    OUTPUT:
        Cleaned dataframe
    '''
    print('CleanupCameo2015')
    column_names = ['CAMEO_INTL_2015','CAMEO_DEUG_2015']
    for column_name in column_names:
        df.loc[df[column_name].astype(str).str.contains("XX", na=False), column_name] = np.nan
        df.loc[df[column_name].astype(str).str.contains("X", na=False), column_name] = np.nan
        df.loc[df[column_name].isnull(), column_name] = '-112'
        df[column_name] = df[column_name].astype('int')
        df.loc[df[column_name] == -112, column_name] = np.nan
    
    return df

def CleanupCameo2015Extended(df):
    '''
    Clean the dataframe by swapping the values 'XX' and 'X' with NaN and converting all values to integer for the columns
    'CAMEO_INTL_2015', 'CAMEO_DEUG_2015', 'CAMEO_DEU_2015', 'CAMEO_DEUINTL_2015'
    
    INPUT: 
        df: Dataframe to clean
        
    OUTPUT:
        Cleaned dataframe
    '''
    print('CleanupCameo2015')
    column_names = ['CAMEO_INTL_2015','CAMEO_DEUG_2015', 'CAMEO_DEU_2015', 'CAMEO_DEUINTL_2015']
    for columnName in column_names:
        try:
            df.loc[df[columnName].astype(str).str.contains("XX", na=False), columnName] = np.nan
            df.loc[df[columnName].astype(str).str.contains("X", na=False), columnName] = np.nan
            df.loc[df[columnName].isnull(), column_name] = '-112'
            df[columnName] = df[columnName].astype('int')
            df.loc[df[columnName] == -112, columnName] = np.nan
        except:
            print('Column name not found: ', columnName)
    
    return df

## visualise naturally missing data

def VisualizeNan(df):
    '''
    Visualize the dataframe's NaN distribution by showing a graph of NaN% of each column 
    
    INPUT: 
        df: Dataframe to analyze and visualize
    '''
    print('VisualizeNan')
    (df.isna().mean().round(4) * 100).sort_values(ascending=False)[:50].plot(kind = 'bar', figsize = (20,8), fontsize = 13, xlabel='Column Names', ylabel='NaN %')

## 2. Get distribution of missing values for all columns

def DropColumnsWithHigherNanPercentageThan(df, threshold):
    '''
    Clean the dataframe by dropping columns that have a higher percentage of NaN than a given threshhold 
    
    INPUT: 
        df: Dataframe to clean
        threshold: % that NaN % needs to exceed to fulfill dropping criteria
        
    OUTPUT:
        Cleaned dataframe
    '''
    print('DropColumnsWithHigherNanPercentageThan')
    nanDistribution = df.isna().mean().round(4) * 100
    
    nanDistribution = nanDistribution.sort_values(ascending=False)
    #nanDistribution.plot.bar()
    #rcParams['figure.figsize'] = 500, 50
    #plt.show()
    #display(nanDistribution)

    ## total 38 columns with threshhold% or above nans, lets drop those columns
    dropList = []
    for (columnName, columnValue) in nanDistribution.items():
        if columnValue > threshold:
            print('Column dropped: ', columnName)
            dropList.append(columnName)
            
    try:
        df = df.drop(columns = dropList)
    except:
        print('failed to drop columns')

    return df

## 3. Unique value columns

def PrintUniqueValuesForColumns(df):
    '''
    Print the unique values for each column of the given dataframe 
    
    INPUT: 
        df: Dataframe to analyze
    '''
    print('PrintUniqueValuesForColumns')
    i = 0;
    uniqueValuesDistribution = {}
    for column_name in df.columns.values:
        print(column_name)
        uniqueLength = len(df[column_name].unique())
        uniqueValuesDistribution[column_name] = uniqueLength

        print(i, 'ColumnName: ', column_name, 'UniqueLength', uniqueLength)
        print('Unique Values: ', df[column_name].unique())

        i += 1

## drop columns  due to reasons above
def DropMiscColumns(df, dropColumns):
    '''
    Drop a list of columns for a given dataframe
    
    INPUT: 
        df: Dataframe to process
        dropColumns: List of columns to drop
        
    OUTPUT:
        The processed dataframe
    '''
    print('DropMiscColumns')
    print(dropColumns)
    df = df.drop(columns=dropColumns)
    print(df.shape)
    return df

## 8. Decide about nan rows (drop/give average values etc)
## 8.1 See whats the distribution of NaNs in each row and if we need to drop any

def GetRowNanDistribution(df, threshold):
    '''
    Print the number and % for rows that have NaNs above a given threshhold % for the given dataframe 
    
    INPUT: 
        df: df to process
        threshold: The % for NaNs in a row before it can be counted
    '''
    print('GetRowNanDistribution')
    (rows, columns) = df.shape
    print(df.shape)
    temp = 100/columns
    nanPercentageList = (df.isnull().sum(axis=1)*temp).tolist()

    sortedNanPercentageList = nanPercentageList.copy()
    sortedNanPercentageList.sort(reverse=True)
    total = 0
    for datum in sortedNanPercentageList:
        if (datum >= threshold):
            total += 1
        else:
            break

    print('Total: ', total)
    print('Total % at or above ', str(threshhold),'%: ', (total/rows) * 100)

## add nan percentage column to aid in dropping higher nan rows

def DropNanRowsAboveThreshhold(df, threshold):
    '''
    Drop rows from the dataframe that have a NaN % higher than the given threshhold
    
    INPUT: 
        df: Dataframe to process
        threshold: % that needs to be exceeded before a row is dropped
        
    OUTPUT:
        The processed dataframe
    '''
    print('DropNanRowsAboveThreshhold')
    (rows, columns) = df.shape
    temp = 100/columns
    nanPercentageList = (df.isnull().sum(axis=1)*temp).tolist()

    nanPercentageColumnName = 'nanPercentageColumn'
    nanPercentageColumnFrame = pd.DataFrame({nanPercentageColumnName : nanPercentageList})
    df = pd.concat([df, nanPercentageColumnFrame], axis = 1)

    df = df.loc[df[nanPercentageColumnName] < threshold]
    df = df.drop(columns=[nanPercentageColumnName])
    df.reset_index(drop=True, inplace=True)
    
    return df

## binary encoding

def BinaryEncode_OST_WEST_KZ_VERS_TYP(df):
    '''
    Process the dataframe to binary encode the columns 'OST_WEST_KZ' and 'VERS_TYPE' 
    
    INPUT: 
        df: Dataframe to process
        
    OUTPUT:
        Return the processed dataframe
    '''
    print('BinaryEncode_OST_WEST_KZ_VERS_TYP')
    columnName = 'OST_WEST_KZ'
    df.loc[df[columnName] == 'O', columnName] = 0
    df.loc[df[columnName] == 'W', columnName] = 1

    df.loc[df['VERS_TYP'] == 2, 'VERS_TYP'] = 0
    
    return df

## 8.2 Replace NaNs with most frequently used (binary) or mean (int)
## this needs to go after binary encoding of OST_WEST_KZ and PRODUCT_GROUP

def ReplaceNans(df, binaryColumns, categoricalColumns):
    '''
    Process the dataframe to replace NaN values with medians or most frequently used value 
    
    INPUT: 
        df: Dataframe to process
        binaryColumns: List of columns with binary values
        categoricalColumns: List of columns with categorical values        
        
    OUTPUT:
        The process dataframe
    '''
    print('ReplaceNans')
    i = 0
    for columnName in df.columns:
        print(i, 'ColumnName: ', columnName)
        i += 1
        if columnName in binaryColumns:
            # binary data
            # replace NaN with most frequently used value
            mostFrequentlyUsed = df[columnName].value_counts().idxmax()
            print('most frequently used: ', mostFrequentlyUsed)
            df.loc[df[columnName].isnull(), columnName] = mostFrequentlyUsed
        elif columnName in categoricalColumns:
            #ignore, skip
            print('Skipping due to being categorical')
        else:
            # numerical data
            # replace NaN with median
            median = df[columnName].median(axis = 0, skipna = True) 
            print('median value: ', median)
            df.loc[df[columnName].isnull(), columnName] = median
            
    return df

## splitting some columns
## can **NOT** run this more than once

## input is 21 (int)
## output needs to be 2 and 1 (respectively)
## strategy is to convert to string and get the first character and convert back to int
def GetFirst(intValue):
    '''
    Return the first digit of a two digit integer. Helper function for Split4Columns()
    
    INPUT: 
        intValue: Two digit integer
        
    OUTPUT:
        First digit of the two digit integer
    '''
    intValue = round(int(intValue))
    retVal =  int(str(intValue)[0])
    return retVal
def GetSecond(intValue):
    '''
    Return the second digit of a two digit integer. Helper function for Split4Columns()
    
    INPUT: 
        intValue: Two digit integer
        
    OUTPUT:
        Second digit of the two digit integer
    '''
    intValue = round(int(intValue))
    retVal = int(str(intValue)[1])
    return retVal

def GetFamily(intValue):
    '''
    Returns if the given value represents Family or not. Helper function for Split4Columns()
    
    INPUT: 
        intValue: Integer representing Family or Business
        
    OUTPUT:
        Binary value depending on whether input represents Family
    '''
    intValue = int(round(intValue))
    if intValue <= 4:
        return 1.0
    elif intValue >= 5:
        return 0.0
    else:
        print('Error in GetFamily: ', intValue)
def GetBusiness(intValue):
    '''
    Returns if the given value represents Business or not. Helper function for Split4Columns()
    
    INPUT: 
        intValue: Integer representing Family or Business
        
    OUTPUT:
        Binary value depending on whether input represents Family
    '''
    intValue = int(round(intValue))
    if intValue <= 4:
        return 0.0
    elif intValue >= 5:
        return 1.0
    else:
        print('Error in GetBusiness: ', intValue)
        
def GetRural(intValue):
    '''
    Returns if the given value represents Rural or not. Helper function for Split4Columns()
    
    INPUT: 
        intValue: Integer representing Rural or not
        
    OUTPUT:
        Binary value depending on whether input represents Rural
    '''
    intValue = int(round(intValue))
    if intValue <=5:
        return 0.0
    elif intValue >= 7:
        return 1.0
    else:
        print('Error in GetRural: ', intValue)

def GetGeneration(intValue):
    '''
    Return the Generation (40's, 50's,..90's) based on the input where multiple input maps to the same Generation.
    Helper function for Split4Columns()
    
    INPUT: 
        intValue: Integer representing Generation
        
    OUTPUT:
        Integer representing Generation
    '''
    intValue = int(round(intValue))
    if intValue <= 2:
        return 1.0
    elif intValue <= 4:
        return 2.0
    elif intValue <= 7:
        return 3.0
    elif intValue <= 9:
        return 4.0
    elif intValue <= 13:
        return 5.0
    elif intValue <= 15:
        return 6.0
    else:
        print('Error in GetGeneration: ', intValue)
def GetMovement(intValue):
    '''
    Return the Movement (Mainstream, Avantagrde) based on the input where multiple input maps to the same Movement.
    Helper function for Split4Columns()
    
    INPUT: 
        intValue: Integer representing Movement
        
    OUTPUT:
        Integer representing Movement
    '''
    ms = [1,3,5,8,10,12,14]
    ag = [2,4,6,7,9,11,13,15]

    intValue = int(round(intValue))
    if intValue in ms:
        return 0.0
    elif intValue in ag:
        return 1.0
    else:
        print('Error in GetMovement: ', intValue)

def Split4Columns(df):
    '''
    Process the dataframe to split 4 specific columns: 'CAMEO_INTL_2015', 'PRAEGENDE_JUGENDJAHRE', 'PLZ8_BAUMAX', 'WOHNLAGE'
    
    INPUT: 
        df: Dataframe to process
        
    OUTPUT:
        Processed dataframe
    '''
    print('Split4Columns')
    (row, column) = df.shape
    
    ## 1. CAMEO_INTL_2015
    columnName = 'CAMEO_INTL_2015'

    newColumnName = 'CI2_WealthType'
    if newColumnName not in df.columns:
        dummyColumnValues = [0] * row
        dummyColumn = pd.DataFrame({newColumnName: dummyColumnValues})
        df = pd.concat([df, dummyColumn], axis = 1)
        df[newColumnName] = df.apply(lambda row: GetFirst(row[columnName]), axis=1)

    newColumnName = 'CI2_FamilyType'
    if newColumnName not in df.columns:
        dummyColumnValues = [0] * row
        dummyColumn = pd.DataFrame({newColumnName: dummyColumnValues})
        df = pd.concat([df, dummyColumn], axis = 1)
        df[newColumnName] = df.apply(lambda row: GetSecond(row[columnName]), axis=1)

    df = df.drop(columns=[columnName])

    
    ## 2. PRAEGENDE_JUGENDJAHRE
    columnName = 'PRAEGENDE_JUGENDJAHRE'

    newColumnName = 'PJ_Movement'
    if newColumnName not in df.columns:
        dummyColumnValues = [0] * row
        dummyColumn = pd.DataFrame({newColumnName: dummyColumnValues})
        df = pd.concat([df, dummyColumn], axis = 1)
        df[newColumnName] = df.apply(lambda row: GetMovement(row[columnName]), axis=1)

    newColumnName = 'PJ_Generation'
    if newColumnName not in df.columns:
        dummyColumnValues = [0] * row
        dummyColumn = pd.DataFrame({newColumnName: dummyColumnValues})
        df = pd.concat([df, dummyColumn], axis = 1)
        df[newColumnName] = df.apply(lambda row: GetGeneration(row[columnName]), axis=1)

    df = df.drop(columns=[columnName])
    

    ## 3. PLZ8_BAUMAX
    columnName = 'PLZ8_BAUMAX'

    newColumnName = 'PB_Family'
    if newColumnName not in df.columns:
        dummyColumnValues = [0] * row
        dummyColumn = pd.DataFrame({newColumnName: dummyColumnValues})
        df = pd.concat([df, dummyColumn], axis = 1)
        df[newColumnName] = df.apply(lambda row: GetFamily(row[columnName]), axis=1)

    newColumnName = 'PB_Business'
    if newColumnName not in df.columns:
        dummyColumnValues = [0] * row
        dummyColumn = pd.DataFrame({newColumnName: dummyColumnValues})
        df = pd.concat([df, dummyColumn], axis = 1)
        df[newColumnName] = df.apply(lambda row: GetBusiness(row[columnName]), axis=1)

    # not dropping since we still have some value for mainly x-y family homes
    #df = df.drop(columns=[columnName])
    

    ## 4. WOHNLAGE
    columnName = 'WOHNLAGE'

    newColumnName = 'WL_Rural'
    if newColumnName not in df.columns:
        dummyColumnValues = [0] * row
        dummyColumn = pd.DataFrame({newColumnName: dummyColumnValues})
        df = pd.concat([df, dummyColumn], axis = 1)
        df[newColumnName] = df.apply(lambda row: GetRural(row[columnName]), axis=1)

    # not dropping since we still have some value for neighbourhood type
    #df = df.drop(columns=[columnName])

    return df

## convert to int

def ConvertToInt(df):
    '''
    Process the dataframe convert all values to integer
    
    INPUT: 
        df: Dataframe to process
        
    OUTPUT:
        Processed dataframe
    '''
    print('ConvertToInt')
    return df.astype(int)

## Removing outliers
## Ref: https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-data-frame

def RemoveOutliers(df, columnsToExclude, threshold):
    '''
    Process the dataframe to drop rows that have any outliers in their columns
    
    INPUT: 
        df: Dataframe to process
        columnstoExceed: Columns to avoid searching for outliers
        threshold: Delta that needs to exceed for a difference between [row][column] value and
                    Standard Deviation of column values before a row is dropped
        
    OUTPUT:
        Processed dataframe
    '''
    print('RemoveOutliers')
    columnNames = [item for item in list(df.columns.values) if item not in columnsToExclude]
    print(df.shape)

    # keep ones within +6 to -6 sd in each column
    for column in columnNames:
        df = df[np.abs(df[column] - df[column].mean()) <= (threshold * df[column].std())]
    
    print(df.shape)
    return df

## Feature Scaling

def ScaleFeature(df):
    '''
    Process the dataframe by standardizing features by removing the mean and scaling to unit variance
    
    INPUT: 
        df: Dataframe to process
        
    OUTPUT:
        Processed dataframe and scaler
    '''
    print('ScaleFeature')
    scaler = StandardScaler()
    scaledDf = pd.DataFrame(scaler.fit_transform(df))

    return (scaledDf, scaler)

def ScaleFeatureWithScaler(df, scaler):
    '''
    Process the dataframe by standardizing features by removing the mean and scaling to unit variance
    
    INPUT: 
        df: Dataframe to process
        scaler: Scaler to transform with 
        
    OUTPUT:
        Processed dataframe
    '''
    print('ScaleFeatureWithScaler')
    return pd.DataFrame(scaler.transform(df))

# one hot encoding
def HotEncodeColumns(df, columnsToEncode):
    '''
    One hot encode dataframe columns
    
    INPUT: 
        df: Dataframe to process
        columnsToEncode: columns that need to be hot encoded
        
    OUTPUT:
        Processed dataframe
    '''    
    print('HotEncodeColumns')
    print(df.shape)
    for (columnName, prefixName) in columnsToEncode.items():
        dfDummies = pd.get_dummies(df[columnName], prefix = prefixName)
        df = pd.concat([df, dfDummies], axis=1)
        df = df.drop(columns=[columnName])
    
    print(df.shape)
    return df

# drop low variance columns
# ref: https://stackoverflow.com/questions/39812885/retain-feature-names-after-scikit-feature-selection
def varianceThresholdDropper(df, threshold = 0.5):
    '''
    Drop columns with low variance
    
    INPUT: 
        df: Dataframe to process
        threshold: variance threshold for dropping columns
        
    OUTPUT:
        Processed dataframe
    '''
    selector = VarianceThreshold(threshold)
    selector.fit(df)
    return df[df.columns[selector.get_support(indices = True)]]
