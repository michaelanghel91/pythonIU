"""Dieses Programm wurde für den Kurs 'PROGRAMMIEREN MIT PYTHON' von Michael Anghel entwickelt.
Datum: Oktober/November 2023
Zielsetzung: Mit der Hilfe von vier Trainingsfunktionen werden vier ideale Funktionen aus einem anderen Datensatz herausselektiert auf Basis bestimmter Kriterien.  
"""

import pandas as pd
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import sqlite3



class Datensatz:
    """
    Diese Klasse dient als Parent-Klasse für die drei unteren Klassen: Test, Ideal, Train. 
    Alle gemeinsamen Funktionen die sich diese drei Klassen teilen, werden im Rahmen dieser Klasse definiert.
    """
    def __init__(self,path_to_csv,pd_object):
        self.path_to_csv = path_to_csv
        self.pd_object = pd_object
        
        
    def loadData(self):
        """
        Alle Klassen der Datensätze müssen mittels der Panda-Bibliothek aus einer CSV Datei ein DataFrame-Objekt generieren. Somit wird diese Methode in der Parent-Klasse definiert.
        """
        
        try:
            return self.pd_object.read_csv(self.path_to_csv)
        except Exception as ex:
            sys.exit("Die CSV Datei konnte nicht geladen werden und der Code wurde abgebrochen")


class IdealSatz(Datensatz):
    def __init__(self,path_to_csv,pd_object):
        super().__init__(path_to_csv,pd_object)
        


class TrainSatz(Datensatz):
    def __init__(self,path_to_csv,pd_object):
        super().__init__(path_to_csv,pd_object)
        
        
    def train(self, IdealSatz):
        """
        Diese Methode sammelt die Train-Funktionen die den idealen Funktionen nahe sind und gibt diese Sammlung als Panda-DataFrame-Objekt zurück.
        """
        t_set = super().loadData()
        i_set = IdealSatz
        trainErgebnis = pd.DataFrame()
        for column in t_set:
            if(column == "x"): next
            else:
                trainErgebnis[f"train_{column}"] = self.findClosestFunctions(super().loadData()[column],i_set)
                
        if 0: print(trainErgebnis)
        return trainErgebnis 

           
    def findClosestFunctions(self,train_set_spalte,ideal_set):
        """
        Vergleicht die Train- und ideale Funktion. Es werden äquivalente Punkte subtrahiert und anschließend entlang der ganzen Spalte summiert und quadriert. Desto kleiner der Wert dieses Rechenvorgangs, desto näher die Funktionen.
        """
        comp_df = pd.DataFrame() 
        pdresults = {} 
        results = {} 
        for func in ideal_set:
            if(func == "x"): next
            else:
                comp_df[f"{func}_id"] = (ideal_set[func] - train_set_spalte)*(ideal_set[func] - train_set_spalte)
                results[f"{func}_id"] = int((comp_df[f"{func}_id"].sum())**2)  
        
        pdresults = pd.DataFrame.from_dict(results,orient='index',columns=['diff']).sort_values('diff')
        if 0: print(pdresults.describe())
        if 0: print(type(pdresults))
        return pdresults
    
    
    def findbestFits(self, trainErgebnis):
        """
        Nachdem die besten Fits gefunden wurden, selektiert diese Funktionen die besten Kanditaten.
        """
        corresp_table = {}
        for col in trainErgebnis:
            trainErgebnis[col] = pd.to_numeric(trainErgebnis[col],errors='coerce')

        for column in trainErgebnis:
            if(column == "x"): next
            else:
                corresp_table[column] = trainErgebnis[column].idxmin()
                
        return corresp_table

############################################################################################################################
class TestSatz(Datensatz):
    """
    Diese Klasse beinhalte alle Funktionen des Testdatensatzes. Dieser wird benutzt um die Selektion der besten Fits zu validieren.
    """
    def __init__(self,path_to_csv,pd_object):
        super().__init__(path_to_csv,pd_object)
        pass
        
    def validieren(self, berechnungen, ergebnis,ideal_df,test_df, train_df):
        """
        Die Validierungsfunktion. Sie ruft andere Funktionen die benötigt sind um die Validierungsvorgänge auszuführen. Ebenfalls sammelt sie alle Variabeln die benötigt sind und gibt diese weiter.
        """
        schnittmenge_df = ideal_df.merge(test_df, on='x', how='inner')
        entriesToKeep = []
        for col,row in ergebnis.items():
            entriesToKeep.append(row.split("_")[0])
        
        entriesToKeep.append('x')
        entriesToKeep.append('y')
        for col in schnittmenge_df:
            if col not in entriesToKeep:             
                schnittmenge_df = schnittmenge_df.drop(col,axis=1)
            
        schnittmenge_df = round(schnittmenge_df,2)
        berech = pd.DataFrame()
        berech['x'] = schnittmenge_df['x']
        
        for col in schnittmenge_df:
            if((col != 'x') and (col != 'y')):
                label = col + " - " + "y"
                funk = col.split("_")[0]
                berech[label] = (schnittmenge_df[funk] - schnittmenge_df['y'])
             
        valGraph_ds  = self.passendeBereicheHervorheben(berech,ideal_df,train_df,ergebnis,test_df)
        self.graphenZeichnen(schnittmenge_df, valGraph_ds,test_df,train_df,ergebnis,ideal_df)
        
        
    def abweichungVonIdeal(self,x,ideal_df,train_df,idF,trF):
        """
        Gibt Abweichungen für x,y Wertepaare zwischen einer gegebener idealen und Trainingsfunktion.
        """
        gefundenID = ideal_df['x'].isin([x])
        index = gefundenID[gefundenID].index[0]
        idY = ideal_df.loc[index, idF]
        trY = train_df.loc[index, trF]
        ratio = round(idY/trY,2)
        maxAbw = round(ratio * math.sqrt(2),2)
        if 0: print(f"Für x_val={x} ist y_id={idY} und y_tr={trY}/Abw: {ratio}")
        return maxAbw 
    
    def passendeBereicheHervorheben(self,berech,ideal_df,train_df,ergebnis,test_df): 
        """
        Jeder Punkt im Testdatensatz wird analysiert. Falls er zu einer idealen Funktion passt, wird er in einer Variabel gespeichert. Später werden nur diejenige Punkte visualisiert, die tatsächlich einer idealen Funktion nahe standen.
        """
        ratio = pd.DataFrame()
        found_ideal_functions = []
        matches_col_names = []
        valGraph_ds = pd.DataFrame(columns=['x','y','idF'])
        
        for key,val in ergebnis.items():
            idF = val.split("_")[0]
            trF = key.split("_")[1]
            colH = idF + "/" + trF
            found_ideal_functions.append(idF)
            matches_col_names.append(idF + "T")
            ratio[colH] = ideal_df[idF] / train_df[trF]
        
        found_ideal_functions.append("x")
        matches_col_names.append('x')
        matches_col_names.append('y')
        ideal_df = ideal_df[found_ideal_functions]
        ratio = ratio.round(2)    
        
        for index, row in test_df.iterrows():
            x = row['x'] # x,y Wertepaare aus Testdatensatz
            y = row['y']
            
            if 0: print(f"##########Außenschleife, Punkt x::y {x}::{y}##########")
            for key,val in ergebnis.items():
                idF = val.split("_")[0]
                trF = key.split("_")[1]
                if 0: print(f"idF: {idF} und trainF {trF}")
                rowMatch = ideal_df['x'].isin([x])
                if rowMatch.any():
                    index = rowMatch[rowMatch].index[0]
                    if 0: print(f"Value {x} found at index {index} ")
                    entYWert = ideal_df.loc[index, idF]
                    ValAbw = abs(round(y/entYWert,2))
                    IdeAbw = self.abweichungVonIdeal(x,ideal_df,train_df,idF,trF)
                    faktor = math.sqrt(2)
                    
                    maxAbs = round(IdeAbw * faktor,2)
                    minAbs = round(IdeAbw / faktor,2)
                    erf1 = ValAbw < maxAbs
                    erf2 = ValAbw > minAbs
                    erf = erf1 and erf2
                    if 0: print(f"Für x_ideal={x} ist y_ideal_{idF}= {entYWert} // Abweichung: {ValAbw}")
                    if 0: print(f"Abweichungen ideal:{IdeAbw} und test:{ValAbw}")
                    if 0: print(f"Max&Min erlaubte Abweichungen wären: {maxAbs} // {minAbs}" )
                    if erf:
                        if 0: print(f"ja für y_ideal_{idF}")
                        
                        data_row = { 'x':x, 'y':y, 'idF':idF }
                        data_row_ds = pd.DataFrame([data_row])
                        valGraph_ds = valGraph_ds.dropna(axis=1, how='all')
                        data_row_ds = data_row_ds.dropna(axis=1, how='all')
                        valGraph_ds = pd.concat([valGraph_ds,data_row_ds],axis=0,ignore_index=True)
                        valGraph_ds = valGraph_ds.astype(valGraph_ds.dtypes)
                        
        valGraph_ds = valGraph_ds.sort_values(by='x')
        if 0: 
            print(valGraph_ds)
            condi = valGraph_ds['idF'] == 'y11'
            print(valGraph_ds[condi])
            
        return valGraph_ds
       
    
    def graphenZeichnen(self, schnittmenge, valGraph_ds,test_df,train_df,ergebnis,ideal_df):
        """
        Diese Methode erhält die Ergebnisse aller bisherigen Methoden und erzeugt entsprechende Graphen.
        """
        # xy = pd.DataFrame()
        ergebnis = {value: key for key, value in ergebnis.items()}
        entriesToKeep = {}
        spezVal = pd.DataFrame()
        for key,value in ergebnis.items():
            entriesToKeep[key.split("_")[0]] = value.split("_")[-1]
           
        if 0: print(valGraph_ds)
        if 0: print(entriesToKeep)    
        if 0: 
            exelPfad = r'C:\Users\Michael\OneDrive\IU Fernuni\IU Python\valGraph_ds.xlsx'
            valGraph_ds.to_excel(exelPfad, index=True) 

        
        for key,value in entriesToKeep.items():
            idF = key
            trF = value
            if 0: print(f'ideale Funktion: {idF} und Trainings-: {trF}')
            
            spezVal = valGraph_ds[valGraph_ds['idF'] == idF]
            if 0: print(spezVal)
            if 0: print(spezVal['idF'])
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])            
            
            ax.plot(ideal_df['x'], ideal_df[idF], label=f"{idF}-ideal")  
            ax.plot(train_df['x'], train_df[trF], label=f"{trF}-train") 
            ax.plot(spezVal['x'], spezVal['y'], label=f"Übereinstimmung mit {idF}") 
            
            ax.legend()
            ax.set_title(f'Id. Funk. {idF}')
            
            
        if 0: # das alles brauch ich nicht..ich mach keine Schnittmenge
            for col in schnittmenge:
                if((col != 'x') and (col != 'y')):
                    xy = schnittmenge[['x','y',col]]
                    xy = schnittmenge[['x','y',col]]
                    xy = xy[valGraph_ds[f'{col} - y']]
                    
                    fig = plt.figure()
                    ax = fig.add_axes([0,0,1,1])
                    
                    ax.plot(xy['x'], xy[col], label=f"{col}-ideal")
    
                    ax.plot(xy['x'], xy['y'], label="y-test gegeb y-val")
                    ax.legend()
                    ax.set_title(f'Id. Funk. {col} gegen Validierungssatz');
                    
                    ax2 = fig.add_axes([0.4, 0.65, 0.3, 0.3])
                    x = train_df['x']
                    y = train_df[entriesToKeep[col]]
                    ax2.plot(x, y, label=f"{entriesToKeep[col]} train")
                    ax2.legend()
                
        #plt.show()
        return None
        
    
    

class ProgrammStart:
    
    id_funk_pfad        = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/ideal.csv'
    train_funk_pfad     = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/train.csv'
    test_funk_pfad      = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/test.csv'
    
    
    def __init__(self):
        self.idealeFunktionen    = IdealSatz(self.id_funk_pfad, pd)
        self.trainFunktionen     = TrainSatz(self.train_funk_pfad, pd)
        self.valFunktionen       = TestSatz(self.test_funk_pfad, pd)
        self.ideal_df           = self.idealeFunktionen.loadData()
        self.train_df           = self.trainFunktionen.loadData()
        self.test_df            = self.valFunktionen.loadData()
        
        


    def datenbankHandler(self,funk_label):
        ##################################################
        num_rows = 10
        num_cols = 5
        # Create a DataFrame with random data
        data = np.random.rand(num_rows, num_cols)  # You can use np.random.randint() for integer data
        columns = ['Column_{}'.format(i) for i in range(num_cols)]
        df = pd.DataFrame(data, columns=columns)
        # Display the random DataFrame
        print(funk_label)
        ##################################################
        datenbak = 'datenbank.db'
        try:
            verbi = sqlite3.connect(datenbak)
            cursor = verbi.cursor()
            df.to_sql('IUPython', verbi, if_exists='replace', index=False)
            verbi.commit()
            verbi.close()
        except sqlite3.Error as er:
            raise SQLiteError(f'Fehler beim Aufbau der SQLite Datei {er}')
            
    
    def datenbankSpeichern(self,funk_label):
        try:
            self.datenbankHandler(funk_label)
        except self.SQLiteError as e:
            print(f"f'Fehler beim Aufbau der SQLite Datei {e}'")
            
            
    def datenbankInhalteAbrufen(self):
        datenbak = 'datenbank.db'
        try:
            verbi = sqlite3.connect(datenbak)
            query = "SELECT * FROM IUPython"
            df = pd.read_sql_query(query,verbi)
            verbi.close()
            return df
        except sqlite3.Error as er:
            raise SQLiteReadError(f'Fehler beim Lesen der SQLite Datei {er}')
    
    def berechnungenDurchführen(self):        
        trainErgebnis = self.trainFunktionen.train(self.ideal_df)
        
        if 0: print(trainErgebnis)
        
        self.bestenFunktionen = self.trainFunktionen.findbestFits(trainErgebnis)
        
        if 0: print(self.bestenFunktionen)
        
        self.valFunktionen.validieren(trainErgebnis,self.bestenFunktionen,self.ideal_df, self.test_df,self.train_df)
          
        
    def wasSindDieBestenFunktionen(self):
        return self.bestenFunktionen
        
        
        

Programm = ProgrammStart()
Programm.berechnungenDurchführen()


 try:
    Programm.datenbankHandler(Programm.ideal_df)
    Programm.datenbankHandler(Programm.train_df)
    Programm.datenbankHandler(Programm.test_df)
    Programm.datenbankHandler(Programm.ERGEBNISSEEEEEEEEEEEEEEEEE)
    db_inhalte = Programm.datenbankInhalteAbrufen()
except (SQLiteError, SQLiteReadError)  as e:
    print(f"f'Fehler beim komminizieren mit der der SQLite Datei: {e}'")


    
    
""" 
    Programm.datenbankSpeichern(Programm.bestenFunktionen) # speichere ideale Funk
    Programm.datenbankSpeichern(Programm.bestenFunktionen) # speichere test Funk
    Programm.datenbankSpeichern(Programm.bestenFunktionen) # speichere train Funk
    Programm.datenbankSpeichern(Programm.bestenFunktionen) # speichere gefundene Funk
"""