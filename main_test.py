import unittest
import os
import pandas as pd
import numpy as np
from Datenanalyseprogramm import Datensatz, TrainSatz, TestSatz, IdealSatz



    
    
    
class TestDatensatz(unittest.TestCase):
    
    def test_loadData(self):
        path_to_csv_ideal = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/ideal.csv'
        datensatz_ideal = Datensatz(path_to_csv_ideal, pd)
        self.assertTrue(os.path.exists(datensatz_ideal.path_to_csv))

        path_to_csv_test = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/test.csv'
        datensatz_test = Datensatz(path_to_csv_test, pd)
        self.assertTrue(os.path.exists(datensatz_test.path_to_csv))
        
        path_to_csv_train = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/train.csv'
        datensatz_train = Datensatz(path_to_csv_train, pd)
        self.assertTrue(os.path.exists(datensatz_train.path_to_csv))


class TestTrainSatz(unittest.TestCase):
    
    id_funk_pfadTest        = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/ideal_test.csv'
    train_funk_pfadTest     = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/train_test.csv'
    test_funk_pfadTest      = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/test_test.csv'
    idealeFunktionenTest    = IdealSatz(id_funk_pfadTest, pd)
    trainFunktionenTest     = TrainSatz(train_funk_pfadTest, pd)
    valFunktionenTest           = TestSatz(test_funk_pfadTest, pd)
    ideal_dfTest           = idealeFunktionenTest.loadData()
    train_dfTest           = trainFunktionenTest.loadData()
    test_dfTest            = valFunktionenTest.loadData()
        
    id_funk_pfad        = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/ideal.csv'
    train_funk_pfad     = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/train.csv'
    test_funk_pfad      = 'C:/Users/Michael/OneDrive/IU Fernuni/IU Python/Spyder Quellcode/test.csv'
    idealeFunktionen    = IdealSatz(id_funk_pfad, pd)
    trainFunktionen     = TrainSatz(train_funk_pfad, pd)
    valFunktionen       = TestSatz(test_funk_pfad, pd)
    ideal_df           = idealeFunktionen.loadData()
    train_df           = trainFunktionen.loadData()
    test_df            = valFunktionen.loadData()

    
    def test_train(self):
       trainErgebnisTest    = TestTrainSatz.trainFunktionenTest.train(self.ideal_dfTest)
       trainErgebnis        = TestTrainSatz.trainFunktionen.train(self.ideal_df)
       erwarteteSpalten = []
       eigentlicheSpalten = []
       for col in trainErgebnisTest:
           erwarteteSpalten.append(col)
          
       for col in trainErgebnis:
           eigentlicheSpalten.append(col)
           
       self.assertEqual(erwarteteSpalten,eigentlicheSpalten)
       self.assertNotEqual(trainErgebnis.empty,True)
       
        
    
    def test_findbestFits(self):
        
       trainErgebnisTest    = TestTrainSatz.trainFunktionenTest.train(self.ideal_dfTest)
       bestenFunktionenTest = TestTrainSatz.trainFunktionenTest.findbestFits(trainErgebnisTest)

       trainErgebnis        = TestTrainSatz.trainFunktionen.train(self.ideal_df)
       bestenFunktionen     = TestTrainSatz.trainFunktionen.findbestFits(trainErgebnis)
       self.assertEqual(bestenFunktionen,bestenFunktionenTest)



class TestTestSatz(unittest.TestCase):
        
    def test_validieren(self):
        print("I made it here")




if __name__ == "__main__":
    unittest.main()