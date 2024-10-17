import clean_text
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


class classify_data():


    def PrepareData(inter, fam, thispath, xvalue):
        correct = pd.read_excel("listofwords.xlsx", na_filter= False)
       
        count_vect = pickle.load(open(thispath + "\\MODELS\\MODEL_CountVectorize" + xvalue+ "_"+ fam + ".sav", "rb"))
        tf_transformer = pickle.load(open(thispath + "\\MODELS\\MODEL_TfTransformer" + xvalue+ "_"+ fam + ".sav", "rb"))

        for row in inter.itertuples():
            #print(row[8]) #- DESC
            inter.loc[row[0], "Clear_Desc_action"] = clean_text.clean_text.clearDesc_list(row[8], correct)


        #Transforma os textos para serem usados no treinamento
        #Description
        x_test_counts = count_vect.transform(inter["Clear_Desc_action"])
        X_test_DESC = tf_transformer.transform(x_test_counts)

        return X_test_DESC
   
    def ClassifyResponsability():    
        thispath = os.path.dirname(os.path.abspath(__file__))


        inter = pd.read_csv(thispath + "\\RAW\\" + "score.csv", sep=';')
        inter = inter.replace([np.inf, -np.inf], np.nan).fillna("EMBRAER COMMENTS FIELD NOT INFORMED")


        inter['Desc_Action'] = inter['PROBLEM'] + inter["ACTION"]


        x_desc = classify_data.PrepareData(inter, fam, thispath, "Clear Desc")
        x_desc_c = classify_data.PrepareData(inter, fam, thispath, "Clear Desc_c")


        clf_char =  pickle.load(open(thispath + "\\MODELS\\MODEL_Chargeable_"+ fam + ".sav", "rb"))
        clf_ex = pickle.load(open(thispath + "\\MODELS\\MODEL_Responsability_"+ fam + ".sav", "rb"))


        inter["Exclusion"] = clf_ex.predict(x_desc)

        inter['Chargeable'] = clf_char.predict(x_desc_c)

        inter.reset_index(inplace=True, drop=True)


        for i in range(len(inter)):
            if inter.loc[i,'Chargeable'] == 'N':
                inter.loc[i, 'Technical_Responsability'] = inter.loc[i, "Exclusion"]
            else:
                inter.loc[i, 'Technical_Responsability'] = inter.loc[i, "Chargeable"]
       
        inter.drop(columns=['Chargeable', 'Exclusion'], inplace=True)
    
        inter.to_excel(thispath + "\\LABELED\\" +"LABELED.xlsx", index=False)    
