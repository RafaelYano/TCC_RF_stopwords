from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


import pickle
import os
import numpy as np
import pandas as pd




#Classes customizadas
import sqlquery
import clean_text




#Treino e salvo os modelos da ML
class train_model():


    def SaveModel(classdata, fam, model):
        thispath = os.path.dirname(os.path.abspath(__file__))
        filename = thispath + "\\MODELS\\MODEL_" + classdata +"_"+ fam+ ".sav"
        pickle.dump(model, open(filename, 'wb'))
   
    #Utiliza o GrindSearchCV para definir os melhores parametros
    def DefineParameters(X_train_DESC, y_train, seed):
        rf = RandomForestClassifier()
        print("Testa hyper parametros")
        #Parametros a serem testados: numero de estimadores e profundidade da arvore
        parameters = {
            'n_estimators': [35, 55, 105, 155, 205],
            'max_depth': [30, 50, 100, 150, 200, 300, None],
            'min_samples_split': [2,5]
        }


        cv = RandomizedSearchCV(rf,parameters,cv=2, n_iter = 30, random_state=seed)
        cv.fit(X_train_DESC,y_train)
        return list(cv.best_params_.values())
   
    #Consulta o RCP para pegar os dados de inter
    def GetData():
        return sqlquery.ImportRcp.query_data()
   
    def PrepareData(inter, fam, y_value, x_value):


        #Description:
        row_num = 4
        thispath = os.path.dirname(os.path.abspath(__file__))
        filename = thispath + "\\listofwords.xlsx"
        correct = pd.read_excel(filename)


        for row in inter.itertuples():
            #print(row[4]) #- PROBLEM

            #Remove Stop Words
            inter.loc[row[0], x_value] = clean_text.clean_text.clearDesc_list(row[row_num], correct)
        

        #Preenche os dados vazios, do contrário o código não funciona
        fc = inter.replace([np.inf, -np.inf], np.nan).fillna(99999)


        y_unique = fc.groupby(fc[y_value]).size().reset_index(name ="count")
        y_unique = y_unique[y_unique["count"] > 1]
        y_unique = list(y_unique[y_value])


        fc = fc[fc[y_value].isin(y_unique)]


        #Separa a base de teste e treino, 80-20
        fc_train, fc_test, y_train, y_test = train_test_split(fc[x_value], fc[y_value], test_size= 0.2, stratify= fc[y_value])


        #Transforma os textos para serem usados no treinamento
        #Treino Desc
        count_vect = CountVectorizer(binary=True)
        x_train_counts = count_vect.fit_transform(fc_train)
        tf_transformer = TfidfTransformer(use_idf=True).fit(x_train_counts)
        X_train_DESC = tf_transformer.transform(x_train_counts)


        #Teste
        #Desc
        X_test_DESC = count_vect.transform(fc_test)


        #Salvando countVectorize
        train_model.SaveModel("CountVectorize" + x_value, fam, count_vect)
        #Salvando tf transformer
        train_model.SaveModel("TfTransformer" + x_value, fam, tf_transformer)


        return X_train_DESC, X_test_DESC, y_train, y_test
   
       
    def TrainModelResponsability():
        #Dados a serem utilizados
        print("Treina Technical Responsability")
        inter = train_model.GetData()


        inter = inter[inter["Chargeable"] != "C"]

           
        #Preparação dos dados
        dados = inter[inter["Family"] == fam]
        X_train, X_test, y_train, y_test = train_model.PrepareData(dados, fam, "Technical Responsability", "Clear Desc")


        #Treina o Random Forest
        #Parametros para treinar o modelo
        seed = np.random.randint(42)


        parameters = train_model.DefineParameters(X_train, y_train, seed)
        #parameters = [55, 2, 100] #Pula a etapa de otimização de parametros
        print(parameters)


        rf = RandomForestClassifier(max_depth= parameters[2], n_estimators= parameters[0], min_samples_split=parameters[1])
        rf.fit(X_train, y_train)


        #Salvando o modelo treinado
        train_model.SaveModel("Responsability", fam, rf)


        #Classifica os dados de teste e salva as informações da qualidade do modelo
        y_test_predict = rf.predict(X_test)


        teste = pd.DataFrame(columns=["Acuracia", "Precisao", "f1", "recall"])


        teste.loc[1,"Acuracia"] = accuracy_score(y_test, y_test_predict).item()
        teste.loc[1,"Precisao"] = precision_score(y_test, y_test_predict, average='macro' ).item()
        teste.loc[1,"f1"] = f1_score(y_test, y_test_predict, average='macro' ).item()
        teste.loc[1,"recall"] = recall_score(y_test, y_test_predict, average='macro' ).item()


        teste.to_excel("Qualidade_Modelo_TechnicalResponsability_listNLTK" + fam+".xlsx", index=False)


       
    def TrainModelChargeability():
        #Dados a serem utilizados
        print("Treina Chargeability")
        inter = train_model.GetData()

           
        #Preparação dos dados
        dados = inter[inter["Family"] == fam]
        X_train, X_test, y_train, y_test = train_model.PrepareData(dados, fam, "Chargeable", "Clear Desc_c")


        #Treina o Random Forest
        #Parametros para treinar o modelo
        seed = np.random.randint(42)


        parameters = train_model.DefineParameters(X_train, y_train, seed)
        #parameters = [55, 2, 100] #Pula a etapa de otimização de parametros
        print(parameters)


        rf = RandomForestClassifier(max_depth= parameters[2], n_estimators= parameters[0], min_samples_split=parameters[1])
        rf.fit(X_train, y_train)


        #Salvando o modelo treinado
        train_model.SaveModel("Chargeable", fam, rf)


        #Classifica os dados de teste e salva as informações da qualidade do modelo
        y_test_predict = rf.predict(X_test)


        teste = pd.DataFrame(columns=["Acuracia", "Precisao", "f1", "recall"])


        teste.loc[1,"Acuracia"] = accuracy_score(y_test, y_test_predict).item()
        teste.loc[1,"Precisao"] = precision_score(y_test, y_test_predict, average='macro' ).item()
        teste.loc[1,"f1"] = f1_score(y_test, y_test_predict, average='macro' ).item()
        teste.loc[1,"recall"] = recall_score(y_test, y_test_predict, average='macro' ).item()


        teste.to_excel("Qualidade_Modelo_TechnicalChargeable" + fam+".xlsx", index=False)
