#Classe para baixar os dados de imterrupções do RCP
import pandas as pd
import oracledb
from sqlalchemy import create_engine, text
from nltk.tokenize import word_tokenize
import re




 #Consulta SQL
def sql_inter():
        return(
            '''
            SELECT
                    a1 AS "Problem",   
                    a4 AS "Description"




                FROM
                    eventos_falahs wp
                    WHERE 1 = 1  
                    AND data >= add_months(sysdate, -72)
                    AND data <= add_months(sysdate, -10)
           
            '''
        )


def all_upper(stopwords):
    return [x.upper() for x in stopwords]
   
 
  
oracledb.init_oracle_client()
engine = create_engine('')


sqlInter = sql_inter()
print ('EXECUTA CONSULTA')


       
#Database query
dfquery = pd.read_sql(sql = text(sqlInter), con = engine.connect())


print('fim consulta')
listofwords = pd.DataFrame(columns=['palavra', 'quantidade'])
print('percorre valores')


for i in range(len(dfquery)):
    x = dfquery.loc[i,'Description']
    x = re.sub(r"[^a-z A-Z]+", "", x)


    # Separate the descriptions words
    tokenize = word_tokenize(x)
    tokenize = all_upper(tokenize)


    # Go through the list of words searching for the stop words


    for w in tokenize:
        if not listofwords['palavra'].str.contains(w).any():
            listofwords.loc[len(listofwords)+1, 'palavra'] = w
            listofwords.loc[len(listofwords), 'quantidade'] = 1


        else:
            listofwords.loc[listofwords['palavra'] == w, 'quantidade'] = listofwords.loc[listofwords['palavra'] == w, 'quantidade']  + 1


listofwords.to_excel('estudostopwords.xlsx', index = False)
