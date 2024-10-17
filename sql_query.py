#Classe para baixar os dados de imterrupções do RCP
import pandas as pd
import oracledb
from sqlalchemy import create_engine, text


class ImportRcp():
    #Consulta SQL
    def sql_inter():
        return(
            '''
            SELECT
                    A1 AS "Problem",  
                    A2 AS "f",
                    A3  AS "ac",  
                    A4 AS "Description",
                    A5 AS "Technical Responsability",
                    A6 AS "Delay Time",
                    A7 AS "Chargeable"




                FROM
                    evento_falhas wp
                    WHERE 1 = 1  
                    AND DATA >= add_months(sysdate, -72)
                    AND DATA <= add_months(sysdate, -10)
           
            '''
        )
   
    def query_data():
       
        oracledb.init_oracle_client()
        #python 3.11 (Oracle_db)
        engine = create_engine('')


        sqlInter = ImportRcp.sql_inter()
        print ('EXECUTA CONSULTA')
               
        #Database query
        dfquery = pd.read_sql(sql = text(sqlInter), con = engine.connect())


        #dfquery = dfquery[dfquery["Chargeable"] != ""]
        dfquery = dfquery[(dfquery["Technical Responsability"] != "COD2") & (dfquery["Technical Responsability"] != "COD0")]
       
        return dfquery
