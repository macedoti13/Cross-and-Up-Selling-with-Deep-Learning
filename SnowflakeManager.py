from pydantic_settings import BaseSettings
from typing import Tuple, Optional, List
import snowflake.connector as sc
import pandas as pd

class SnowflakeManagerSettings(BaseSettings):
    SNOWFLAKE_ACCOUNT: str
    SNOWFLAKE_USER: str
    SNOWFLAKE_PRIVATE_KEY_FILE: str
    SNOWFLAKE_PRIVATE_KEY_FILE_PWD: str
    SNOWFLAKE_DATABASE: str
    SNOWFLAKE_SCHEMA: str
    SNOWFLAKE_WAREHOUSE: Optional[str] = None
    
    class Config:
        env_file = "config/.env"
        
    def get_conn_params(self):
        return {
            'account': self.SNOWFLAKE_ACCOUNT,
            'user': self.SNOWFLAKE_USER,
            'private_key_file': self.SNOWFLAKE_PRIVATE_KEY_FILE,
            'private_key_file_pwd': self.SNOWFLAKE_PRIVATE_KEY_FILE_PWD,
            'warehouse': self.SNOWFLAKE_WAREHOUSE,
            'database': self.SNOWFLAKE_DATABASE,
            'schema': self.SNOWFLAKE_SCHEMA
        }

class SnowflakeManager:
    def __init__(self):
        settings = SnowflakeManagerSettings()
        self.conn_params = settings.get_conn_params()
        self.conn = None
    
    def connect(self) -> None:
        if not isinstance(self.conn, sc.SnowflakeConnection):
            try:
                self.conn = sc.connect(**self.conn_params)
            except sc.Error as e:
                print(f"Error connecting to Snowflake: {e}")
                raise
    
    def disconnect(self) -> None:
        if self.conn:
            try:
                self.conn.close()
            except sc.Error as e:
                print(f"Error disconnecting from Snowflake: {e}")
            finally:
                self.conn = None
    
    def execute_query(self, query: str) -> sc.cursor:
        self.connect()
        try:
            cursor =  self.conn.cursor() 
            cursor.execute(query)
            return cursor
        except sc.Error as e:
            print(f"Error executing query: {e}")
            raise
        
    def run_query(self, query: str) -> Optional[List[Tuple]]:
        cursor = self.execute_query(query)
        if cursor:
            try:
                return cursor.fetchall()

            except sc.Error as e:
                print(f"Error running query: {e}")
                return None
        else:
            print("No cursor available to run query.")
            return None
        
    def run_query_pandas_all_data(self, query: str) -> Optional[pd.DataFrame]:
        cursor = self.execute_query(query)
        if cursor:
            try:
                return cursor.fetch_pandas_all()
            
            except sc.Error as e:
                print(f"Error fetching data: {e}")
                return None
        else:
            print("No cursor available to fetch data.")
            return None

    def run_query_pandas_in_batches(self, query: str, batch_size: int = 100000) -> Optional[pd.DataFrame]:
        cursor = self.execute_query(query)
        if cursor:
            try:
                return cursor.fetch_pandas_batches(batch_size)
            
            except sc.Error as e:
                print(f"Error fetching data in batches: {e}")
                return None
        else:
            print("No cursor available to fetch data.")
            return None

    def fetch_selling_data(self, sample: Optional[int] = None) -> Optional[pd.DataFrame]:
        if sample:
            query = f""" SELECT * FROM PUC_VENDAS LIMIT {sample} """
        else:
            query = """ SELECT * FROM PUC_VENDAS """
        cursor = self.execute_query(query)
        if cursor:
            try:
                return cursor.fetch_pandas_all()

            except sc.Error as e:
                print(f"Error fetching data from selling data: {e}")
                return None

        else:
            print("No cursor available to fetch data.")
            return None
    
    def fetch_campaign_data(self) -> Optional[pd.DataFrame]:
        query =  """ SELECT * FROM PUC_CAMPANHAS """
        cursor = self.execute_query(query)
        if cursor:
            try:
                return cursor.fetch_pandas_all()
            
            except sc.Error as e:
                print(f"Error fetching data from selling data: {e}")
                return None
        else:
            print("No cursor available to fetch data.")
            return None
        
    def get_data_dict_selling_data(self) -> None:
        cursor = self.execute_query("SELECT GET_DDL('TABLE', 'PUC_VENDAS');")
        if cursor:
            try:
                return cursor.fetchall()[0][0]
            
            except sc.Error as e:
                print(f"Error fetching data dictionary: {e}")
                return None
        else:
            print("No cursor available to fetch data.")
            return None
        
    def get_data_dict_campaign_data(self) -> None:
        cursor = self.execute_query("SELECT GET_DDL('TABLE', 'PUC_CAMPANHAS');")
        if cursor:
            try:
                return cursor.fetchall()[0][0]
            
            except sc.Error as e:
                print(f"Error fetching data dictionary: {e}")
                return None
        else:
            print("No cursor available to fetch data.")
            return None
