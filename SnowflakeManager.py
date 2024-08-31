from pydantic_settings import BaseSettings
from typing import Tuple, Optional, Iterator
import snowflake.connector as sc

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
            cursor = self.conn.cursor()
            cursor.execute(query)
            return cursor
        except sc.Error as e:
            print(f"Error executing query: {e}")
            raise

    def run_query(self, query: str) -> Optional[Iterator[Tuple]]:
        cursor = self.execute_query(query)
        if cursor:
            try:
                while True:
                    rows = cursor.fetchmany(1000)
                    if not rows:
                        break
                    for row in rows:
                        yield row
            except sc.Error as e:
                print(f"Error running query: {e}")
                return None
        else:
            print("No cursor available to run query.")
            return None

    def fetch_selling_data(self) -> Optional[Iterator[Tuple]]:
        query = """SELECT * FROM PUC_VENDAS"""
        return self.run_query(query)
    
    def fetch_campaign_data(self) -> Optional[Iterator[Tuple]]:
        query = """SELECT * FROM PUC_CAMPANHAS"""
        return self.run_query(query)
