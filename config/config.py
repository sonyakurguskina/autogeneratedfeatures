import os
from dotenv import load_dotenv


class Config:
    def __init__(self):
        load_dotenv()
        self.pg_user = os.getenv("PG_USER")
        self.pg_password = os.getenv("PG_PASSWORD")
        self.pg_host = os.getenv("PG_HOST")
        self.pg_port = os.getenv("PG_PORT")
        self.pg_database = os.getenv("PG_DATABASE")
