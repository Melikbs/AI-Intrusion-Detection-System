import time
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

DATABASE_URL = "postgresql+psycopg2://clouduser:cloudpass@postgres/cloud_ids"

# Retry until database is ready
max_retries = 10
retry_delay = 5  # seconds
for i in range(max_retries):
    try:
    #create engine
        engine = create_engine(DATABASE_URL)
        engine.connect()  # Test connection
        print("Database connected!")
        break
    except OperationalError:
        print(f"Database not ready, retrying in {retry_delay}s... ({i+1}/{max_retries})")
        time.sleep(retry_delay)
else:
    raise Exception("Could not connect to the database after several retries.")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, nullable=False)
    alert_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    protocol_type = Column(String, nullable=False)  # <-- String now
    service = Column(String, nullable=False)       # <-- String now
    src_bytes = Column(Integer, nullable=False)
    dst_bytes = Column(Integer, nullable=False)

def init_db():
    Base.metadata.create_all(bind=engine)

