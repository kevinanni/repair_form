from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Initialize database connection
database_uri = 'mysql://root:root@localhost/znjw'
db = create_engine(database_uri)

# Create a configured "Session" class
Session = sessionmaker(bind=db)

# Create a session instance
session = Session()
