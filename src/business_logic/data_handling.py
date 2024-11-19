from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Initialize database connection
database_uri = 'mysql://delta:delta001@localhost/znjw?unix_socket=/var/run/mysqld/mysqld.sock'
db = create_engine(database_uri, pool_size=10, max_overflow=20)

# Create a configured "Session" class
Session = sessionmaker(bind=db)


def get_session():
    return Session()
