from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Simplified database URI without unix_socket
database_uri = 'mysql+pymysql://delta:delta001@localhost/znjw'
db = create_engine(database_uri, pool_size=10, max_overflow=20)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db)

class SessionContextManager:
    def __init__(self):
        self.session = SessionLocal()

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()

def get_session():
    return SessionContextManager()

# Test the connection
if __name__ == "__main__":
    print("start...")
    with get_session() as session:
        print("Connection successful!")




