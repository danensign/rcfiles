from sqlalchemy import create_engine

d = {'name': '',
     'pass': '',
     'host': '',
     'port': '',
     'db': ''}

engine = create_engine('postgresql+psycopg2://{name}:{password}@{host}:{port}/{db}'.format(d))
                                                                                    
