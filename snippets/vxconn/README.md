# vxconn
Installation:

    python setup.py install

Usage:

    >>> from pg_vxconn import pg_engine
    >>> pd.read_sql('select * from ...', pg_engine())
