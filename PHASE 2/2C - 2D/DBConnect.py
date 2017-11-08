# -*- coding: utf-8 -*-
#!/usr/bin/python
import psycopg2
import sys
import pprint

con_str = "host='localhost' dbname ='mwdphase2' user='postgres' password='manoj123'"

class DBConnect:
    test = ""
    conn = None
    cursor = None
    def __init__(self):
        self.connect()
        
    def connect(self):
        #print "Connecting to database\n	->%s" % (con_str)
        #print "Connected"
        self.conn = psycopg2.connect(con_str)
        self.cursor = self.conn.cursor()
        #cursor.execute("select * from mlmovies limit 10");
        #records = cursor.fetchall()
        #print "Records:  "
        #pprint.pprint(records)
        
    def executeQuery(self,query):
        self.cursor.execute(query);
        records = self.cursor.fetchall()
        #print "Records executeQuery:  "
        #pprint.pprint(records)
        return records
        









    

