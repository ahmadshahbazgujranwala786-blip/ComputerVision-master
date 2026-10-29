import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg, faiss
load_dotenv()
conn=psycopg.connect(host=os.getenv('DB_HOST','localhost'),port=os.getenv('DB_PORT','5432'),dbname=os.getenv('DB_NAME','attendance_db'),user=os.getenv('DB_USER','postgres'),password=os.getenv('DB_PASSWORD'))
with conn.cursor() as cur:
 cur.execute('SELECT student_id,name,phone_number FROM students ORDER BY student_id')
 rows=cur.fetchall()
print('DB_STUDENTS=',rows)
conn.close()
idx=faiss.read_index('data/face_index.faiss')
ids=faiss.vector_to_array(idx.id_map).astype('int64').tolist()
dbids={int(x[0]) for x in rows}
print('FAISS_COUNT=',idx.ntotal,'FAISS_IDS=',ids)
print('FAISS_WITHOUT_DB=',sorted(set(ids)-dbids))
print('DB_WITHOUT_FAISS=',sorted(dbids-set(ids)))
