from pathlib import Path
import faiss
root=Path('.')
for p in root.rglob('*.faiss'):
 try:
  idx=faiss.read_index(str(p))
  ids=faiss.vector_to_array(idx.id_map).astype('int64').tolist() if hasattr(idx,'id_map') else []
  print(f'{p} | count={idx.ntotal} | ids={ids} | bytes={p.stat().st_size}')
 except Exception as e: print(f'{p} | ERROR {e}')
