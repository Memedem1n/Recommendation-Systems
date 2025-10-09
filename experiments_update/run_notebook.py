from nbclient import NotebookClient
import nbformat
from pathlib import Path
import asyncio

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

nb_path = Path('experiments_update/Hybrid_Concern_Test/Hybrid_Concern_Test.ipynb')
nb = nbformat.read(nb_path.open(encoding='utf-8'), as_version=4)
client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(nb_path.parent)}})
client.execute()
nbformat.write(nb, nb_path.open('w', encoding='utf-8'))
print('Notebook executed.')
