#  Hogyan reprodukáljuk a database appot

##  Setup:
Előkövetelmény, hogy fel legyen telepíttve a DOcker Engine, és a Docker composite.
Létre kell hozni egy projekt mappát. Ezt a következő parancsok futtatásával lehet megcsinálni:
```
$ mkdir docker_python_sql_tutorial
$ cd docker_python_sql_tutorial
$ mkdir app
$ mkdir database
```

##  PostgreSQL:
A PostgreSQL adatbázis Dockeren keresztül történő példányosításához a hivatalos image-et használunk. Ehhez a következő Dockerfile-t kell létrehozni az adatbázis mappán belül:
```
FROM postgres:latest
ENV POSTGRES_PASSWORD=secret
ENV POSTGRES_USER=username
ENV POSTGRES_DB=database
```

A fájl tartalma:

+ FROM: ez az utasítás azt a image-t azonosítja, amelyből az új image-t szeretnénk létrehozni. A postgres:latest-et választotuk, ami a hivatalos Docker Image a latest címkével, ami a legújabb verziót jelenti.
  
+ ENV: ezzel az utasítással különböző környezeti változókat adhatunk meg. Ehhez az image-hez megadtuk a POSTGRES_PASSWORD, POSTGRES_USER, POSTGRES_DB környezeti változókat.
  
+ COPY: a megadott create_fixtures.sql fájl egy adott mappában lévő fájlnak a létrehozott /docker-entrypoint-initb.d/ képbe történő másolására szolgál.
A /docker-entrypoint-initb.d/ mappában lévő fájl másolása nagyon hasznos, mert lehetővé teszi számunkra néhány inicializáló SQL parancs indítását. Ebben az esetben úgy döntöttem, hogy létrehozok egy egyszerű táblázatot két mezővel (lásd alább).

##Python Script:
Létrehozzuk az adatbázissal együttműködő Python szkriptet. A létrehozott szkript az app mappán belül található.
```
import time
import random

from sqlalchemy import create_engine

db_name = 'database'
db_user = 'username'
db_pass = 'secret'
db_host = 'db'
db_port = '5432'

# Connect to the database
db_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_user, db_pass, db_host, db_port, db_name)
db = create_engine(db_string)

```

Következőt csinálja a script:

+  Az SQLAlchemy számára szükséges kapcsolati karakterlánc létrehozásához szükséges paraméterek határozza meg, amely lehetővé teszi számunkra, hogy kapcsolatot létesítsünk a PostgreSQL-hez. Mint látható, a db_name, db_user, db_pass ugyanazok, amelyeket korábban a PostgreSQL Dockerfile-ban környezeti változóként jeleztünk. A db_host változót később fogjuk elmagyarázni, a db_port pedig az alapértelmezett PostgreSQL port.

## Ahhoz, hogy a python szkript működjön, a requirements.txt fájlban megadtuk a függőségeket.

A következők szoftverek és csomagok szükségesek a rekreációhoz:

+  Python 3.8
+  Docker
+  sqlalchemy,psycopg2,PyVCF,numpy,pandas,tensorflow,scikit-learn,umap,statsmodels,pyearth,xgboost,lightgbm,catboost,matplotlib,minisom python csomagok.

## Létrehozunk egy Docker-képet a Dockerfile-on keresztül az app mappán belül.

```
FROM python:latest
WORKDIR /code
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY app.py app.py
CMD ["python", "-u", "app.py"]
```

## A Docker Image és a script együtműködése

A bemutató utolsó lépése az általunk készített két kép egyesítése. Ennek legelegánsabb módja, ha létrehozunk egy docker-compose.yml fájlt a projekt gyökerében.

```
version: "3.8"
services:
  app :
    build: ./app/
  db:
    build: ./database/
```

Ebben két konténert deklarálunk az alkalmazásunkon belül:

    app: az, amelyik a /app/Dockerfile-ban van definiálva.
    db: az, amelyik a /database/Dockerfile-ban található.

A szolgáltatások nevével értjük a db_host='db' az app.py-n belül , a Docker az, ami a két image közötti hálózatot kezeli indítás után, így a db-et az adatbázis szolgáltatás hostneveként fordítja le.

Az utolsó parancs, ami ahhoz szükséges, hogy minden fusson, a docker-compose up - build. Ez felépíti a képeket, majd elindítja a konténereket.
