#Hogyan reprodukáljuk a database appot

##Setup:
Előkövetelmény, hogy fel legyen telepíttve a DOcker Engine, és a Docker composite.
Létre kell hozni egy projekt mappát. Ezt a következő parancsok futtatásával lehet megcsinálni:
```s
$ mkdir docker_python_sql_tutorial
$ cd docker_python_sql_tutorial
$ mkdir app
$ mkdir database
```

##PostgreSQL:
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