# Steps for creating data image and data deployment in Portainer

1. Train models and store `projects.cfg` and `data/` directory to `~/annif-projects`.

2. Build a data image (which [could also be versioned with a custom tag](https://docs.docker.com/engine/reference/commandline/build/#tag-an-image--t), the default tag is `latest`):

    ```docker build -t quay.io/natlibfi/annif-data -f Dockerfile-data ~/annif-projects```

    Here the data for models are included in the image, but the corpora are not (even if they happen to reside in `~/annif-projects`).

3. Push the image to https://quay.io/repository/natlibfi/annif-data repository: 

    ```docker push quay.io/natlibfi/annif-data```

4. In the [Services view of Portainer](https://portainer.kansalliskirjasto.fi/#/services) first select the data service (`annif-test_data`) and update it using the GUI button. Select to pull the latest image version when asked. Then, to make Annif use the new data, similarly update the `annif-test_gunicorn_server` service (now pulling the latest image is not necessary).
