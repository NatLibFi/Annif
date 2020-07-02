## This branch contains files used for API instances of Annif

The components of the API instances are combined in Docker Stacks, which are set up using Portainer GUI. 
The Stacks do not automatically follow the docker-compose files in this branch, the files are here to allow easier referencing to them. 

The content for the production API web pages are in directories with the name of the service domain. 
Testing domain [ai.dev.finto.fi](https://ai.dev.finto.fi) uses the content of `ai.finto.fi` directory and [annif.dev.finto.fi](https://annif.dev.finto.fi) uses the content of `api.annif.org` directory.


## Data deployment for the APIs

The directory `/srv/annif-data/{service-domain}` at annif-kk is transferred/synced to the `data` volume of the Stack corresponding to the service domain. 

1. In Portainer Stacks view open the Stack that corresponds to the API instance for which the deployment is being performed.
  
2. Begin data sync by starting the container `{stack-name}_modeldata-sync` by scaling its desired number (under the column "Scheduling Mode") from 0 to 1 (or from 1 to 2 etc. if needed). A full transfer takes about half hour. Sync progress can be monitored in Graylog (in Annif dashboards there is a separate panel showing logs from the sync containers).

3. When the sync is finished (status of the modeldata-sync container is "complete"), put the new data in use by restarting Annif container by selecting its checkbox from the container list and choose update. It is not necessary to "pull latest image".


Webpage content for the APIs can be similarly updated using the "{stack-name}_webdata-sync" container, which pulls the directories from this branch to `webdata` volume using svn client. The modified content is used without restarting NGINX. The webdata for test instances are pulled automatically on each commit as Portainer listens to webhooks from GitHub.
