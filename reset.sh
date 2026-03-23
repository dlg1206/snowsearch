docker compose down abstract_db
docker volume rm snowsearch_abstract_db
docker compose -f compose.yaml -f override/database.yaml up abstract_db -d