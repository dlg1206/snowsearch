# Commands

> Overview of all commands currently available

Ensure `--network=snowsearch-net` and `--env-file=<path/to/env/file>` are always present.

Docker: `docker run --rm -it --network=snowsearch-net --env-file=.env snowsearch`

```
usage: snowsearch [-h] [-c <path to config file>] [-l <log level>] [-s] {slr,snowball,search,inspect,rank,upload} ...

SnowSearch: AI-powered snowball systemic literature review assistant

positional arguments:
  {slr,snowball,search,inspect,rank,upload}
    slr                 Perform a complete literature search with search generation, rounds of snowballing, and final abstract LLM ranking.
    snowball            Perform snowballing using papers stored in the database without the initial OpenAlex search or LLM ranking
    search              Search the database for matching papers
    inspect             Get details about a paper
    rank                Rank papers that best match the provided search
    upload              Upload papers locally to the database

options:
  -h, --help            show this help message and exit

Configuration:
  -c, --config <path to config file>
                        Path to config file to use

Logging:
  -l, --log-level <log level>
                        Set log level (Default: INFO) (['INFO', 'WARN', 'ERROR', 'DEBUG'])
  -s, --silent          Run in silent mode
```

A [configuration file](../config.yaml) is also available to use. If no config file is provided, SnowSearch will look for
a `config.yaml` to use, otherwise will use default values. To use with docker, mount the file like so:

```bash
docker run --rm -it --network=snowsearch-net --env-file=.env -v "$(pwd)/config.yaml:/snowsearch/config.yaml" snowsearch <args>
```

## Table of Contents

### slr

Docker: `docker run --rm -it --network=snowsearch-net --env-file=.env snowsearch slr -h`

```
usage: snowsearch slr [-h] [--ignore-quota] [-zu <zotero-id> | -zg <group-id>] [-zc [<collection-id>]] [-q <query>] [-j <json-file-path> | --skip-ranking] <semantic-search>

Perform a complete literature search with search generation, rounds of snowballing, and final abstract LLM ranking.

positional arguments:
  <semantic-search>     Descriptive, natural language search for desired papers. i.e. "AI-driven optimization of renewable energy systems"

options:
  -h, --help            show this help message and exit
  --ignore-quota        Do not retry to process additional papers to meet the round paper quota
  -zu, --zotero-user-library <zotero-id>
                        Upload to personal Zotero library. The User ID can be found here: https://www.zotero.org/settings/keys
  -zg, --zotero-group-library <group-id>
                        Upload to a group Zotero library. The library must be private and user must have write access
  -zc, --zotero-collection [<collection-id>]
                        Collection ID to a specific collection
  -q, --query <query>   OpenAlex formatted query to use directly. See https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities#boolean-searches for formatting rules
  -j, --json <json-file-path>
                        Save the results to json file instead of printing to stdout
  --skip-ranking        Skip the final paper ranking using an LLM
```

To save the JSON file, mount a directory like so:

```bash
docker run --rm -it --network=snowsearch-net --env-file=.env -v "$(pwd)/out:/out" snowsearch slr <args> -j /out/output.json
```

### Snowball

Docker: `docker run --rm -it --network=snowsearch-net --env-file=.env snowsearch snowball -h`

```
usage: snowsearch snowball [-h] [-ss <semantic-search>] [--ignore-quota] [--no-limit] [-p <paper-titles> [<paper-titles> ...] | -i <csv-file-path>]

Perform snowballing using papers stored in the database without the initial OpenAlex search or LLM ranking

options:
  -h, --help            show this help message and exit
  -ss, --semantic-search <semantic-search>
                        Descriptive, natural language search for desired papers. i.e. "AI-driven optimization of renewable energy systems"
  --ignore-quota        Do not retry to process additional papers to meet the round paper quota
  --no-limit            Set no citation cap and process all new unprocessed papers
  -p, --papers <paper-titles> [<paper-titles> ...]
                        One or more paper titles to start with. i.e "Graph Attention Networks" "GINE"
  -i, --papers-input <csv-file-path>
                        Path to csv file with list of paper titles to start with
```

To read the csv file, mount a directory like so:

```bash
docker run --rm -it --network=snowsearch-net --env-file=.env -v "$(pwd)/in/input.csv:/in/input.csv" snowsearch snowball <args> -i /in/input.csv
```

Where `"$(pwd)/in/input.csv` is the absolute path to your csv file. The csv file should be formatted like so:

```csv
Title1
Title with spaces
"Title, with, commas"
...
```

### Search

Docker: `docker run --rm -it --network=snowsearch-net --env-file=.env snowsearch search -h`

```
usage: snowsearch search [-h] [-l <limit>] [-m <score>] [-j <json-file-path>] [-zu <zotero-id> | -zg <group-id>] [-zc [<collection-id>]] [--only-open-access] [--only-processed] [--order-by-abstract] [-e] <semantic-search>

Search the database for matching papers

positional arguments:
  <semantic-search>     Descriptive, natural language search for desired papers. i.e. "AI-driven optimization of renewable energy systems"

options:
  -h, --help            show this help message and exit
  -l, --limit <limit>   Limit the number of papers to return
  -m, --min-similarity-score <score>
                        Score between -1 and 1 to be the minimum similarity match to filter for
  -j, --json <json-file-path>
                        Save the results to json file instead of printing to stdout
  -zu, --zotero-user-library <zotero-id>
                        Upload to personal Zotero library. The User ID can be found here: https://www.zotero.org/settings/keys
  -zg, --zotero-group-library <group-id>
                        Upload to a group Zotero library. The library must be private and user must have write access
  -zc, --zotero-collection [<collection-id>]
                        Collection ID to a specific collection
  --only-open-access    Only return papers that are publicly accessible
  --only-processed      Only return papers that have been successfully processed with Grobid
  --order-by-abstract   Order by abstract similarity first
  -e, --exact-match     Title must contain the search query exactly (case insensitive).
```

To save the JSON file, mount a directory like so:

```bash
docker run --rm -it --network=snowsearch-net --env-file=.env -v "$(pwd)/out:/out" snowsearch search <args> -j /out/output.json
```

### Inspect

Docker: `docker run --rm -it --network=snowsearch-net --env-file=.env snowsearch inspect -h`

```
usage: snowsearch inspect [-h] <title-of-paper>

Get details about a paper

positional arguments:
  <title-of-paper>  Print details of a given paper

options:
  -h, --help        show this help message and exit
```

### Rank

Docker: `docker run --rm -it --network=snowsearch-net --env-file=.env snowsearch rank -h`

```
usage: snowsearch rank [-h] [-l <limit>] [-m <score>] [-j <json-file-path>] [-zu <zotero-id> | -zg <group-id>] [-zc [<collection-id>]] [-p <paper-titles> [<paper-titles> ...] | -i <csv-file-path>] <semantic-search>

Rank papers that best match the provided search

positional arguments:
  <semantic-search>     Descriptive, natural language search for desired papers. i.e. "AI-driven optimization of renewable energy systems"

options:
  -h, --help            show this help message and exit
  -l, --limit <limit>   Limit the number of papers to return
  -m, --min-similarity-score <score>
                        Score between -1 and 1 to be the minimum similarity match to filter for
  -j, --json <json-file-path>
                        Save the results to json file instead of printing to stdout
  -zu, --zotero-user-library <zotero-id>
                        Upload to personal Zotero library. The User ID can be found here: https://www.zotero.org/settings/keys
  -zg, --zotero-group-library <group-id>
                        Upload to a group Zotero library. The library must be private and user must have write access
  -zc, --zotero-collection [<collection-id>]
                        Collection ID to a specific collection
  -p, --papers <paper-titles> [<paper-titles> ...]
                        One or more paper titles to start with. i.e "Graph Attention Networks" "GINE"
  -i, --papers-input <csv-file-path>
                        Path to csv file with list of paper titles to start with
```

To save the JSON file, mount a directory like so:

```bash
docker run --rm -it --network=snowsearch-net --env-file=.env -v "$(pwd)/out:/out" snowsearch rank <args> -j /out/output.json
```

To read the csv file, mount a directory like so:

```bash
docker run --rm -it --network=snowsearch-net --env-file=.env -v "$(pwd)/in/input.csv:/in/input.csv" snowsearch rank <args> -i /in/input.csv
```

Where `"$(pwd)/in/input.csv` is the absolute path to your csv file. The csv file should be formatted like so:

```csv
Title1
Title with spaces
"Title, with, commas"
...
```

### Upload

Docker: `docker run --rm -it --network=snowsearch-net --env-file=.env snowsearch upload -h`

```
usage: snowsearch upload [-h] (-f <pdf-path> | -d <pdf-directory-path>)

Upload papers locally to the database

options:
  -h, --help            show this help message and exit
  -f, --file <pdf-path>
                        Path to pdf file to upload and process
  -d, --directory <pdf-directory-path>
                        Path to root directory of pdf files to upload
```

To save the pdf file, mount a file like so:

```bash
docker run --rm -it --network=snowsearch-net --env-file=.env -v "$(pwd)/in/input.pdf:/in/input.pdf" snowsearch snowball <args> -f /in/input.pdf
```

Where `"$(pwd)/in/input.pdf` is the absolute path to your pdf file.

To save the directory of pdfs, mount a directory like so:

```bash
docker run --rm -it --network=snowsearch-net --env-file=.env -v "$(pwd)/in:/in" snowsearch snowball <args> -d /in
```

Where `"$(pwd)/in` is the absolute path the directory of pdfs