# SRIModules

Authors:
* Héctor Miguel Rodríguez Sosa
* Sebastián Suárez Gómez

### Description
In this project, there are three models. Which are the Boolean Model, Extended Boolean Model(EBM)
with modifications and the actual Extended Boolean Model(EBM). These three models
use the *beir/trec-covid* dataset from the _ir-datasets_(python library) to make its recoveries.

**Boolean Model**(*boolean_model.py*): It is a simple and intuitive approach to document retrieval, 
where documents are considered to be in the presence or absence of a query term. 
This model is based on the concept of Boolean algebra, where documents are represented 
as sets of terms, and queries are formulated using Boolean operators (AND, OR, NOT)
to specify the desired documents.

**Extended Boolean Model**(*canonical.py*): The Extended Boolean Model (EBM) of Information Retrieval is a
query model that extends the traditional Boolean model to allow for more complex queries.
This model is designed to enhance the ability of information retrieval systems to handle
complex queries by introducing additional operators and constructs beyond the simple
AND, OR, and NOT operations of the basic Boolean model.


**Modified Extended Boolean Model**(*querier.py*): This model is kind of similar to the original EBM.
In first place, it recovers the documents with boolean model, then expands the query
with the KL divergence, after that recovers the documents with the expanded query.
All recovered documents scores are multiplied by a value that indicates the relevance
of the documents in previous queries.


### How to use
This project is executed in a *gui.ipynb*. Before to run this file, you need to install
all the modules used in the project.

Execute this in your command line:

``pip install -r requirements.txt``


### Results

| Model      | Precision | Recall   | F1 Score  | F3 Score  | R-Precision |
|------------|-----------|----------|-----------|-----------|-------------|
| Boolean    | 0.000000  | 0.00000  | 0.000000  | 0.000000  | 0.0         |
| Extended   | 0.129107  | 0.07121  | 0.081234  | 0.072301  | 0.0         |
| Cannonical | 0.004669  | 0.65069  | 0.009119  | 0.039561  | 0.0         |



