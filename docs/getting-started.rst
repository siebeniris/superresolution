Getting started
===============
The superresolution project is build after the
`Cookiecutter data science <https://drivendata.github.io/cookiecutter-data-science/>`_ project template.   
We use `DVC <https://dvc.org/>`_ to manage data, store artifacts and create pipelines.
on top of `DVC <https://dvc.org/>`_, we use `MLFlow <https://www.mlflow.org/docs/latest/index.html>`_
to log our model files, track `experiments <10.195.1.42:5000>`_ and execute revisions.
We combine `MLFlow <https://www.mlflow.org/docs/latest/index.html>`_
with `conda <https://docs.conda.io/en/>`_ to create our python environment.


Execution Endpoint Requirements
-------------------------------
On a execution endpoint multiple things need to be configured.

- An endpoint must be configured to be able to log in to the data server with ssh.

- The environment variable MLFLOW_TRACKING_URI must be configured to the MLFlow experiment tracking server.

- `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ must be installed to execute MLFlow runs.

Development Requirements
------------------------
Includes all requirements for an execution endpoint.
Install dvc[all] to manually submit data files.


Time Tracking
-------------
We use `GTT <https://github.com/kriskbx/gitlab-time-tracker>`_ and the gitlab time tracker api to track spend time.







