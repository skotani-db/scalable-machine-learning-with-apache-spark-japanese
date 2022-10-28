# Databricks notebook source
# MAGIC %pip install \
# MAGIC git+https://github.com/databricks-academy/dbacademy-gems@e8183eed9481624f25b34436810cf6666b4438c0 \
# MAGIC git+https://github.com/databricks-academy/dbacademy-rest@9f11b96e8d7e641f6141c7da0acd90a975075a9f \
# MAGIC git+https://github.com/databricks-academy/dbacademy-helper@e0e819d661e6bf972c6fb1872c1ae1a7d2a74b23 \
# MAGIC --quiet --disable-pip-version-check

# COMMAND ----------

# MAGIC %run ./_dataset_index

# COMMAND ----------

from dbacademy_gems import dbgems
from dbacademy_helper import DBAcademyHelper, Paths

helper_arguments = {
    "course_code" : "sml",             # The abreviated version of the course
    "course_name" : "scalable-machine-learning-with-apache-spark",      # The full name of the course, hyphenated
    "data_source_name" : "scalable-machine-learning-with-apache-spark", # Should be the same as the course
    "data_source_version" : "v02",     # New courses would start with 01
    "enable_streaming_support": False, # This couse uses stream and thus needs checkpoint directories
    "install_min_time" : "2 min",      # The minimum amount of time to install the datasets (e.g. from Oregon)
    "install_max_time" : "5 min",      # The maximum amount of time to install the datasets (e.g. from India)
    "remote_files": remote_files,      # The enumerated list of files in the datasets
}
