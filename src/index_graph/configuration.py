# index_configuration.py

from dataclasses import dataclass, field
from typing import Annotated
from src.shared.configuration import BaseConfiguration
DEFAULT_DOCS_FILE = "src/sample_docs.json"
@dataclass(kw_only=True)
class IndexConfiguration(BaseConfiguration):
    """
    Configuration class tailored for the Index Graph,
    inheriting common parameters from BaseConfiguration.
    """

    docs_file: str = field(
        default= DEFAULT_DOCS_FILE,
        metadata={"description": "Path to a JSON file containing documents."}
    )