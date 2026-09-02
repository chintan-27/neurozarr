from dataclasses import dataclass, field
from typing import Iterator, Protocol

import mne
import pandas as pd

from .entities import Entities


@dataclass
class Recording:
	entities: Entities
	raw: "mne.io.BaseRaw"
	meta: dict = field(default_factory=dict)
	prefix: tuple = ()


@dataclass
class Table:
	entities: Entities
	name: str
	df: pd.DataFrame
	meta: dict = field(default_factory=dict)
	prefix: tuple = ()


@dataclass
class Attrs:
	path: tuple
	attrs: dict


class Reader(Protocol):
	def read(self) -> Iterator[Recording | Table | Attrs]: ...
