API reference
=============

Everything below is importable directly from ``neurozarr``.

At a glance
-----------

Building a store:

.. autosummary::

   neurozarr.Repo
   neurozarr.Repo.create
   neurozarr.Repo.open
   neurozarr.Repo.create_subject
   neurozarr.Repo.set_attrs
   neurozarr.Repo.ingest
   neurozarr.Repo.save
   neurozarr.Repo.transaction
   neurozarr.Repo.abort
   neurozarr.Repo.migrate
   neurozarr.Repo.subjects
   neurozarr.Repo.subject
   neurozarr.Repo.find
   neurozarr.Repo.history
   neurozarr.Repo.tag
   neurozarr.Repo.tags
   neurozarr.Subject.add_visit
   neurozarr.Subject.set_attrs
   neurozarr.Subject.visits
   neurozarr.Subject.visit
   neurozarr.Subject.recordings
   neurozarr.Subject.tables
   neurozarr.Subject.arrays
   neurozarr.Subject.external_files
   neurozarr.Visit.add_recording
   neurozarr.Visit.add_behavioral_table
   neurozarr.Visit.add_derivative
   neurozarr.Visit.add
   neurozarr.Visit.add_array
   neurozarr.Visit.set_attrs
   neurozarr.Visit.recording

Reading data back:

.. autosummary::

   neurozarr.RecordingView.data
   neurozarr.RecordingView.raw
   neurozarr.RecordingView.channels
   neurozarr.RecordingView.events
   neurozarr.RecordingView.ch_names
   neurozarr.RecordingView.sfreq
   neurozarr.RecordingView.duration
   neurozarr.RecordingView.shape
   neurozarr.RecordingView.array
   neurozarr.RecordingView.set_attrs
   neurozarr.TableView.df
   neurozarr.TableView.columns
   neurozarr.TableView.set_attrs
   neurozarr.ArrayView.data
   neurozarr.ArrayView.set_attrs
   neurozarr.ExternalFileView.set_attrs

The item stream a reader yields and the writer consumes:

.. autosummary::

   neurozarr.Entities
   neurozarr.Recording
   neurozarr.Table
   neurozarr.Attrs
   neurozarr.ExternalFile
   neurozarr.Array
   neurozarr.Reader
   neurozarr.BidsReader
   neurozarr.ManifestReader
   neurozarr.Writer
   neurozarr.CodecConfig
   neurozarr.ExistingPolicy

Supporting a new source format:

.. autosummary::

   neurozarr.register_reader
   neurozarr.available_readers
   neurozarr.reader_for
   neurozarr.open_reader
   neurozarr.register_format
   neurozarr.registered_formats
   neurozarr.UnclaimedPolicy

When something is wrong:

.. autosummary::

   neurozarr.NeurozarrError
   neurozarr.ValidationError
   neurozarr.SchemaVersionError
   neurozarr.StoreIntegrityError
   neurozarr.WriteConflictError
   neurozarr.UnsupportedFormatError
   neurozarr.ValidationReport
   neurozarr.ValidationIssue
   neurozarr.Severity

Everything else:

.. autosummary::

   neurozarr.parallel.convert_parallel
   neurozarr.storage_from
   neurozarr.validate_source
   neurozarr.inspect_source
   neurozarr.inspect_store
   neurozarr.verify
   neurozarr.export_bids
   neurozarr.set_verbosity

Repo, Subject, Visit
--------------------

.. autoclass:: neurozarr.Repo
   :members:

.. autoclass:: neurozarr.Subject
   :members:

.. autoclass:: neurozarr.Visit
   :members:

Data model
----------

.. autoclass:: neurozarr.Entities
   :members:

.. autoclass:: neurozarr.Recording
   :members:

.. autoclass:: neurozarr.Table
   :members:

.. autoclass:: neurozarr.Attrs
   :members:

.. autoclass:: neurozarr.ExternalFile
   :members:

.. autoclass:: neurozarr.Array
   :members:

.. autoclass:: neurozarr.Reader
   :members:

Readers
-------

.. autoclass:: neurozarr.BidsReader
   :members:

.. autoclass:: neurozarr.ManifestReader
   :members:

.. autofunction:: neurozarr.register_reader

.. autofunction:: neurozarr.available_readers

.. autofunction:: neurozarr.reader_for

.. autofunction:: neurozarr.open_reader

.. autofunction:: neurozarr.register_format

.. autofunction:: neurozarr.registered_formats

.. autoclass:: neurozarr.UnclaimedPolicy
   :members:

Reading data back
------------------

.. autoclass:: neurozarr.RecordingView
   :members:

.. autoclass:: neurozarr.TableView
   :members:

.. autoclass:: neurozarr.ArrayView
   :members:

.. autoclass:: neurozarr.ExternalFileView
   :members:

Writing
-------

.. autoclass:: neurozarr.Writer
   :members:

.. autoclass:: neurozarr.CodecConfig
   :members:

.. autoclass:: neurozarr.ExistingPolicy
   :members:

Parallel conversion
-------------------

.. autofunction:: neurozarr.parallel.convert_parallel

Storage, validation, and verification
--------------------------------------

.. autofunction:: neurozarr.storage_from

.. autodata:: neurozarr.StorageTarget

.. autodata:: neurozarr.StorageFactory

.. autofunction:: neurozarr.set_verbosity

.. autofunction:: neurozarr.validate_source

.. autofunction:: neurozarr.inspect_source

.. autofunction:: neurozarr.inspect_store

.. autofunction:: neurozarr.verify

.. autofunction:: neurozarr.export_bids

Diagnostics
-----------

A :func:`~neurozarr.inspect_source` or :func:`~neurozarr.inspect_store` call
returns a report of structured issues rather than strings, so callers can filter
on ``code`` and ``severity`` instead of parsing messages.

.. autoclass:: neurozarr.ValidationReport
   :members:

.. autoclass:: neurozarr.ValidationIssue
   :members:

.. autoclass:: neurozarr.Severity
   :members:

Exceptions
----------

Everything neurozarr raises deliberately derives from
:class:`~neurozarr.NeurozarrError`, so one ``except`` clause catches the package
without also catching unrelated bugs. Several also derive from the built-in that
best describes them, so existing ``except ValueError`` handlers keep working.

.. autoexception:: neurozarr.NeurozarrError
   :members:

.. autoexception:: neurozarr.ValidationError
   :members:

.. autoexception:: neurozarr.SchemaVersionError
   :members:

.. autoexception:: neurozarr.StoreIntegrityError
   :members:

.. autoexception:: neurozarr.WriteConflictError
   :members:

.. autoexception:: neurozarr.UnsupportedFormatError
   :members:
