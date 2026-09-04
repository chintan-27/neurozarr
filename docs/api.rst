API reference
=============

Everything below is importable directly from ``neurozarr``.

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

.. autoclass:: neurozarr.Reader
   :members:

Readers
-------

.. autoclass:: neurozarr.BidsReader
   :members:

.. autoclass:: neurozarr.ManifestReader
   :members:

Reading data back
------------------

.. autoclass:: neurozarr.RecordingView
   :members:

.. autoclass:: neurozarr.TableView
   :members:

Writing
-------

.. autoclass:: neurozarr.Writer
   :members:

.. autoclass:: neurozarr.CodecConfig
   :members:

Parallel conversion
-------------------

.. autofunction:: neurozarr.parallel.convert_parallel

Storage, validation, and verification
--------------------------------------

.. autofunction:: neurozarr.storage_from

.. autofunction:: neurozarr.set_verbosity

.. autofunction:: neurozarr.validate_source

.. autofunction:: neurozarr.verify

.. autofunction:: neurozarr.export_bids
