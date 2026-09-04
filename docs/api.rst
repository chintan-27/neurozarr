API reference
=============

Everything below is importable directly from ``bidszarr``.

Repo, Subject, Visit
--------------------

.. autoclass:: bidszarr.Repo
   :members:

.. autoclass:: bidszarr.Subject
   :members:

.. autoclass:: bidszarr.Visit
   :members:

Data model
----------

.. autoclass:: bidszarr.Entities
   :members:

.. autoclass:: bidszarr.Recording
   :members:

.. autoclass:: bidszarr.Table
   :members:

.. autoclass:: bidszarr.Attrs
   :members:

.. autoclass:: bidszarr.Reader
   :members:

Readers
-------

.. autoclass:: bidszarr.BidsReader
   :members:

.. autoclass:: bidszarr.ManifestReader
   :members:

Reading data back
------------------

.. autoclass:: bidszarr.RecordingView
   :members:

.. autoclass:: bidszarr.TableView
   :members:

Writing
-------

.. autoclass:: bidszarr.Writer
   :members:

.. autoclass:: bidszarr.CodecConfig
   :members:

Parallel conversion
-------------------

.. autofunction:: bidszarr.parallel.convert_parallel

Storage, validation, and verification
--------------------------------------

.. autofunction:: bidszarr.storage_from

.. autofunction:: bidszarr.set_verbosity

.. autofunction:: bidszarr.validate_source

.. autofunction:: bidszarr.verify

.. autofunction:: bidszarr.export_bids
