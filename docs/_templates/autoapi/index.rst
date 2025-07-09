API Reference
=============

.. toctree::
   :maxdepth: 4
   :titlesonly:

{% for page in pages %}
   {{ page.include_path }}
{%- endfor %}