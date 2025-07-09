{{ objname | escape | underline}}

.. py:module:: {{ name }}

{% if summary %}
.. autoapi-nested-parse::

   {{ summary }}

{% endif %}

{% block content %}
{% if children %}
{% set visible_children = children | selectattr("display") | list %}
{% set visible_classes = visible_children | selectattr("type", "equalto", "class") | list %}
{% set visible_functions = visible_children | selectattr("type", "equalto", "function") | list %}
{% set visible_attributes = visible_children | selectattr("type", "equalto", "data") | list %}

{% if visible_classes %}
Classes
-------

.. toctree::
   :maxdepth: 1

{% for item in visible_classes %}
   {{ item.name }} <#{{ item.id }}>
{%- endfor %}

{% for item in visible_classes %}
{{ item.render() | indent(0) }}
{% endfor %}
{% endif %}

{% if visible_functions %}
Functions
---------

.. toctree::
   :maxdepth: 1

{% for item in visible_functions %}
   {{ item.name }} <#{{ item.id }}>
{%- endfor %}

{% for item in visible_functions %}
{{ item.render() | indent(0) }}
{% endfor %}
{% endif %}

{% if visible_attributes %}
Module Attributes
-----------------

{% for item in visible_attributes %}
{{ item.render() | indent(0) }}
{% endfor %}
{% endif %}

{% endif %}
{% endblock %}