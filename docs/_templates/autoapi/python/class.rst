.. py:class:: {{ name }}({{ args }}){% if obj.bases %} -> {{ obj.bases|join(", ") }}{% endif %}

{% if docstring %}
   {{ docstring|indent(3) }}
{% endif %}

{% if children %}
{% set visible_children = children | selectattr("display") | list %}
{% for child in visible_children %}
{{ child.render() | indent(0) }}
{% endfor %}
{% endif %}