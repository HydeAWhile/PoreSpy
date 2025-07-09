.. py:function:: {{ name }}({{ args }}){% if obj.return_annotation %} -> {{ obj.return_annotation }}{% endif %}

{% if docstring %}
   {{ docstring|indent(3) }}
{% endif %}
